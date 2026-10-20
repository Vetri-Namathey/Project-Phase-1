import { useCallback, useEffect, useRef, useState } from 'react'
import AnalystChat from './AnalystChat'
import LoadingCard from './LoadingCard'

// Upload an image or a short video. The backend scores it (/api/infer, /api/infer-video)
// and, on demand, explains one image or one frame (/api/explain-image, /api/explain-frame).
// A video is paused at its first detection and explained there; the timeline jumps to any frame.

const IMAGE_VIEWS = [
  ['detection', 'Detection'],
  ['fused', 'Anomaly score'],
  ['disagreement', 'Head disagreement'],
  ['input', 'Input'],
]

const OLD_SERVER = 'The server that is running is an older version of server.py (its reply has no per-frame data). '
  + 'Stop it, start server.py again, and re-upload. The video below still works, without the timeline and explanations.'

async function readError(r) {
  const body = await r.json().catch(() => ({}))
  return body.detail || `server error ${r.status}`
}

function Timeline({ info, threshold, current, onPick }) {
  const ref = useRef(null)
  const pick = e => {
    const rect = ref.current.getBoundingClientRect()
    const f = Math.min(info.length - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * info.length)))
    onPick(f)
  }
  return (
    <div className="tl" aria-label="Detections over time">
      <div className="tl-bars" ref={ref} onClick={pick} role="slider" aria-valuemin={0} aria-valuemax={info.length - 1}
        aria-valuenow={current} tabIndex={0}
        onKeyDown={e => {
          if (e.key === 'ArrowLeft') onPick(Math.max(0, current - 1))
          if (e.key === 'ArrowRight') onPick(Math.min(info.length - 1, current + 1))
        }}>
        {info.map((f, i) => (
          <div key={i} className={`tl-bar${f.boxes > 0 ? ' hit' : ''}${i === current ? ' now' : ''}`}
            style={{ height: `${Math.max(4, f.max * 100)}%` }} title={`frame ${i} · max score ${f.max.toFixed(2)}`} />
        ))}
        <div className="tl-thr" style={{ bottom: `${threshold * 100}%` }} />
      </div>
      <div className="tl-legend mono">
        <span><i className="hit" /> frame with a box</span>
        <span><i /> no box</span>
        <span>line = detection threshold {threshold.toFixed(3)}</span>
      </div>
    </div>
  )
}

export default function LiveAnalyzer() {
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const [drag, setDrag] = useState(false)
  const [res, setRes] = useState(null)      // image result
  const [vid, setVid] = useState(null)      // video result
  const [view, setView] = useState('detection')
  const [explain, setExplain] = useState({ state: 'idle' })
  const [frame, setFrame] = useState(0)
  const [chatExpanded, setChatExpanded] = useState(false)
  const [preview, setPreview] = useState(null)   // the user's own file while it is being scored
  const videoRef = useRef(null)
  const reqId = useRef(0)
  const initialSeekDone = useRef(false)
  const seekSettled = useRef(false)

  const runExplain = useCallback(async (fetcher, question) => {
    const id = ++reqId.current
    setExplain({ state: 'loading', question })
    try {
      const r = await fetcher()
      if (!r.ok) throw new Error(await readError(r))
      const data = await r.json()
      if (id === reqId.current) setExplain({ state: 'ready', data, question })
    } catch (e) {
      if (id === reqId.current) setExplain({ state: 'error', error: `${e.message}. Is server.py running?`, question })
    }
  }, [])

  const explainFrame = useCallback((clip, f, fps) => {
    if (!clip) return
    setFrame(f)
    runExplain(
      () => fetch(`/api/explain-frame?clip=${encodeURIComponent(clip)}&frame=${f}`, { method: 'POST' }),
      `Why was frame ${f} (${(f / fps).toFixed(1)} s) flagged?`,
    )
  }, [runExplain])

  const seekTo = (f, fps) => {
    const v = videoRef.current
    if (!v) return
    v.pause()
    v.currentTime = (f + 0.5) / fps
  }

  // keep the local preview's object URL tidy
  useEffect(() => () => { if (preview?.url) URL.revokeObjectURL(preview.url) }, [preview?.url])

  const run = async file => {
    if (!file) return
    const isVideo = /^video\//.test(file.type)
    if (!isVideo && !/^image\//.test(file.type)) { setErr('Please choose a JPG/PNG image or an MP4 video.'); return }
    const kind = isVideo ? 'video' : 'image'
    reqId.current++
    initialSeekDone.current = false
    seekSettled.current = false
    setErr(null); setRes(null); setVid(null); setFrame(0)
    setBusy(kind)
    setExplain({ state: 'loading', question: `${isVideo ? 'Analyse' : 'Explain'} ${file.name}`,
      label: isVideo ? 'Waiting for the model to score the video…' : 'Waiting for the model…' })
    const url = URL.createObjectURL(file)
    setPreview({ url, kind, name: file.name, duration: null })
    if (isVideo) {   // read the length so the progress estimate is realistic
      const probe = document.createElement('video')
      probe.preload = 'metadata'
      probe.onloadedmetadata = () => setPreview(p => (p && p.url === url ? { ...p, duration: probe.duration } : p))
      probe.src = url
    }
    try {
      if (isVideo) {
        const r = await fetch('/api/infer-video', { method: 'POST', body: file, headers: { 'Content-Type': file.type } })
        if (!r.ok) throw new Error(await readError(r))
        const data = await r.json()
        const legacy = !Array.isArray(data.frame_info) || !data.clip
        if (legacy) setErr(OLD_SERVER)
        setVid({ ...data, frame_info: legacy ? [] : data.frame_info, first_detection: legacy ? null : data.first_detection, legacy })
        if (legacy) setExplain({ state: 'idle' })
      } else {
        const r = await fetch('/api/infer', { method: 'POST', body: file, headers: { 'Content-Type': file.type } })
        if (!r.ok) throw new Error(await readError(r))
        setRes(await r.json()); setView('detection')
        setBusy(false)
        runExplain(() => fetch('/api/explain-image', { method: 'POST', body: file, headers: { 'Content-Type': file.type } }),
          'Why was this image flagged?')
      }
    } catch (e) {
      setErr(`${e.message}. Is server.py running?`)
      setExplain({ state: 'idle' })
    } finally {
      setBusy(false)
    }
  }

  // video: pause on the first detection and explain it. Called on loadedmetadata AND canplay,
  // because some browsers ignore a seek made before the video can play; the explanation runs once.
  const onVideoReady = () => {
    if (!vid || vid.legacy) return
    const v = videoRef.current
    if (vid.first_detection != null) {
      const target = (vid.first_detection + 0.5) / vid.fps
      if (v && !seekSettled.current) {
        if (Math.abs(v.currentTime - target) > 0.2) { v.pause(); v.currentTime = target }
        else seekSettled.current = true   // arrived: never move the video again, so scrubbing is left alone
      }
      if (!initialSeekDone.current) {
        initialSeekDone.current = true
        explainFrame(vid.clip, vid.first_detection, vid.fps)
      }
    } else if (!initialSeekDone.current) {
      initialSeekDone.current = true
      setExplain({ state: 'ready', question: 'Anything flagged in this video?', data: {
        threshold: vid.threshold, timing_ms: 0, images: {},
        messages: [{ id: 'none', title: 'Nothing flagged',
          text: `No frame has a region above the detection threshold of ${vid.threshold.toFixed(3)} (highest score ${vid.max_score.toFixed(3)}). Scrub the video and press "Explain frame" to look at any frame anyway.`,
          facts: [['Highest score', vid.max_score.toFixed(3)]], image: null }] } })
    }
  }

  useEffect(() => {
    const v = videoRef.current
    if (!v || !vid) return
    const tick = () => setFrame(Math.min(Math.max(vid.frames - 1, 0), Math.max(0, Math.floor(v.currentTime * vid.fps))))
    v.addEventListener('timeupdate', tick)
    v.addEventListener('seeked', tick)
    return () => { v.removeEventListener('timeupdate', tick); v.removeEventListener('seeked', tick) }
  }, [vid])

  const hasResult = !!(res || vid)
  const canExplain = !!(vid && !vid.legacy)

  return (
    <div className="analyzer">
      <label
        className={`live-drop${drag ? ' drag' : ''}${busy ? ' busy' : ''}${hasResult || busy ? ' compact' : ''}`}
        onDragOver={e => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={e => { e.preventDefault(); setDrag(false); if (!busy) run(e.dataTransfer.files[0]) }}
      >
        <input type="file" accept="image/png,image/jpeg,video/mp4,video/quicktime,video/webm"
          onChange={e => { run(e.target.files[0]); e.target.value = '' }} disabled={!!busy} />
        <span className="live-drop-t">
          {busy ? 'Working on it, one moment…'
            : hasResult ? 'Drop another image or video, or click to choose' : 'Drop a road image or video here, or click to choose'}
        </span>
        {!hasResult && !busy && <span className="live-drop-d mono">Image: JPG / PNG up to 15 MB · Video: MP4 up to 200 MB, first 30 s at 10 fps · scored at 1024×512</span>}
      </label>
      {err && <div className={`live-err${vid?.legacy ? ' warn' : ''}`}>{err}</div>}

      <div className="analyzer-grid">
        <div className="analyzer-media">
          {busy && preview && <LoadingCard key={preview.url} kind={preview.kind} name={preview.name} previewUrl={preview.url} duration={preview.duration} />}

          {!hasResult && !busy && (
            <div className="analyzer-placeholder">
              <div className="ph-title">Your result appears here</div>
              <div className="ph-sub">A video pauses on its first detection and the analyst explains it. An image is explained straight away.</div>
            </div>
          )}

          {!busy && vid && (
            <div className="panel-soft">
              <video ref={videoRef} className="live-img" src={vid.url} controls muted playsInline preload="auto"
                onLoadedMetadata={onVideoReady} onCanPlay={onVideoReady} />
              <div className="media-cap mono">Top: detection · middle: anomaly score · bottom: head disagreement</div>
              {vid.frame_info.length > 0 && (
                <Timeline info={vid.frame_info} threshold={vid.threshold} current={frame}
                  onPick={f => { seekTo(f, vid.fps); explainFrame(vid.clip, f, vid.fps) }} />
              )}
              {canExplain && (
                <div className="media-actions">
                  <button className="btn-soft" onClick={() => explainFrame(vid.clip, frame, vid.fps)}>
                    Explain frame {frame} ({(frame / vid.fps).toFixed(1)} s)
                  </button>
                  {vid.first_detection != null && (
                    <button className="btn-soft" onClick={() => { seekTo(vid.first_detection, vid.fps); explainFrame(vid.clip, vid.first_detection, vid.fps) }}>
                      Jump to first detection
                    </button>
                  )}
                </div>
              )}
              <div className="proof-row live-stats">
                <div className="proof-stat"><div className="k mono">FRAMES WITH A BOX</div><div className="v mono small">{vid.frames_with_box} / {vid.frames}</div></div>
                <div className="proof-stat"><div className="k mono">MAX ANOMALY SCORE</div><div className="v mono">{vid.max_score.toFixed(3)}</div></div>
                <div className="proof-stat"><div className="k mono">MODEL FORWARD / FRAME</div><div className="v mono small">{vid.ms_per_frame.model_forward} ms</div></div>
                <div className="proof-stat"><div className="k mono">TOTAL / FRAME</div><div className="v mono small">{vid.ms_per_frame.total} ms</div></div>
              </div>
              <p className="media-note">
                {vid.frames} frames at {vid.fps} fps (source {vid.source_fps} fps){vid.truncated ? ', first 30 s only' : ''}, processed on {vid.device}{' '}
                and played back, so this is not real time. Each frame is scored independently, like a single image.
              </p>
            </div>
          )}

          {!busy && res && (
            <div className="panel-soft">
              <div className="x1-seg" role="tablist">
                {IMAGE_VIEWS.map(([k, label]) => (
                  <button key={k} role="tab" aria-selected={view === k}
                    className={`x1-seg-btn ${view === k ? 'active' : ''}`} onClick={() => setView(k)}>{label}</button>
                ))}
              </div>
              <img className="live-img" src={`data:image/png;base64,${res[view]}`} alt={view} />
              <div className="proof-row live-stats">
                <div className="proof-stat"><div className="k mono">BOXES (SCORE ≥ {res.threshold.toFixed(3)})</div><div className="v mono">{res.boxes}</div></div>
                <div className="proof-stat"><div className="k mono">MAX ANOMALY SCORE</div><div className="v mono">{res.max_score.toFixed(3)}</div></div>
                <div className="proof-stat"><div className="k mono">MODEL FORWARD</div><div className="v mono small">{res.timing_ms.model_forward} ms</div></div>
                <div className="proof-stat"><div className="k mono">END TO END</div><div className="v mono small">{res.timing_ms.total} ms</div></div>
              </div>
              <p className="media-note">
                Measured on {res.device}, one pass of {res.checkpoint}. The threshold was fitted on the Fishyscapes validation half; on
                photos unlike Lost &amp; Found (large or close objects, other cameras) expect misses, as on RoadAnomaly21.
              </p>
            </div>
          )}
        </div>

        <AnalystChat key={`${explain.state}|${explain.question}|${explain.data?.timing_ms}`}
          state={explain.state} data={explain.data} question={explain.question} error={explain.error} loadingLabel={explain.label}
          expanded={chatExpanded} onToggleExpanded={() => setChatExpanded(v => !v)} />
      </div>
    </div>
  )
}
