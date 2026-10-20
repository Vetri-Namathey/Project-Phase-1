import { useEffect, useState } from 'react'
import FadeImage from '../components/FadeImage'
import LiveAnalyzer from '../components/LiveAnalyzer'

function ReelItem({ item, index, active, onClick }) {
  return (
    <button className={`reel-item${active ? ' active' : ''}`} onClick={onClick}>
      <FadeImage src={item.panels.raw} alt="" />
      <div className="meta">
        <div className="idx mono">{String(index + 1).padStart(2, '0')}</div>
        <div className="name">{item.title}</div>
      </div>
    </button>
  )
}

function ReelSkeleton() {
  return (
    <div className="skel-reel-item">
      <div className="skeleton skel-thumb" />
      <div className="meta">
        <div className="skel-line" style={{ width: '18%' }} />
        <div className="skel-line" style={{ width: '75%' }} />
      </div>
    </div>
  )
}

function StageSkeleton() {
  return (
    <div>
      <div className="stage-head">
        <div className="skel-line" style={{ width: '38%', height: 17, marginBottom: 10 }} />
        <div className="skel-line" style={{ width: '68%' }} />
      </div>
      <div className="proof-row">
        {Array.from({ length: 4 }).map((_, i) => (
          <div className="proof-stat skeleton" key={i} style={{ height: 62 }} />
        ))}
      </div>
      <div className="hero-panel skel-hero detection-panel">
        <div className="skeleton" style={{ height: 280 }} />
      </div>
      <div className="hero-row">
        <div className="hero-panel skel-hero"><div className="skeleton" style={{ height: 220 }} /></div>
        <div className="hero-panel skel-hero"><div className="skeleton" style={{ height: 220 }} /></div>
      </div>
      <div className="spec-grid" style={{ marginTop: 24 }}>
        {Array.from({ length: 6 }).map((_, i) => (
          <div className="spec-card skeleton" key={i} style={{ height: 140 }} />
        ))}
      </div>
    </div>
  )
}

function gridValueAt(grid, xFrac, yFrac) {
  if (!grid) return null
  const gx = Math.min(grid.width - 1, Math.max(0, Math.floor(xFrac * grid.width)))
  const gy = Math.min(grid.height - 1, Math.max(0, Math.floor(yFrac * grid.height)))
  return grid.values[gy][gx]
}

function HoverCard({ n, label, src, grid, extra }) {
  const [hover, setHover] = useState(null)

  function onMove(e) {
    const rect = e.currentTarget.getBoundingClientRect()
    const xFrac = (e.clientX - rect.left) / rect.width
    const yFrac = (e.clientY - rect.top) / rect.height
    const value = extra ? extra(xFrac, yFrac) : gridValueAt(grid, xFrac, yFrac)
    setHover({ x: e.clientX - rect.left, y: e.clientY - rect.top, value })
  }

  const visible = !!(hover && hover.value !== null)

  return (
    <div className="spec-card">
      <div className="img-wrap" onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        <FadeImage src={src} alt={label} />
        <div
          className={`hover-readout mono${visible ? ' visible' : ''}`}
          style={hover ? { left: hover.x, top: hover.y } : undefined}
        >
          {hover ? hover.value?.toFixed(3) : ''}
        </div>
      </div>
      <div className="cap">
        <div className="n mono">{n}</div>
        <div className="t">{label}</div>
      </div>
    </div>
  )
}

const MODES = [
  ['frames', 'Test frames'],
  ['upload', 'Your own image or video'],
  ['driving', 'Driving sequence'],
]

export default function DemoPage() {
  const [mode, setMode] = useState('frames')
  const [manifest, setManifest] = useState(null)
  const [activeId, setActiveId] = useState(null)
  const [grids, setGrids] = useState({})

  useEffect(() => {
    fetch('/api/manifest').then(r => r.json()).then(data => {
      setManifest(data)
      if (data.length > 0) setActiveId(data[0].id)
    })
  }, [])

  const active = manifest?.find(m => m.id === activeId)

  useEffect(() => {
    if (!active) return
    let cancelled = false
    // Clear immediately so a HoverCard can never read the previous frame's grid
    // values while this frame's own grids are still in flight -- without this,
    // hovering right after switching frames shows numbers for the wrong image.
    setGrids({})
    const entries = Object.entries(active.grids).filter(([k]) => k === 'fused')
    Promise.all(entries.map(([k, url]) => fetch(url).then(r => r.json()).then(g => [k, g])))
      .then(pairs => {
        if (!cancelled) setGrids(Object.fromEntries(pairs))
      })
    return () => { cancelled = true }
  }, [active])

  return (
    <div className="content">
      <div className="page-head">
        <div className="stamp">Live demo</div>
        <h1 className="headline">See what the detector sees.</h1>
        <p className="lede">
          Real output from the trained checkpoint (the 2000-object-bank model). Browse held-out test
          frames, run the model on your own image or video, or watch a CARLA driving sequence.
        </p>
      </div>

      <div className="mode-tabs" role="tablist">
        {MODES.map(([k, label]) => (
          <button key={k} role="tab" aria-selected={mode === k}
            className={`mode-tab${mode === k ? ' active' : ''}`} onClick={() => setMode(k)}>{label}</button>
        ))}
      </div>

      {mode === 'frames' && (
        <>
          <p className="lede" style={{ marginBottom: 26 }}>
            Real Fishyscapes Lost&amp;Found anomaly photos from the held-out test half, never seen during
            training or model selection. Frames shown: the 3 test frames with the highest per-frame AP and
            the 3 around the median, picked by that rule, not by hand.
          </p>
      <div className="body-row">
        <div className="rail">
          <div className="rail-label mono">TEST FRAMES</div>
          <div className="reel">
            {manifest
              ? manifest.map((item, i) => (
                <ReelItem
                  key={item.id}
                  item={item}
                  index={i}
                  active={item.id === activeId}
                  onClick={() => setActiveId(item.id)}
                />
              ))
              : Array.from({ length: 6 }).map((_, i) => <ReelSkeleton key={i} />)}
          </div>
        </div>

        <div className="stage">
          {!active ? (
            <StageSkeleton />
          ) : (
            <div key={active.id} className="reactive">
              <div className="stage-head">
                <h2>{active.title}</h2>
                <p>{active.selection && `${active.selection} frame (per-frame AP ${active.frame_ap.toFixed(3)}). `}{active.anomaly_pixels.toLocaleString()} ground-truth anomaly pixels in this frame. Hover any heatmap below for the exact score under your cursor.</p>
              </div>

              <div className="section-tag">Is the real object actually being flagged?</div>
              <div className="proof-row">
                <div className="proof-stat">
                  <div className="k mono">SCORE INSIDE THE REAL OBJECT</div>
                  <div className="v mono accent-text">{active.score_inside.toFixed(3)}</div>
                </div>
                <div className="proof-stat">
                  <div className="k mono">SCORE ON BACKGROUND</div>
                  <div className="v mono">{active.score_outside.toFixed(3)}</div>
                </div>
                <div className="proof-stat">
                  <div className="k mono">SEPARATION</div>
                  <div className="v mono">{(active.score_inside - active.score_outside).toFixed(3)}</div>
                </div>
                <div className="proof-stat">
                  <div className="k mono">BOXES DRAWN</div>
                  <div className="v mono">{active.boxes_drawn}</div>
                </div>
              </div>
              <div className="hero-panel accent detection-panel">
                <span className="tick mono accent-tick">DETECTION — FUSED SCORE ≥ 0.477, BOXED</span>
                <FadeImage src={active.panels.detection} alt="detection" />
                <div className="cap">
                  <div className="t">Boxes around the largest regions (up to 3) where the fused score is at least 0.477</div>
                  <div className="d">One fixed threshold for every frame: the score cut-off with the best pixel F1 on the validation half (eval_fishyscapes_T122.log), never tuned on these test frames. A region also needs at least 0.05% of the image (about 1,000 pixels) to get a box, so a small object can rank well (high AP) yet stay unboxed when its score sits below 0.477, as in some frames here.</div>
                </div>
              </div>

              <div className="section-tag">Input, score &amp; ground truth (hover the heatmap for values)</div>
              <div className="spec-grid">
                <div className="spec-card">
                  <div className="img-wrap"><FadeImage src={active.panels.raw} alt="raw" /></div>
                  <div className="cap"><div className="n mono">01</div><div className="t">Camera input</div></div>
                </div>
                <HoverCard n="02" label="Anomaly score" src={active.panels.fused} grid={grids.fused} />
                <div className="spec-card">
                  <div className="img-wrap"><FadeImage src={active.panels.segmentation} alt="segmentation" /></div>
                  <div className="cap"><div className="n mono">03</div><div className="t">Segmentation</div></div>
                </div>
                <div className="spec-card">
                  <div className="img-wrap"><FadeImage src={active.panels.ground_truth} alt="ground truth" /></div>
                  <div className="cap"><div className="n mono">04</div><div className="t">Ground-truth anomaly</div></div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

        </>
      )}

      {mode === 'upload' && <LiveAnalyzer />}
      {mode === 'driving' && <DrivingVideo />}
    </div>
  )
}

const VIDEO_BASE = '/static/video/twinguard_video_town02_pasted'

// PLAN.md 2d: render_video.py's split-screen mp4 + its _stats.json. Numbers are
// read from the stats file, never typed in here, so they can't drift from the video.
function DrivingVideo() {
  const [stats, setStats] = useState(null)
  const [missing, setMissing] = useState(false)

  useEffect(() => {
    fetch(`${VIDEO_BASE}_stats.json`)
      .then(r => (r.ok ? r.json() : Promise.reject()))
      .then(setStats)
      .catch(() => setMissing(true))
  }, [])

  if (missing) return null
  const t = stats?.totals

  return (
    <div className="video-block">
      <div className="section-tag">Driving sequence: CARLA Town02, frame by frame</div>
      <div className="hero-panel accent">
        <span className="tick mono accent-tick">LEFT: ONE HEAD · RIGHT: 3-HEAD ENSEMBLE</span>
        <video src={`${VIDEO_BASE}.mp4`} controls loop muted playsInline preload="metadata" className="demo-video" />
        <div className="cap">
          <div className="t">Same model, same frames: one head alone vs all three heads together</div>
          <div className="d">
            Objects on the road are world-anchored pastes: COCO cutouts from outside the 1500-object training
            subsample (assuming the local COCO bank is the one used in training, still being confirmed) and
            CARLA props from the training bank, plus CARLA's own street props. Green outlines are
            ground truth. The left column is a single-head ablation of the same model, not a separately
            trained baseline, so its disagreement panel is empty: one head cannot disagree with itself.
          </div>
          <div className="d">
            These numbers measure pasted objects on CARLA, not real-world performance. Very large or close
            objects are outside the training paste scale (at most 0.20 of the image's short side). Boxes on
            yellow lane dashes are a CARLA→Cityscapes paint gap (Cityscapes lane paint is white), left visible
            on purpose.
          </div>
        </div>
      </div>
      {t && (
        <div className="proof-row">
          <div className="proof-stat">
            <div className="k mono">OBJECTS BOXED · ONE HEAD</div>
            <div className="v mono">{(100 * t.baseline.found / t.baseline.gt_objects).toFixed(1)}%</div>
          </div>
          <div className="proof-stat">
            <div className="k mono">OBJECTS BOXED · 3 HEADS</div>
            <div className="v mono accent-text">{(100 * t.ours.found / t.ours.gt_objects).toFixed(1)}%</div>
          </div>
          <div className="proof-stat">
            <div className="k mono">FALSE BOXES / FRAME · ONE HEAD</div>
            <div className="v mono">{(t.baseline.false_boxes / stats.frames).toFixed(2)}</div>
          </div>
          <div className="proof-stat">
            <div className="k mono">FALSE BOXES / FRAME · 3 HEADS</div>
            <div className="v mono">{(t.ours.false_boxes / stats.frames).toFixed(2)}</div>
          </div>
        </div>
      )}
    </div>
  )
}
