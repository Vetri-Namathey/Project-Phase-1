import { useEffect, useState } from 'react'
import FadeImage from '../components/FadeImage'

function ReelItem({ item, index, active, onClick }) {
  return (
    <button className={`reel-item${active ? ' active' : ''}`} onClick={onClick}>
      <FadeImage src={item.panels.raw} alt="" />
      <div className="meta">
        <div className="idx mono">{String(index + 1).padStart(2, '0')} · {item.selection}</div>
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
    </div>
  )
}

function gridValueAt(grid, xFrac, yFrac) {
  if (!grid) return null
  const gx = Math.min(grid.width - 1, Math.max(0, Math.floor(xFrac * grid.width)))
  const gy = Math.min(grid.height - 1, Math.max(0, Math.floor(yFrac * grid.height)))
  return grid.values[gy][gx]
}

// Hover readout of the raw value under the cursor. Heatmaps are drawn on a
// fixed 0-1 scale (scores) or scaled to the frame's own maximum (uncertainty
// maps, whose raw values are tiny) -- the number shown is always the raw one.
function HoverImage({ src, alt, grid, prefix = '', digits = 3, accent = false }) {
  const [hover, setHover] = useState(null)
  function onMove(e) {
    const rect = e.currentTarget.getBoundingClientRect()
    const value = gridValueAt(grid, (e.clientX - rect.left) / rect.width, (e.clientY - rect.top) / rect.height)
    setHover({ x: e.clientX - rect.left, y: e.clientY - rect.top, value })
  }
  const visible = !!(hover && hover.value !== null)
  return (
    <div className="img-wrap" onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
      <FadeImage src={src} alt={alt} />
      <div
        className={`hover-readout mono${accent ? ' accent-readout' : ''}${visible ? ' visible' : ''}`}
        style={hover ? { left: hover.x, top: hover.y } : undefined}
      >
        {hover && hover.value !== null ? `${prefix}${hover.value.toFixed(digits)}` : ''}
      </div>
    </div>
  )
}

function HoverCard({ n, label, src, grid }) {
  return (
    <div className="spec-card">
      <HoverImage src={src} alt={label} grid={grid} />
      <div className="cap">
        <div className="n mono">{n}</div>
        <div className="t">{label}</div>
      </div>
    </div>
  )
}

function Stat({ k, v, accent = false }) {
  return (
    <div className="proof-stat">
      <div className="k mono">{k}</div>
      <div className={`v mono${accent ? ' accent-text' : ''}`}>{v}</div>
    </div>
  )
}

export default function DemoPage() {
  const [manifest, setManifest] = useState(null)
  const [activeId, setActiveId] = useState(null)
  const [grids, setGrids] = useState({})

  useEffect(() => {
    fetch('/api/manifest').then(r => r.json()).then(data => {
      setManifest(data)
      if (data.frames.length > 0) setActiveId(data.frames[0].id)
    })
  }, [])

  const frames = manifest?.frames
  const active = frames?.find(m => m.id === activeId)

  useEffect(() => {
    if (!active) return
    let cancelled = false
    // Clear first, so a hover can never read the previous frame's values
    // while this frame's grids are still loading.
    setGrids({})
    Promise.all(Object.entries(active.grids).map(([k, url]) => fetch(url).then(r => r.json()).then(g => [k, g])))
      .then(pairs => { if (!cancelled) setGrids(Object.fromEntries(pairs)) })
    return () => { cancelled = true }
  }, [active])

  return (
    <div className="content">
      <div className="page-head">
        <div className="stamp">EXHIBIT A</div>
        <h1 className="headline">Detection pipeline — real inference</h1>
        <p className="lede">
          Real output of the trained 3-head checkpoint (test AUROC 0.991, AP 0.750) on Fishyscapes Lost&amp;Found
          photos from the held-out <b>test half</b>, never seen in training or checkpoint selection. Three frames are
          the clearest of the {manifest?.test_frames_ranked ?? '…'} ranked test frames and three are typical
          (median), labelled in the list, so this is not a highlight reel.
        </p>
      </div>

      <div className="body-row">
        <div className="rail">
          <div className="rail-label mono">TEST FRAMES</div>
          <div className="reel">
            {frames
              ? frames.map((item, i) => (
                <ReelItem key={item.id} item={item} index={i} active={item.id === activeId}
                  onClick={() => setActiveId(item.id)} />
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
                <h2>{active.title} <span className="mono" style={{ fontSize: 13 }}>· {active.selection}</span></h2>
                <p>
                  {active.anomaly_pixels.toLocaleString()} ground-truth anomaly pixels. Hover any heatmap for the
                  value under your cursor.
                </p>
              </div>

              <div className="section-tag">Is the real object actually being flagged?</div>
              <div className="proof-row">
                <Stat k="SCORE INSIDE OBJECT" v={active.score_inside.toFixed(3)} accent />
                <Stat k="SCORE ON BACKGROUND" v={active.score_outside.toFixed(3)} />
                <Stat k="FRAME AP" v={active.frame_ap.toFixed(3)} />
                <Stat k="OBJECTS FOUND" v={`${active.objects_found} / ${active.objects_total}`} />
              </div>
              <div className="hero-panel accent detection-panel">
                <span className="tick mono accent-tick">
                  DETECTION @ t={manifest.threshold.toFixed(3)} · {active.boxes_on_object}/{active.boxes_drawn} BOXES ON A REAL OBJECT
                </span>
                <FadeImage src={active.panels.detection} alt="detection" />
                <div className="cap">
                  <div className="t">Blue boxes: regions the model flags. Green outline: the real anomaly.</div>
                  <div className="d">
                    One fixed threshold for every frame ({manifest.threshold.toFixed(3)}), fitted for best F1 on the
                    validation half and never tuned on these images. A blue box with no green inside it is a false alarm.
                    {active.boundary_f1 !== null && ` Outline agreement on found objects: boundary F1 ${active.boundary_f1.toFixed(2)}.`}
                  </div>
                </div>
              </div>

              <div className="section-tag">Dual-mode inference — the two uncertainty signals</div>
              <div className="proof-row">
                <Stat k="CONTINUOUS MODE" v={`${active.latency_continuous_ms.toFixed(0)} ms`} />
                <Stat k={`SAFETY MODE (${manifest.mc_passes} PASSES)`} v={`${active.latency_safety_ms.toFixed(0)} ms`} />
                <Stat k="PEAK SCORE" v={active.peak_score.toFixed(3)} />
                <Stat k={`TRIGGER @ ${manifest.trigger_threshold}`} v={active.safety_triggered ? 'FIRES' : 'no'} accent={active.safety_triggered} />
              </div>
              <div className="hero-row">
                <div className="hero-panel accent">
                  <span className="tick mono accent-tick">BETWEEN-HEAD · FREE IN CONTINUOUS MODE</span>
                  <HoverImage src={active.panels.disagreement} alt="head disagreement" grid={grids.disagreement}
                    prefix="std " digits={4} accent />
                  <div className="cap">
                    <div className="t">Head disagreement (std across the 3 heads)</div>
                    <div className="d">Costs nothing extra: all three heads already ran. Separates anomalies at AUROC 0.985 on the test half.</div>
                  </div>
                </div>
                <div className="hero-panel">
                  <span className="tick mono">WITHIN-HEAD · SAFETY MODE ONLY</span>
                  <HoverImage src={active.panels.within} alt="MC-Dropout variance" grid={grids.within}
                    prefix="var " digits={5} />
                  <div className="cap">
                    <div className="t">MC-Dropout variance ({manifest.mc_passes} passes per head, one encoder pass)</div>
                    <div className="d">Tracks the same regions as disagreement. Measured: the two rise together under noise and blur, so they don't separate "bad camera" from "novel object".</div>
                  </div>
                </div>
              </div>
              <p className="calib-note">
                Latency on this laptop's {manifest.device} (bf16, 512×1024). It is not the deployment GPU the paper's
                &lt;35 ms / &lt;100 ms targets refer to; the ratio between the two modes is the meaningful part. The
                safety trigger at {manifest.trigger_threshold} is not yet tuned: no threshold met the &lt;5% normal /
                &gt;90% anomalous targets on this checkpoint (see Training Runs).
              </p>

              <div className="section-tag">Segmentation, fused score &amp; per-head breakdown (hover for values)</div>
              <div className="spec-grid">
                <div className="spec-card">
                  <div className="img-wrap"><FadeImage src={active.panels.segmentation} alt="segmentation" /></div>
                  <div className="cap"><div className="n mono">01</div><div className="t">Segmentation (19 classes)</div></div>
                </div>
                <HoverCard n="02" label="Fused anomaly score" src={active.panels.fused} grid={grids.fused} />
                <div className="spec-card">
                  <div className="img-wrap"><FadeImage src={active.panels.ground_truth} alt="ground truth" /></div>
                  <div className="cap"><div className="n mono">03</div><div className="t">Ground-truth anomaly</div></div>
                </div>
                <HoverCard n="04" label="OOD head 0" src={active.panels.head0} grid={grids.head0} />
                <HoverCard n="05" label="OOD head 1" src={active.panels.head1} grid={grids.head1} />
                <HoverCard n="06" label="OOD head 2" src={active.panels.head2} grid={grids.head2} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
