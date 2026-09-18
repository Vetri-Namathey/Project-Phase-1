import { useEffect, useMemo, useState } from 'react'
import FadeImage from '../components/FadeImage'

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

function HoverDisagreement({ src, compute }) {
  const [hover, setHover] = useState(null)
  function onMove(e) {
    const rect = e.currentTarget.getBoundingClientRect()
    const xFrac = (e.clientX - rect.left) / rect.width
    const yFrac = (e.clientY - rect.top) / rect.height
    setHover({ x: e.clientX - rect.left, y: e.clientY - rect.top, value: compute(xFrac, yFrac) })
  }
  const visible = !!(hover && hover.value !== null)
  return (
    <div className="img-wrap" onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
      <FadeImage src={src} alt="disagreement" />
      <div
        className={`hover-readout accent-readout mono${visible ? ' visible' : ''}`}
        style={hover ? { left: hover.x, top: hover.y } : undefined}
      >
        std {hover ? hover.value?.toFixed(3) : ''}
      </div>
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
    const entries = Object.entries(active.grids)
    Promise.all(entries.map(([k, url]) => fetch(url).then(r => r.json()).then(g => [k, g])))
      .then(pairs => {
        if (!cancelled) setGrids(Object.fromEntries(pairs))
      })
    return () => { cancelled = true }
  }, [active])

  // Std computed here is raw; server.py normalizes the *displayed* disagreement
  // heatmap by its own per-image max before colorizing (server.py, build_manifest).
  // Normalize by the max std across this image's own grid so the hover number
  // roughly agrees with the color the cursor is sitting on, instead of a raw
  // value that can look tiny next to a visually "hot" pixel.
  const disagreementAt = useMemo(() => {
    if (!grids.head0 || !grids.head1 || !grids.head2) return () => null

    const stdAt = (xFrac, yFrac) => {
      const vals = [grids.head0, grids.head1, grids.head2].map(g => gridValueAt(g, xFrac, yFrac))
      const mean = vals.reduce((a, b) => a + b, 0) / 3
      const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / 3
      return Math.sqrt(variance)
    }

    let maxStd = 0
    const { width, height } = grids.head0
    for (let gy = 0; gy < height; gy++) {
      for (let gx = 0; gx < width; gx++) {
        maxStd = Math.max(maxStd, stdAt((gx + 0.5) / width, (gy + 0.5) / height))
      }
    }

    return (xFrac, yFrac) => (maxStd > 0 ? stdAt(xFrac, yFrac) / maxStd : 0)
  }, [grids])

  return (
    <div className="content">
      <div className="page-head">
        <div className="stamp">EXHIBIT A</div>
        <h1 className="headline">Detection pipeline — live inference</h1>
        <p className="lede">
          Real output from the trained checkpoint (best AUROC 0.619, epoch 1 of Checkpoint B) run on real Fishyscapes
          Lost&amp;Found anomaly photos — never seen during training.
        </p>
      </div>

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
                <p>{active.anomaly_pixels.toLocaleString()} ground-truth anomaly pixels in this frame. Hover any heatmap below for the exact score under your cursor.</p>
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
                <span className="tick mono accent-tick">DETECTION — TOP {100 - 97}% SCORING REGION, BOXED</span>
                <FadeImage src={active.panels.detection} alt="detection" />
                <div className="cap">
                  <div className="t">Bounding box drawn where the model's own fused score is highest in this frame</div>
                  <div className="d">Threshold is relative to this image's own score distribution (fused scores run low in absolute terms — see Training Runs) — this is the region the model itself considers most anomalous, boxed directly on the real photo, not a diffuse heatmap you have to interpret.</div>
                </div>
              </div>

              <div className="section-tag">Input &amp; the uncertainty signal</div>
              <div className="hero-row">
                <div className="hero-panel">
                  <span className="tick mono">RAW FRAME</span>
                  <FadeImage src={active.panels.raw} alt="raw" />
                  <div className="cap">
                    <div className="t">Camera input</div>
                    <div className="d">Unmodified real-world photo, never seen during training</div>
                  </div>
                </div>
                <div className="hero-panel accent">
                  <span className="tick mono accent-tick">HEAD DISAGREEMENT</span>
                  <HoverDisagreement src={active.panels.disagreement} compute={disagreementAt} />
                  <div className="cap">
                    <div className="t">Uncertainty signal (per-pixel std across 3 heads)</div>
                    <div className="d">High values mark pixels where the three independently-seeded heads disagree with each other</div>
                  </div>
                </div>
              </div>

              <div className="section-tag">Segmentation, fused score &amp; per-head breakdown (hover for values)</div>
              <div className="spec-grid">
                <div className="spec-card">
                  <div className="img-wrap"><FadeImage src={active.panels.segmentation} alt="segmentation" /></div>
                  <div className="cap"><div className="n mono">01</div><div className="t">Segmentation</div></div>
                </div>
                <HoverCard n="02" label="Fused OOD heatmap" src={active.panels.fused} grid={grids.fused} />
                <div className="spec-card">
                  <div className="img-wrap"><FadeImage src={active.panels.ground_truth} alt="ground truth" /></div>
                  <div className="cap"><div className="n mono">03</div><div className="t">Ground truth anomaly</div></div>
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
