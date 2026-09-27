import { Link } from 'react-router-dom'
import { useEffect, useRef, useState } from 'react'

// phase2a_coco_run1, Fishyscapes Lost&Found test half; mIoU on all 500
// Cityscapes val images. Every number: runs/RESULTS.md.
const METRICS = [
  { k: 'TEST AUROC', v: 0.9910, decimals: 4 },
  { k: 'TEST AP', v: 0.7499, decimals: 4 },
  { k: 'FPR@95', v: 0.0209, decimals: 4 },
  { k: 'SEG mIoU', v: 0.7700, decimals: 4 },
]

const FEATURES = [
  {
    n: '01',
    t: 'Frozen encoder, 3 independent heads',
    d: 'A SegFormer backbone stays frozen while three separately-seeded OOD heads each learn to score "does this pixel look like something I was never trained on."',
  },
  {
    n: '02',
    t: 'CutMix outlier exposure',
    d: 'COCO object cutouts (every Cityscapes-known category removed) are pasted onto the road in real Cityscapes frames, sized like real lost cargo, teaching the heads what an anomaly looks like without ever seeing a real one.',
  },
  {
    n: '03',
    t: 'Disagreement as uncertainty',
    d: 'The three heads disagree where something is novel: that spread alone separates anomalies at AUROC 0.985, and it comes free with the single continuous-mode pass. One head needs 10 MC-Dropout passes to get a similar signal.',
  },
]

// phase2a_coco_run1 + phase2b_calib2, Fishyscapes test half (runs/RESULTS.md).
// Temperatures fitted on the val half with plain NLL. band-ECE = calibration
// on pixels within 8px of an object's edge (Novelty 6).
const CALIB_MODELS = [
  {
    id: 'raw', label: 'Raw', sub: 'No calibration',
    auroc: 0.9909, ap: 0.7491, ece: 0.0003, band: 0.1950,
  },
  {
    id: 'temp', label: 'Temp (whole)', sub: 'One scalar T = 1.22, fitted on all val pixels',
    auroc: 0.9937, ap: 0.7490, ece: 0.0004, band: 0.1844,
  },
  {
    id: 'tempband', label: 'Temp (edges)', sub: 'One scalar T = 3.62, fitted on edge pixels only',
    auroc: 0.9938, ap: 0.7486, ece: 0.0203, band: 0.1058,
  },
  {
    id: 'calib', label: 'L_calib', sub: 'Joint fine-tune: edge-band soft-ECE + confidence anchor',
    auroc: 0.9909, ap: 0.7527, ece: 0.0003, band: 0.1894,
  },
]

function useOnScreen(ref) {
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true)
          obs.disconnect()
        }
      },
      { threshold: 0.4 },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [ref])
  return visible
}

function CountUpStat({ k, v, decimals }) {
  const ref = useRef(null)
  const visible = useOnScreen(ref)
  const [display, setDisplay] = useState(0)

  useEffect(() => {
    if (!visible) return
    const duration = 900
    const start = performance.now()
    let raf
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration)
      const eased = 1 - Math.pow(1 - t, 3)
      setDisplay(v * eased)
      if (t < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [visible, v])

  return (
    <div className="proof-stat" ref={ref}>
      <div className="k mono">{k}</div>
      <div className="v mono">{display.toFixed(decimals)}</div>
    </div>
  )
}

function PulseGrid() {
  const ROWS = 4
  const COLS = 9
  const [cells, setCells] = useState(() => Array(ROWS * COLS).fill(0))

  useEffect(() => {
    const id = setInterval(() => {
      setCells((prev) => {
        const next = prev.map((c) => Math.max(0, c - 0.12))
        // simulate a single head disagreeing on one "novel object" pixel at a time
        const idx = Math.floor(Math.random() * next.length)
        next[idx] = 0.5 + Math.random() * 0.4
        return next
      })
    }, 900)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="pulse-grid" style={{ gridTemplateColumns: `repeat(${COLS}, 1fr)` }} aria-hidden="true">
      {cells.map((intensity, i) => (
        <div
          key={i}
          className="pulse-cell"
          style={{
            background: intensity > 0.05
              ? `rgba(61,90,254,${intensity.toFixed(2)})`
              : 'transparent',
          }}
        />
      ))}
    </div>
  )
}

function CalibrationPanel() {
  const [active, setActive] = useState('raw')
  const model = CALIB_MODELS.find((m) => m.id === active)
  const raw = CALIB_MODELS[0]

  const rows = [
    { label: 'AUROC', key: 'auroc', decimals: 4, higherBetter: true },
    { label: 'AP', key: 'ap', decimals: 4, higherBetter: true },
    { label: 'ECE (whole image)', key: 'ece', decimals: 4, higherBetter: false },
    { label: 'ECE (object edges)', key: 'band', decimals: 4, higherBetter: false },
  ]

  return (
    <section className="calib-block">
      <div className="stage-head">
        <h2>Calibration — the honest result</h2>
        <p>Phase 2b on the 3-head checkpoint, Fishyscapes test half. Toggle a model to compare it against raw.</p>
      </div>

      <div className="calib-tabs" role="tablist">
        {CALIB_MODELS.map((m) => (
          <button
            key={m.id}
            role="tab"
            aria-selected={active === m.id}
            className={`calib-tab${active === m.id ? ' active' : ''}`}
            onClick={() => setActive(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="calib-body">
        <div className="calib-sub mono">{model.sub}</div>
        <table className="calib-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Raw</th>
              <th>{model.label}</th>
              <th>Δ</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const rawV = raw[r.key]
              const curV = model[r.key]
              const delta = curV - rawV
              const isSame = active === 'raw'
              const improved = r.higherBetter ? delta > 0.00005 : delta < -0.00005
              const worsened = r.higherBetter ? delta < -0.00005 : delta > 0.00005
              return (
                <tr key={r.key}>
                  <td className="mono">{r.label}</td>
                  <td className="mono">{rawV.toFixed(r.decimals)}</td>
                  <td className="mono">{curV.toFixed(r.decimals)}</td>
                  <td className={`mono delta${isSame ? '' : improved ? ' delta-good' : worsened ? ' delta-bad' : ''}`}>
                    {isSame ? '—' : `${delta >= 0 ? '+' : ''}${delta.toFixed(r.decimals)}`}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <p className="calib-note">
          {active === 'raw' && 'Whole-image calibration is already near perfect (ECE 0.0003), because 99.7% of pixels are easy background. At object edges the same model is off by 0.195: that edge gap is where calibration actually matters.'}
          {active === 'temp' && 'One temperature fitted on all pixels barely moves the edges (0.195 → 0.184). A single global rescale also cannot change the outline of the flagged region: UBQ is identical at the fitted threshold.'}
          {active === 'tempband' && 'Fitting the temperature on edge pixels fixes the edges (0.106) but makes the rest of the image 70× worse (0.0003 → 0.020). One global knob cannot fix both, which is the gap L_calib was meant to close.'}
          {active === 'calib' && 'The finding: L_calib keeps whole-image ECE and AP, but its edge ECE (0.189) is no better than raw (95% CI includes 0). The model is ~99% sure of pasted training objects and only 50–69% sure of real ones, so calibration learned on pastes does not transfer to real anomalies.'}
        </p>
      </div>
    </section>
  )
}

export default function HomePage() {
  return (
    <div className="content home">
      <section className="hero-block">
        <div className="hero-row-flex">
          <div className="hero-copy">
            <div className="stamp">EXPERIMENT B · SEGFORMER-B5 · 3 OOD HEADS</div>
            <h1 className="hero-title">
              Catch what the model<br />was <span className="accent-text">never trained</span> to see.
            </h1>
            <p className="hero-sub">
              TwinGuard flags road-scene pixels a segmentation model has never encountered —
              debris, animals, unfamiliar cargo — with a confidence score built to be trustworthy,
              not just high. Trained on Cityscapes, stress-tested on Fishyscapes Lost&amp;Found.
            </p>
            <div className="hero-cta">
              <Link to="/demo" className="cta-btn cta-primary">View Live Demo →</Link>
              <Link to="/runs" className="cta-btn cta-ghost">Training Runs</Link>
            </div>
          </div>
          <div className="hero-visual-wrap">
            <div className="hero-visual-label mono">HEAD DISAGREEMENT · LIVE SIMULATION</div>
            <PulseGrid />
            <div className="hero-visual-caption">
              Each pulse is a simulated pixel where the three OOD heads disagree —
              the actual signal the demo's disagreement heatmap is built from.
            </div>
          </div>
        </div>
      </section>

      <section className="proof-row home-proof">
        {METRICS.map((m) => (
          <CountUpStat key={m.k} k={m.k} v={m.v} decimals={m.decimals} />
        ))}
      </section>
      <p className="metric-note">
        Measured on the Fishyscapes Lost&amp;Found test half, held out from checkpoint
        selection. The untrained off-the-shelf baseline scores AUROC 0.837 / AP 0.011 on the
        same images. See Training Runs for every run, including the ones that failed.
      </p>

      <CalibrationPanel />

      <section className="feature-block">
        <div className="stage-head">
          <h2>How it actually works</h2>
          <p>Three ideas, in the order they run at inference time.</p>
        </div>
        <div className="feature-grid">
          {FEATURES.map((f) => (
            <div className="feature-card" key={f.n}>
              <div className="feature-n mono">{f.n}</div>
              <div className="feature-t">{f.t}</div>
              <div className="feature-d">{f.d}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
