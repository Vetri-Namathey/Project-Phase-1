import { Link } from 'react-router-dom'
import { useEffect, useRef, useState } from 'react'

const METRICS = [
  { k: 'TEST AUROC', v: 0.9920, decimals: 4 },
  { k: 'FPR@95', v: 0.0287, decimals: 4 },
  { k: 'PRE-CALIB GATE', v: 0.75, decimals: 2 },
  { k: 'POST-CALIB TARGET', v: 0.83, decimals: 2 },
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
    d: 'CARLA-rendered props and filtered COCO cutouts are pasted onto real Cityscapes frames during training, teaching the heads what an anomaly looks like without ever seeing a real one.',
  },
  {
    n: '03',
    t: 'Disagreement as uncertainty',
    d: 'The three heads rarely agree on genuinely novel objects. That spread — not a single confidence score — is the signal a downstream system can actually trust.',
  },
]

// Real numbers from PLAN.md's 2026-09-25 L_calib run against the verified
// AUROC=0.9920 checkpoint. Honest finding, not a clean win: whole-image ECE
// barely moves because Fishyscapes' 0.28% positive rate makes it dominated by
// trivial true-negatives regardless of what happens on the hard pixels.
const CALIB_MODELS = [
  {
    id: 'raw',
    label: 'Raw',
    sub: 'Sigmoid, no calibration',
    auroc: 0.9920, ap: 0.6215, fpr95: 0.0290, ece: 0.0004,
  },
  {
    id: 'temp',
    label: 'Temp-scaled',
    sub: 'Post-hoc scalar T, per head',
    auroc: 0.9924, ap: 0.6193, fpr95: 0.0287, ece: 0.0020,
  },
  {
    id: 'calib',
    label: 'L_calib',
    sub: 'Joint fine-tune, soft-ECE surrogate',
    auroc: 0.9924, ap: 0.6024, fpr95: 0.0293, ece: 0.0005,
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
  const [active, setActive] = useState('calib')
  const model = CALIB_MODELS.find((m) => m.id === active)
  const raw = CALIB_MODELS[0]

  const rows = [
    { label: 'AUROC', key: 'auroc', decimals: 4, higherBetter: true },
    { label: 'AP', key: 'ap', decimals: 4, higherBetter: true },
    { label: 'FPR@95', key: 'fpr95', decimals: 4, higherBetter: false },
    { label: 'ECE', key: 'ece', decimals: 4, higherBetter: false },
  ]

  return (
    <section className="calib-block">
      <div className="stage-head">
        <h2>Calibration — the honest result</h2>
        <p>Phase 2b, run 2026-09-25 on the verified AUROC=0.9920 checkpoint. Toggle a model to compare it against raw.</p>
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
          {active === 'raw' && 'The unmodified checkpoint — no post-hoc or joint-trained calibration applied.'}
          {active === 'temp' && 'A single scalar temperature fit per head. It moves AUROC up fractionally but makes whole-image ECE worse (0.0004 → 0.0020) — calibration and discrimination trade off in different directions than expected.'}
          {active === 'calib' && 'Joint fine-tune against a differentiable ECE surrogate (SoftECELoss), selected on validation ECE. Whole-image ECE (0.0005) is barely different from raw\'s already-tiny 0.0004 — Fishyscapes\' 0.28% positive rate means whole-image ECE is dominated by trivial true negatives, so it cannot separate these three models. Boundary-only ECE + UBQ (in progress) is the metric that could.'}
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
        selection. Gate numbers are the pre- and post-calibration bars this project reports
        against — see the Training Runs page for the full history.
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
