import { Link } from 'react-router-dom'
import { useEffect, useRef, useState } from 'react'

// train.log, selected epoch 4, Fishyscapes test half. AP first: AUROC is
// saturated near 0.99, and Fishyscapes ranks on AP.
const METRICS = [
  { k: 'TEST AP', v: 0.6218, decimals: 4 },
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

// Every number from eval_fishyscapes_T122.log (model_3head_best.pth /
// model_3head_calib_best.pth, Fishyscapes test half, input scale 1; every
// calibrator fit on the val half only). The temperature row uses the plain-NLL
// refit T=1.2251; the earlier whole-image T was mis-fitted (MISTAKES.md M1).
// bece = ECE restricted to pixels within 8px of a true anomaly edge.
// Whole-image ECE is shown at 4 dp; temp(whole) is 0.00012 at 5 dp
// (calibrate_compare_T122.log).
const CALIB_MODELS = [
  {
    id: 'raw',
    label: 'Raw',
    sub: 'Sigmoid, no calibration',
    auroc: 0.9920, ap: 0.6215, ece: 0.0004, bece: 0.2273,
    note: 'Near-perfect whole-image ECE (0.0004) hides a boundary ECE of 0.2273. With well under 1% of pixels anomalous, whole-image ECE is dominated by easy background. At object edges the model is under-confident: true-anomaly edge pixels average a score of 0.36.',
  },
  {
    id: 'temp',
    label: 'T = 1.2251',
    sub: 'Temperature fit on the whole image (plain NLL)',
    auroc: 0.9924, ap: 0.6206, ece: 0.0001, bece: 0.2095,
    note: 'Standard temperature scaling, fit by plain NLL on held-out real images. It calibrates the whole image (ECE 0.00012) but barely moves the edges: 0.2273 to 0.2095. One global temperature is tuned to the background pixels that make up almost all of the image.',
  },
  {
    id: 'mixed',
    label: 'T = 3.15',
    sub: 'Temperature fit 50/50 on whole image + boundary',
    auroc: 0.9925, ap: 0.6177, ece: 0.0177, bece: 0.1277,
    note: 'The compromise point. A version that varied T with head disagreement did no better than this single temperature (tested against a pre-registered control). Disagreement marks where the edges are, but not which way to correct them.',
  },
  {
    id: 'band',
    label: 'T = 4.36',
    sub: 'Temperature fit on boundary pixels only',
    auroc: 0.9925, ap: 0.6173, ece: 0.0420, bece: 0.0942,
    note: 'Cuts boundary ECE from 0.2095 (T = 1.2251) to 0.0942, but whole-image ECE rises from 0.0001 to 0.0420. The whole image wants T ≈ 1.2 and the edges want T ≈ 4.4, fitted on the same objective. No single temperature fixes both. That trade-off is the finding.',
  },
  {
    id: 'calib',
    label: 'L_calib',
    sub: 'Joint fine-tune on CutMix pastes, soft-ECE surrogate',
    auroc: 0.9924, ap: 0.6024, ece: 0.0005, bece: 0.2396,
    note: 'Calibration learned from training pastes transfers the wrong correction. It lowers edge scores. On real Fishyscapes, where edges are under-confident, that is worse than raw and worse than temperature scaling (CIs exclude 0). On a pasted CARLA driving route, where edges are over-confident, it beats raw but is no better than the same whole-image temperature. Calibration has to be fit on held-out real data.',
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
  const [active, setActive] = useState('band')
  const model = CALIB_MODELS.find((m) => m.id === active)
  const raw = CALIB_MODELS[0]

  const rows = [
    { label: 'AP', key: 'ap', decimals: 4, higherBetter: true },
    { label: 'AUROC', key: 'auroc', decimals: 4, higherBetter: true },
    { label: 'ECE (whole image)', key: 'ece', decimals: 4, higherBetter: false },
    { label: 'ECE (object edges, r=8)', key: 'bece', decimals: 4, higherBetter: false },
  ]

  return (
    <section className="calib-block">
      <div className="stage-head">
        <h2>Calibration — the edges tell a different story</h2>
        <p>Fishyscapes test half, verified AUROC=0.9920 checkpoint. Every calibrator is fit on held-out real images. Pick one to compare against raw.</p>
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
        <p className="calib-note">{model.note}</p>
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
