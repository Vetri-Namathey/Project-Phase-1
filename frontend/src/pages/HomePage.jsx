import { Link } from 'react-router-dom'
import { useEffect, useRef, useState } from 'react'
import Reveal from '../components/Reveal'

// train.log, selected epoch 4, Fishyscapes Lost&Found test half. AP first: AUROC is
// saturated near 0.99, and Fishyscapes ranks on AP.
const METRICS = [
  { k: 'Average precision', sub: 'ranks anomalies above normal pixels', v: 0.6218, decimals: 4 },
  { k: 'AUROC', sub: 'saturated near its ceiling', v: 0.9920, decimals: 4 },
  { k: 'FPR @ 95% recall', sub: 'normal pixels wrongly flagged', v: 0.0287, decimals: 4 },
]

// eval_fishyscapes_T122.log: raw score, Fishyscapes test half. Band = within 8 px of object outlines.
const ECE_WHOLE = 0.0004
const ECE_EDGE = 0.2273
const ECE_SCALE_MAX = 0.25

const STEPS = [
  {
    n: '01',
    t: 'A frozen encoder, three independent heads',
    d: 'A SegFormer-B5 backbone stays frozen while three separately seeded heads each learn to score how unfamiliar a pixel looks.',
  },
  {
    n: '02',
    t: 'Trained on pasted anomalies',
    d: 'CARLA props and filtered COCO cutouts are pasted onto Cityscapes road scenes, so the heads learn what "out of place" looks like without a single real anomaly label.',
  },
  {
    n: '03',
    t: 'Disagreement as uncertainty',
    d: 'The final score is the mean of the three heads. How much they disagree comes free from the same pass and is shown for the images and videos you upload.',
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

function CountUpMetric({ k, sub, v, decimals }) {
  const ref = useRef(null)
  const visible = useOnScreen(ref)
  const [display, setDisplay] = useState(0)
  useEffect(() => {
    if (!visible) return
    const duration = 1100
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
    <div className="metric" ref={ref}>
      <div className="v mono grad-text">{display.toFixed(decimals)}</div>
      <div className="k"><b>{k}</b> · {sub}</div>
    </div>
  )
}

// One real, pre-computed test frame from the backend (the same data the demo uses).
function useHeroFrame() {
  const [frame, setFrame] = useState(null)
  useEffect(() => {
    let cancelled = false
    fetch('/api/manifest')
      .then(r => (r.ok ? r.json() : Promise.reject()))
      .then(m => { if (!cancelled && m.length) setFrame(m[0]) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])
  return frame
}

export default function HomePage() {
  const frame = useHeroFrame()
  return (
    <div className="content home">
      <section className="hero">
        <div>
          <div className="stamp">SegFormer-B5 · 3 OOD heads</div>
          <h1 className="hero-title">
            Catch what the model was <span className="grad-text">never trained</span> to see.
          </h1>
          <p className="hero-sub">
            TwinGuard flags road-scene pixels a segmentation network has never encountered, such as
            debris, animals or fallen cargo, and says how much to trust each flag.
          </p>
          <div className="hero-cta">
            <Link to="/demo" className="cta-btn cta-primary">Open the demo →</Link>
            <a href="#finding" className="cta-btn cta-ghost">Why it matters</a>
          </div>
        </div>
        <div>
          <div className="frame-window">
            <div className="frame-bar"><i /><i /><i /><span className="mono">real test frame · detection</span></div>
            <div className="frame-img">
              {frame
                ? <img src={frame.panels.detection} alt="Detection on a Fishyscapes test frame" />
                : <div className="skeleton" style={{ width: '100%', height: '100%', borderRadius: 0 }} />}
            </div>
            {frame && (
              <>
                <div className="frame-chip chip-a">
                  <div className="k">Score on the object</div>
                  <div className="v mono accent-text">{frame.score_inside.toFixed(3)}</div>
                </div>
                <div className="frame-chip chip-b">
                  <div className="k">Score on background</div>
                  <div className="v mono">{frame.score_outside.toFixed(3)}</div>
                </div>
              </>
            )}
          </div>
          <p className="frame-note">A real output, not a mock-up. Needs the backend running; open the demo to explore more.</p>
        </div>
      </section>

      <section className="metrics">
        {METRICS.map((m) => <CountUpMetric key={m.k} {...m} />)}
      </section>
      <p className="metric-note">
        Fishyscapes Lost&amp;Found test half, held out from checkpoint selection. Large, close
        anomalies are a known weak spot (RoadAnomaly21); the project report has the full results.
      </p>

      <section className="section" id="finding">
        <Reveal><div className="section-eyebrow">The finding</div></Reveal>
        <Reveal delay={60}><h2 className="section-title">A score can look perfectly calibrated and still be wrong where it matters.</h2></Reveal>
        <Reveal delay={120}>
          <p className="section-lede">
            Calibration means a score of 0.8 should be right about 80% of the time. Averaged over the
            whole image it looks flawless, because almost every pixel is easy background. At the outline
            of an object, which decides how big the obstacle looks to a planner, it is not.
          </p>
        </Reveal>
        <Reveal delay={160} className="gap-compare">
          <div className="gap-row">
            <div className="gap-top">
              <span className="gap-label">Calibration error, whole image</span>
              <span className="gap-val mono" style={{ color: 'var(--good)' }}>{ECE_WHOLE.toFixed(4)}</span>
            </div>
            <div className="gap-track"><div className="gap-fill ok" style={{ '--w': `${(100 * ECE_WHOLE / ECE_SCALE_MAX).toFixed(2)}%` }} /></div>
          </div>
          <div className="gap-row">
            <div className="gap-top">
              <span className="gap-label">Calibration error, within 8 px of object outlines</span>
              <span className="gap-val mono" style={{ color: 'var(--bad)' }}>{ECE_EDGE.toFixed(4)}</span>
            </div>
            <div className="gap-track"><div className="gap-fill warn" style={{ '--w': `${(100 * ECE_EDGE / ECE_SCALE_MAX).toFixed(2)}%` }} /></div>
          </div>
        </Reveal>
        <Reveal delay={200}>
          <p className="gap-foot">
            The edge error is over <b>500 times</b> the whole-image figure, and no single temperature fixes
            both. The calibration analysis, the methods compared and the baseline results are in the project report.
          </p>
        </Reveal>
      </section>

      <section className="section">
        <Reveal><div className="section-eyebrow">How it works</div></Reveal>
        <Reveal delay={60}><h2 className="section-title">Three ideas, in the order they run.</h2></Reveal>
        <div className="steps">
          {STEPS.map((s, i) => (
            <Reveal className="step" key={s.n} delay={i * 110}>
              <div className="step-n mono">{s.n}</div>
              <div className="step-t">{s.t}</div>
              <div className="step-d">{s.d}</div>
            </Reveal>
          ))}
        </div>
      </section>

      <Reveal className="cta-band">
        <h2>See it on a road image of your own.</h2>
        <p>Upload an image or a short video and watch the detection, the anomaly score and the head disagreement.</p>
        <Link to="/demo" className="cta-btn cta-primary">Try the demo →</Link>
      </Reveal>
    </div>
  )
}
