import { useEffect, useState } from 'react'
import BigChart from '../components/BigChart'

function TimelineSkeleton() {
  return (
    <div className="timeline">
      {Array.from({ length: 2 }).map((_, i) => (
        <div className="timeline-step" key={i}>
          <div className="timeline-marker">
            <div className="timeline-dot skeleton" />
            {i === 0 && <div className="timeline-line" />}
          </div>
          <div className="timeline-card">
            <div className="skel-line" style={{ width: '45%', height: 15, marginBottom: 12 }} />
            <div className="skel-line" style={{ width: '85%', marginBottom: 8 }} />
            <div className="skel-line" style={{ width: '60%' }} />
          </div>
        </div>
      ))}
    </div>
  )
}

export default function HistoryPage() {
  const [history, setHistory] = useState(null)

  useEffect(() => {
    fetch('/api/history').then(r => r.json()).then(setHistory)
  }, [])

  return (
    <div className="content">
      <div className="page-head">
        <div className="stamp">CHAIN OF CUSTODY</div>
        <h1 className="headline">How this output was reached</h1>
        <p className="lede">
          Every attempt, in order, no skipped steps. The demo shown uses Checkpoint B's epoch-1 weights (best AUROC
          0.619), reached only after Checkpoint A ran to completion and was confirmed insufficient first.
        </p>
      </div>

      <div className="report-card">
        <div className="report-head">
          <h2>Phase 2b — calibration, whole image vs object edges</h2>
          <span className="report-best mono accent-text">against AUROC=0.9920</span>
        </div>
        <p className="report-config">
          L_calib fine-tune run 2026-09-25; edge-ECE study 2026-09-26 to 2026-10-04 (eval_spatial.py).
          Fishyscapes test half, all calibrators fit on the val half. The reliability diagram
          below is whole-image only (raw, temp T=1.71, L_calib).
        </p>
        <img
          src="/static/calibration_reliability.png"
          alt="Reliability diagram: confidence vs. actual accuracy for raw, temperature-scaled, and L_calib models"
          style={{ width: '100%', maxWidth: 520, display: 'block', border: '2px solid var(--line)', marginBottom: 16 }}
        />
        <table className="data-table">
          <thead><tr><th>Model</th><th>AUROC</th><th>AP</th><th>ECE (whole)</th><th>ECE (edges, r=8)</th></tr></thead>
          <tbody>
            <tr><td className="mono">Raw</td><td className="mono">0.9920</td><td className="mono">0.6215</td><td className="mono">0.0004</td><td className="mono">0.2273</td></tr>
            <tr><td className="mono">Temp T=1.71 (whole-image fit)</td><td className="mono">0.9924</td><td className="mono">0.6193</td><td className="mono">0.0020</td><td className="mono">0.1845</td></tr>
            <tr><td className="mono">Temp T=3.15 (50/50 fit)</td><td className="mono">0.9925</td><td className="mono">0.6177</td><td className="mono">0.0177</td><td className="mono">0.1277</td></tr>
            <tr className="best-row"><td className="mono">Temp T=4.36 (edge fit)</td><td className="mono">0.9925</td><td className="mono">0.6173</td><td className="mono">0.0420</td><td className="mono">0.0942</td></tr>
            <tr><td className="mono">T(d), head disagreement</td><td className="mono">0.9923</td><td className="mono">0.6106</td><td className="mono">0.0193</td><td className="mono">0.1312</td></tr>
            <tr><td className="mono">L_calib (CutMix-trained)</td><td className="mono">0.9924</td><td className="mono">0.6024</td><td className="mono">0.0005</td><td className="mono">0.2396</td></tr>
          </tbody>
        </table>
        <p className="calib-note" style={{ marginTop: 14 }}>
          Whole-image ECE hides the problem: raw scores 0.0004 there but 0.2273 at object edges,
          where the model is under-confident. No single temperature fixes both. Fitting to the
          edges (T=4.36) halves edge ECE but makes whole-image ECE 20× worse. A temperature that
          varies with head disagreement did no better than one fixed temperature (pre-registered
          control). L_calib, learned on training pastes, is worse than raw at the edges. Every
          comparison is a paired bootstrap over the 50 test images; full numbers in PLAN.md.
        </p>
      </div>

      {!history ? (
        <TimelineSkeleton />
      ) : (
        <div className="timeline">
          {history.map((step, i) => (
            <div className={`timeline-step${step.id === 'checkpoint_b' ? ' current' : ''}`} key={step.id}>
              <div className="timeline-marker">
                <div className="timeline-dot mono">{i + 1}</div>
                {i < history.length - 1 && <div className="timeline-line" />}
              </div>
              <div className="timeline-card">
                <div className="timeline-head">
                  <h2>{step.label}</h2>
                  <div className="timeline-auroc mono">
                    AUROC <span className="accent-text">{step.best_auroc.toFixed(4)}</span>
                  </div>
                </div>
                <p className="timeline-desc">{step.description}</p>
                {step.epochs && (
                  <>
                    <div className="timeline-sub mono">
                      full {step.epochs.length}-epoch log — precise, unedited
                    </div>
                    <BigChart epochs={step.epochs} bestEpoch={step.epochs.indexOf(Math.max(...step.epochs)) + 1} />
                    <table className="data-table">
                      <thead><tr><th>Epoch</th><th>AUROC</th></tr></thead>
                      <tbody>
                        {step.epochs.map((v, e) => (
                          <tr key={e} className={v === step.best_auroc ? 'best-row' : ''}>
                            <td className="mono">{e + 1}</td>
                            <td className="mono">{v.toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </>
                )}
                {step.id === 'checkpoint_b' && (
                  <div className="current-flag mono">← CURRENT DEMO CHECKPOINT</div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
