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
          <h2>Phase 2b — calibration (L_calib), run 2026-09-25</h2>
          <span className="report-best mono accent-text">against AUROC=0.9920</span>
        </div>
        <p className="report-config">
          Joint fine-tune of the verified checkpoint with a differentiable soft-ECE surrogate
          (SoftECELoss), against raw and post-hoc temperature scaling. Fishyscapes test half.
        </p>
        <img
          src="/static/calibration_reliability.png"
          alt="Reliability diagram: confidence vs. actual accuracy for raw, temperature-scaled, and L_calib models"
          style={{ width: '100%', maxWidth: 520, display: 'block', border: '2px solid var(--line)', marginBottom: 16 }}
        />
        <table className="data-table">
          <thead><tr><th>Model</th><th>AUROC</th><th>AP</th><th>FPR@95</th><th>ECE</th></tr></thead>
          <tbody>
            <tr><td className="mono">Raw</td><td className="mono">0.9920</td><td className="mono">0.6215</td><td className="mono">0.0290</td><td className="mono">0.0004</td></tr>
            <tr><td className="mono">Temp-scaled</td><td className="mono">0.9924</td><td className="mono">0.6193</td><td className="mono">0.0287</td><td className="mono">0.0020</td></tr>
            <tr className="best-row"><td className="mono">L_calib</td><td className="mono">0.9924</td><td className="mono">0.6024</td><td className="mono">0.0293</td><td className="mono">0.0005</td></tr>
          </tbody>
        </table>
        <p className="calib-note" style={{ marginTop: 14 }}>
          Honest read, not a clean win: temperature scaling made whole-image ECE <em>worse</em>
          (0.0004 → 0.0020), and L_calib's ECE (0.0005) is barely different from raw's already-tiny
          0.0004. At Fishyscapes' 0.28% positive-pixel rate, whole-image ECE is dominated by trivial
          true negatives and can't separate these three models — boundary-only ECE and UBQ
          (in progress, see PLAN.md) are the metrics that could.
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
