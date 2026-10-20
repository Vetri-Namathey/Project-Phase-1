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

// server.py EXPERIMENT_HISTORY id of the checkpoint the demo actually loads (config.PRIMARY_RAW).
const CURRENT_ID = 'twinguard_2000bank'

export default function HistoryPage() {
  const [history, setHistory] = useState(null)

  useEffect(() => {
    fetch('/api/history').then(r => r.json()).then(setHistory)
  }, [])

  return (
    <div className="content">
      <div className="page-head">
        <div className="stamp">Training runs</div>
        <h1 className="headline">How this model was reached.</h1>
        <p className="lede">
          Every attempt, in order, with no skipped steps. The demo uses the last entry: the 3-head model trained
          on the 2000-object CARLA + COCO bank (test AUROC 0.9920, AP 0.6218; train.log). Calibration results
          are in the project report.
        </p>
      </div>

      {!history ? (
        <TimelineSkeleton />
      ) : (
        <div className="timeline">
          {history.map((step, i) => (
            <div className={`timeline-step${step.id === CURRENT_ID ? ' current' : ''}`} key={step.id}>
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
                {step.id === CURRENT_ID && (
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
