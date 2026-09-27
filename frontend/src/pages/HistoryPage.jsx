import { useEffect, useState } from 'react'

const STATUS_LABEL = {
  baseline: 'BASELINE',
  superseded: 'SUPERSEDED',
  current: 'CURRENT DEMO CHECKPOINT',
  ablation: 'ABLATION',
  finding: 'REPORTED AS A FINDING',
}

function fmt(v) {
  return v === null || v === undefined ? '—' : v.toFixed(4)
}

function TimelineSkeleton() {
  return (
    <div className="timeline">
      {Array.from({ length: 3 }).map((_, i) => (
        <div className="timeline-step" key={i}>
          <div className="timeline-marker">
            <div className="timeline-dot skeleton" />
            {i < 2 && <div className="timeline-line" />}
          </div>
          <div className="timeline-card">
            <div className="skel-line" style={{ width: '45%', height: 15, marginBottom: 12 }} />
            <div className="skel-line" style={{ width: '85%', marginBottom: 8 }} />
          </div>
        </div>
      ))}
    </div>
  )
}

// Every number below is in runs/RESULTS.md with the log that produced it.
function FindingsCard() {
  return (
    <div className="report-card">
      <div className="report-head">
        <h2>What the evaluation found</h2>
        <span className="report-best mono accent-text">3-head run 1, test half</span>
      </div>
      <table className="data-table">
        <thead><tr><th>Question</th><th>Measured</th><th>Verdict</th></tr></thead>
        <tbody>
          <tr><td>Beats the untrained baseline?</td><td className="mono">AP 0.750 vs 0.011 · FPR@95 0.021 vs 1.000</td><td>Yes, clearly</td></tr>
          <tr><td>Hurts normal segmentation?</td><td className="mono">mIoU 0.770 (1-head 0.768); no class moves &gt; 0.02</td><td>No</td></tr>
          <tr><td>3 heads detect better than 1?</td><td className="mono">AP 0.750 vs 0.777 test, 0.730 vs 0.705 val</td><td>Equal within noise</td></tr>
          <tr><td>What do 3 heads add?</td><td className="mono">disagreement AUROC 0.985 in 1 pass vs MC-Dropout 0.968 in 10</td><td>Uncertainty for free</td></tr>
          <tr><td>Outline quality (UBQ)</td><td className="mono">40/85 objects found · spill 2.5 px · miss 6.7 px · boundary F1 0.57</td><td>Stable over thresholds 0.3–0.7</td></tr>
          <tr><td>L_calib beats temperature at edges?</td><td className="mono">edge ECE 0.189 vs raw 0.195 vs temp(edges) 0.106</td><td>No (finding)</td></tr>
          <tr><td>Separates bad camera from novel object?</td><td className="mono">both signals ×28–31 under noise</td><td>No (finding)</td></tr>
          <tr><td>Safety trigger meets &lt;5% / &gt;90%?</td><td className="mono">best: 20% false / 47.5% caught</td><td>No, not yet</td></tr>
          <tr><td>Generalises to RoadAnomaly21?</td><td className="mono">AUROC 0.721 vs baseline 0.871 (10 labelled images)</td><td>No: large objects fail</td></tr>
        </tbody>
      </table>
      <p className="calib-note" style={{ marginTop: 14 }}>
        Robustness: camera noise drops AP from 0.75 to 0.55 and motion blur to 0.57; fog barely matters (0.74).
      </p>
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
          Every attempt, in order, including the ones that failed. Fishyscapes Lost&amp;Found test half; the
          checkpoint is always selected on the separate val half.
        </p>
      </div>

      <FindingsCard />

      {!history ? (
        <TimelineSkeleton />
      ) : (
        <div className="timeline">
          {history.map((step, i) => (
            <div className={`timeline-step${step.status === 'current' ? ' current' : ''}`} key={step.id}>
              <div className="timeline-marker">
                <div className="timeline-dot mono">{i + 1}</div>
                {i < history.length - 1 && <div className="timeline-line" />}
              </div>
              <div className="timeline-card">
                <div className="timeline-head">
                  <h2>{step.label}</h2>
                  <div className="timeline-auroc mono">
                    AUROC <span className="accent-text">{fmt(step.auroc)}</span> · AP {fmt(step.ap)} · FPR@95 {fmt(step.fpr95)}
                  </div>
                </div>
                <p className="timeline-desc">{step.description}</p>
                {step.epochs_ap && (
                  <table className="data-table">
                    <thead><tr><th>Epoch</th>{step.epochs_ap.map((_, e) => <th key={e}>{e + 1}</th>)}</tr></thead>
                    <tbody>
                      <tr>
                        <td className="mono">test AP</td>
                        {step.epochs_ap.map((v, e) => <td className="mono" key={e}>{v === null ? '·' : v.toFixed(3)}</td>)}
                      </tr>
                    </tbody>
                  </table>
                )}
                <div className="current-flag mono">{STATUS_LABEL[step.status]}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
