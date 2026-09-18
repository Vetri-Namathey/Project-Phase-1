import { useEffect, useState } from 'react'
import FadeImage from '../components/FadeImage'

function TrainCardSkeleton() {
  return (
    <div className="train-card">
      <div className="skeleton" style={{ aspectRatio: '4/3' }} />
      <div className="cap"><div className="skel-line" style={{ width: '80%' }} /></div>
    </div>
  )
}

export default function TrainingPage() {
  const [samples, setSamples] = useState(null)
  const [showOverlay, setShowOverlay] = useState(true)

  useEffect(() => {
    fetch('/api/training-samples').then(r => r.json()).then(setSamples)
  }, [])

  return (
    <div className="content">
      <div className="page-head">
        <div className="stamp">EXHIBIT B</div>
        <h1 className="headline">Training data — synthetic CARLA anomalies</h1>
        <p className="lede">
          No live CARLA connection and no recorded video feed — these are 8 of the 50 individual static frames
          actually used to train the OOD heads (<span className="mono">generate_anomalies.py</span>), each a
          different random object spawned at a different simulated moment, not a continuous recording of one scene.
          The amber highlight is the real ground-truth mask pasted via CutMix — exactly what the model was told
          counts as anomalous in that frame.
        </p>
      </div>

      {!samples ? (
        <>
          <div className="skel-line" style={{ width: 200, height: 34, marginBottom: 16 }} />
          <div className="train-grid">
            {Array.from({ length: 8 }).map((_, i) => <TrainCardSkeleton key={i} />)}
          </div>
        </>
      ) : (
        <>
          <button className="toggle-btn" onClick={() => setShowOverlay(v => !v)}>
            {showOverlay ? 'SHOWING: MASK OVERLAY' : 'SHOWING: RAW FRAME'} — CLICK TO TOGGLE
          </button>
          <div className="train-grid">
            {samples.map(s => (
              <div className="train-card" key={s.id}>
                <FadeImage
                  key={showOverlay ? 'overlay' : 'raw'}
                  src={showOverlay ? s.panels.overlay : s.panels.raw}
                  alt={s.source_file}
                />
                <div className="cap"><div className="t mono">{s.source_file}</div></div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
