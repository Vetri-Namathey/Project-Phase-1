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
        <h1 className="headline">Training data — CutMix composites</h1>
        <p className="lede">
          8 composites made by the actual training pipeline: Cityscapes frames (validation split, fixed seed) with
          objects from the 2000-object bank — tiles cut from 45 CARLA frames plus COCO cutouts — pasted on road or
          sidewalk at realistic sizes. The highlight is the mask the heads were told counts as anomalous. The CARLA
          tiles are cut with those frames' own masks, and at least two of those masks are misaligned, so some CARLA
          pastes may not show the intended object.
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
