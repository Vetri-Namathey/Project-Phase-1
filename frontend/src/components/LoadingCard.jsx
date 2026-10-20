import { useEffect, useState } from 'react'

// Shown while the server scores an upload. The server does not stream progress, so the bar is an
// ESTIMATE from the clip length (about 2.5 s of processing per second of video) and is labelled so.

const VIDEO_STEPS = [
  ['Uploading the file', 0],
  ['Decoding and sampling frames at 10 per second', 0.08],
  ['Scoring every frame with the three heads', 0.16],
  ['Encoding the result video', 0.9],
]
const IMAGE_STEPS = [
  ['Uploading the image', 0],
  ['Scoring it with the three heads', 0.2],
  ['Drawing boxes and heat maps', 0.8],
]

export default function LoadingCard({ kind, name, previewUrl, duration }) {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const t0 = performance.now()
    const id = setInterval(() => setElapsed((performance.now() - t0) / 1000), 250)
    return () => clearInterval(id)
  }, [])

  const isVideo = kind === 'video'
  const seconds = isVideo ? Math.min(duration || 10, 30) : 0
  const estimate = isVideo ? 4 + 2.5 * seconds : 3
  const fraction = Math.min(0.95, elapsed / estimate)
  const steps = isVideo ? VIDEO_STEPS : IMAGE_STEPS
  const active = steps.reduce((a, [, from], i) => (fraction >= from ? i : a), 0)
  const over = elapsed > estimate * 1.15

  return (
    <div className="panel-soft loading-card" role="status" aria-live="polite">
      <div className="loading-media">
        {previewUrl && (isVideo
          ? <video src={previewUrl} muted loop playsInline autoPlay />
          : <img src={previewUrl} alt="" />)}
        <div className="loading-scan" />
        <div className="loading-shade" />
        <div className="loading-badge mono">{isVideo ? 'ANALYSING VIDEO' : 'ANALYSING IMAGE'}</div>
      </div>

      <div className="loading-body">
        <div className="loading-title">{isVideo ? 'Scoring your video frame by frame' : 'Running the model on your image'}</div>
        <div className="loading-file mono">{name}</div>

        <div className="loading-bar" aria-hidden="true"><i style={{ width: `${(fraction * 100).toFixed(1)}%` }} /></div>
        <div className="loading-meta mono">
          <span>{Math.round(fraction * 100)}% (estimate)</span>
          <span>{elapsed.toFixed(0)} s{!over ? ` of about ${estimate.toFixed(0)} s` : ' · taking longer than expected, still working'}</span>
        </div>

        <ol className="loading-steps">
          {steps.map(([label], i) => (
            <li key={label} className={i < active ? 'done' : i === active ? 'now' : ''}>
              <span className="dot" aria-hidden="true">{i < active ? '✓' : ''}</span>
              {label}
            </li>
          ))}
        </ol>
        <p className="loading-note">
          {isVideo
            ? 'The explanation of the first detection appears on the right as soon as the video is ready.'
            : 'The explanation appears on the right in a moment.'}
        </p>
      </div>
    </div>
  )
}
