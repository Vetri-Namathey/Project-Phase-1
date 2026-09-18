export default function BigChart({ epochs, bestEpoch }) {
  const max = Math.max(...epochs, 0.7)
  return (
    <div className="bigchart">
      {epochs.map((v, i) => (
        <div className="bigchart-col" key={i}>
          <div className="bigchart-val mono">{v.toFixed(3)}</div>
          <div
            className={`bigchart-bar${i === bestEpoch - 1 ? ' best' : ''}`}
            style={{ height: `${(v / max) * 100}%` }}
          />
          <div className="bigchart-n mono">E{i + 1}</div>
        </div>
      ))}
    </div>
  )
}
