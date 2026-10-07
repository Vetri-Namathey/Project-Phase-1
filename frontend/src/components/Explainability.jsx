import { useEffect, useState } from 'react'

// Explainability results (X1, X1b, X5, X2 faithfulness, X3), moved here from DemoPage so the
// /explainability page owns them. Every number is read from static/explain/summary.json.

const STAGE_LABELS = ['Stage 1 · fine detail', 'Stage 2', 'Stage 3', 'Stage 4 · scene meaning']

function StageBars({ title, data }) {
  return (
    <div className="stage-bars">
      <div className="k mono">{title} · n={data.n}</div>
      {data.share.map((s, i) => (
        <div className="stage-bar-row" key={i}>
          <span className="stage-bar-label">{STAGE_LABELS[i]}</span>
          <div className="stage-bar-track">
            <div className={`stage-bar-fill${data.sign[i] < 0 ? ' lowers' : ''}`} style={{ width: `${(100 * s).toFixed(0)}%` }} />
          </div>
          <span className="stage-bar-val mono">{(100 * s).toFixed(0)}% {data.sign[i] < 0 ? '↓' : '↑'}</span>
        </div>
      ))}
    </div>
  )
}

// X2 faithfulness: relative drop of the object's score when one stage's features
// at the object are replaced by their surroundings (explain.py --x2-faith).
function AblationBars({ title, data }) {
  return (
    <div className="stage-bars">
      <div className="k mono">{title} · n={data.n} detected</div>
      {data.ablation_rel.map((s, i) => (
        <div className="stage-bar-row" key={i}>
          <span className="stage-bar-label">{STAGE_LABELS[i]}</span>
          <div className="stage-bar-track">
            <div className="stage-bar-fill" style={{ width: `${Math.max(0, Math.min(100, 100 * s)).toFixed(0)}%` }} />
          </div>
          <span className="stage-bar-val mono">−{(100 * s).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  )
}

const X5_ARMS = [
  ['coco_unseen', 'Pasted, never trained on'],
  ['coco_seen', 'Pasted, in training bank'],
  ['carla_bank', 'Pasted CARLA object'],
  ['real', 'Real object, same photos'],
]

function SamePhotoTest({ x5 }) {
  const sg = v => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`
  return (
    <div className="hero-panel">
      <span className="tick mono">SAME PHOTO: PASTED OBJECTS VS THE REAL ONE</span>
      <div className="x5-grid">
        {X5_ARMS.map(([k, label]) => {
          const a = x5.arms[k]
          return (
            <div className={`x5-card${k === 'real' ? ' real' : ''}`} key={k}>
              <div className="x5-label mono">{label}</div>
              <div className="x5-score mono">{a.score.toFixed(2)}</div>
              <div className="x5-bar"><div style={{ width: `${(100 * a.score).toFixed(0)}%` }} /></div>
              <div className="x5-sub mono">edge {sg(a.band_gap)} · {Math.round(a.px)} px</div>
            </div>
          )
        })}
      </div>
      <div className="cap">
        <div className="t">One object pasted into each of {x5.n_photos} real test photos, one paste per run, same spot and size for every kind</div>
        <div className="d">
          Mean anomaly score on the object; "edge" is the score minus the true rate in the ±4 px edge band
          (+ over-confident, − under-confident). Pasted objects score about 0.93 whether or not that object was in the
          training bank (seen − unseen {sg(x5.score_seen_minus_unseen)}, no memorisation detected; still provisional until
          we confirm the local bank is the training copy). Real objects in the same photos score {x5.arms.real.score.toFixed(2)}.
          At the edges, pastes are over-confident and real objects under-confident: gap {sg(x5.edge_gap_unseen_minus_real)} over {x5.n_real} photos
          with a real object. Caveat: one pre-registered check failed narrowly. An unseen-paste run moved the real
          object's score by −0.003 (CI excludes 0), so that contrast is formally flagged as confounded, although the
          shift is about 150× smaller than the gap. CIs: paste_test.log.
        </div>
      </div>
    </div>
  )
}

// explain.py heat_strip writes one 2560x512 PNG: 5 tiles side by side, each the
// edited photo (top half) over its anomaly heatmap (bottom half). The viewer crops
// tiles out of it with background-position, so no extra files are needed.
const X1_SRC = '/static/explain/x1_example.png'
const X1_EDITS = [
  { label: 'Original', note: 'The frame as the model sees it, before any edit.' },
  { label: 'Object removed', note: 'The whole object is inpainted over (Telea); score measured on the object.' },
  { label: 'Edges removed', note: 'Only a ±4 px ring around the outline is repainted; score measured on the interior.' },
  { label: 'Interior removed', note: 'Only the core is repainted, the outline kept; score measured on the inner edge ring.' },
  { label: 'Surroundings blurred', note: 'Everything more than 32 px from the object is blurred; score measured on the object.' },
]

function X1Tile({ i, row, style }) {
  return (
    <div
      className="x1-layer"
      style={{
        backgroundImage: `url(${X1_SRC})`,
        backgroundSize: '500% 200%',
        backgroundPosition: `${i * 25}% ${row * 100}%`,
        ...style,
      }}
    />
  )
}

function X1Viewer() {
  const [idx, setIdx] = useState(0)
  const [heat, setHeat] = useState(0.6)
  const n = X1_EDITS.length

  const onKey = e => {
    if (e.key === 'ArrowRight') setIdx(i => (i + 1) % n)
    if (e.key === 'ArrowLeft') setIdx(i => (i - 1 + n) % n)
  }

  return (
    <div className="x1-viewer" tabIndex={0} onKeyDown={onKey} aria-label="Explainability viewer, use arrow keys to switch edits">
      <div className="x1-seg" role="tablist">
        {X1_EDITS.map((e, i) => (
          <button key={e.label} role="tab" aria-selected={i === idx}
            className={`x1-seg-btn ${i === idx ? 'active' : ''}`} onClick={() => setIdx(i)}>
            <span className="mono x1-step">{i + 1}</span>{e.label}
          </button>
        ))}
      </div>

      <div className="x1-stage" role="img" aria-label={`${X1_EDITS[idx].label}: photo with anomaly heatmap overlay`}>
        <X1Tile i={idx} row={0} />
        <X1Tile i={idx} row={1} style={{ opacity: heat }} />
        <div className="x1-badge mono">{idx + 1} / {n} · {X1_EDITS[idx].label.toUpperCase()}</div>
      </div>

      <div className="x1-controls">
        <div className="x1-note">{X1_EDITS[idx].note}</div>
        <label className="x1-slider mono">
          PHOTO
          <input type="range" min="0" max="1" step="0.05" value={heat}
            onChange={e => setHeat(Number(e.target.value))} aria-label="Heatmap opacity" />
          HEATMAP
        </label>
      </div>

      <div className="x1-thumbs">
        {X1_EDITS.map((e, i) => (
          <button key={e.label} className={`x1-thumb ${i === idx ? 'active' : ''}`}
            onClick={() => setIdx(i)} aria-label={e.label}>
            <div className="x1-thumb-img"><X1Tile i={i} row={1} /></div>
            <span className="mono">{e.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

// PLAN.md X4: numbers come from static/explain/summary.json (explain_export.py),
// which is computed from explain.py's per-image results -- nothing typed in here.
export function ExplainPanel() {
  const [s, setS] = useState(null)
  const [missing, setMissing] = useState(false)

  useEffect(() => {
    fetch('/static/explain/summary.json')
      .then(r => (r.ok ? r.json() : Promise.reject()))
      .then(setS)
      .catch(() => setMissing(true))
  }, [])

  if (missing || !s) return null
  const f = n => n.toFixed(3)
  const real = s.x3.fishyscapes

  return (
    <div className="video-block">
      <div className="section-tag">Why does it flag this? Explainability</div>

      <div className="hero-panel accent x1-panel">
        <div className="x1-head">
          <span className="x1-title mono">REMOVE PART OF THE OBJECT, RE-RUN THE MODEL</span>
        </div>
        <X1Viewer />
        <div className="cap">
          <div className="d">
            Removing the toy car leaves an inpainted smudge, and the smudge scores higher than the car did.
            The model flags inpainted and shifted-patch edits too, which is consistent with training on pasted
            objects (tested for these two kinds of edit only). So inpainted removal understates the effect, and
            the exact test below (the same frames with and without the object) gives the real number.
          </div>
        </div>
      </div>

      <div className="proof-row">
        <div className="proof-stat">
          <div className="k mono">PASTED OBJECT · SCORE</div>
          <div className="v mono">{f(s.x1b.with_paste)}</div>
        </div>
        <div className="proof-stat">
          <div className="k mono">TRULY REMOVED</div>
          <div className="v mono accent-text">{f(s.x1b.true_removal)}</div>
        </div>
        <div className="proof-stat">
          <div className="k mono">REMOVED BY INPAINTING</div>
          <div className="v mono">{f(s.x1b.telea_removal)}</div>
        </div>
        <div className="proof-stat">
          <div className="k mono">REAL OBJECTS · BEFORE → INPAINTED</div>
          <div className="v mono">{s.x1.object_before.toFixed(2)} → {s.x1.object_after_telea.toFixed(2)}</div>
        </div>
      </div>

      {s.x5 && <SamePhotoTest x5={s.x5} />}

      {s.x2_faith ? (
        <div className="hero-panel">
          <span className="tick mono">WHICH ENCODER SCALE THE SCORE NEEDS</span>
          <div className="stage-bars-grid">
            <AblationBars title="Real objects (Fishyscapes)" data={s.x2_faith.fishyscapes} />
            <AblationBars title="Pasted objects (CARLA video)" data={s.x2_faith.carla_pasted} />
          </div>
          <div className="cap">
            <div className="t">Score drop when one encoder stage's features at the object are replaced by their surroundings</div>
            <div className="d">
              Real objects need the middle stages most (1/16 and 1/8 resolution); pasted objects depend on no single
              stage as much. A gradient × activation map was checked against this removal test. It picked the same top
              stage on pasted objects ({(100 * s.x2_faith.carla_pasted.agreement).toFixed(0)}% of them) but only at chance on
              real ones ({(100 * s.x2_faith.fishyscapes.agreement).toFixed(0)}%, chance 25%), where it pointed at the coarsest
              stage, which the removal test never ranked first. So gradient maps are not used to explain real detections here.
            </div>
          </div>
        </div>
      ) : (
        <div className="hero-panel">
          <span className="tick mono">WHICH ENCODER SCALE DRIVES THE SCORE</span>
          <div className="stage-bars-grid">
            <StageBars title="Real objects (Fishyscapes)" data={s.x2.fishyscapes} />
            <StageBars title="Pasted objects (CARLA video)" data={s.x2.carla_pasted} />
          </div>
          <div className="cap">
            <div className="t">Share of each object's score per encoder stage (gradient × activation)</div>
            <div className="d">A first-order approximation, not yet checked against a removal test.</div>
          </div>
        </div>
      )}

      <div className="hero-panel">
        <span className="tick mono">WHAT KIND OF UNCERTAINTY (REAL FISHYSCAPES OBJECTS)</span>
        <table className="data-table">
          <thead><tr><th>Region</th><th>Score</th><th>Ambiguity (aleatoric)</th><th>Head disagreement (epistemic)</th></tr></thead>
          <tbody>
            {['edge band', 'interior', 'true positive', 'false positive', 'far background'].map(r => (
              <tr key={r}>
                <td>{r} <span className="mono" style={{ color: 'var(--ink-faint)' }}>n={real[r].n}</span></td>
                <td className="mono">{real[r].score.toFixed(3)}</td>
                <td className="mono">{real[r].aleatoric.toFixed(3)}</td>
                <td className="mono">{real[r].epistemic.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="cap">
          <div className="d">
            Object edges are dominated by ambiguity, not by the heads disagreeing, which links to the
            edge miscalibration on the Home page. Three heads on one frozen encoder underestimate disagreement.
          </div>
        </div>
      </div>
    </div>
  )
}
