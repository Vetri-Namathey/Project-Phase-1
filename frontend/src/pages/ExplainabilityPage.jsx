import { useEffect, useState } from 'react'
import { ExplainPanel } from '../components/Explainability'

// The tests in the order they were run: each one exists because the one before it
// left a question open, so the numbering is real sequence, not decoration.
const TESTS = [
  { id: 'X1', name: 'Remove it', q: 'Does the score depend on the object, its outline, or its surroundings?',
    how: 'Repaint the whole object, only its edge band, only its core, or blur everything around it, then re-run the model.' },
  { id: 'X1b', name: 'Remove it exactly', q: 'How much of X1 is the repaint itself being flagged?',
    how: 'On the CARLA route the same frame exists with and without each pasted object, so the clean frame is a perfect removal.' },
  { id: 'X5', name: 'Same photo, pasted vs real', q: 'Do pasted objects behave differently from real ones when the photo is identical?',
    how: 'Paste one object into each real test photo (one per run, same spot and size), compare it with the real object in that photo.' },
  { id: 'X2', name: 'Which encoder stage', q: 'Does a gradient map show what the score actually relies on?',
    how: 'Compare gradient × activation per stage with a removal test that blanks one stage at the object.' },
  { id: 'X3', name: 'What kind of uncertainty', q: 'Are the edges uncertain because the heads disagree, or because the pixels are ambiguous?',
    how: 'Split the three heads’ uncertainty into disagreement (epistemic) and shared ambiguity (aleatoric) by region.' },
]

function Findings() {
  const [s, setS] = useState(null)
  useEffect(() => {
    fetch('/static/explain/summary.json').then(r => (r.ok ? r.json() : null)).then(setS).catch(() => setS(null))
  }, [])
  if (!s) return null
  const items = [
    { k: 'X1b · PASTE → TRULY REMOVED', v: `${s.x1b.with_paste.toFixed(2)} → ${s.x1b.true_removal.toFixed(2)}`,
      d: `Inpainting leaves ${s.x1b.telea_removal.toFixed(2)}: the fill itself gets flagged.` },
    s.x5 && { k: 'X5 · PASTED vs REAL, SAME PHOTO', v: `${s.x5.arms.coco_unseen.score.toFixed(2)} vs ${s.x5.arms.real.score.toFixed(2)}`,
      d: `Edges: pastes over-confident, real under-confident (gap +${s.x5.edge_gap_unseen_minus_real.toFixed(2)}).` },
    s.x5 && { k: 'X5 · SEEN − UNSEEN OBJECT', v: `${s.x5.score_seen_minus_unseen >= 0 ? '+' : '−'}${Math.abs(s.x5.score_seen_minus_unseen).toFixed(3)}`,
      d: 'No detectable memorisation: the model reacts to pasting, not to the object.' },
    s.x2_faith && { k: 'X2 · GRADIENT MAP AGREES (REAL)', v: `${(100 * s.x2_faith.fishyscapes.agreement).toFixed(0)}%`,
      d: 'Chance is 25%, so gradient maps are not trusted for real objects; the removal test is used.' },
  ].filter(Boolean)
  return (
    <div className="proof-row xp-findings">
      {items.map(it => (
        <div className="proof-stat" key={it.k}>
          <div className="k mono">{it.k}</div>
          <div className="v mono">{it.v}</div>
          <div className="xp-find-d">{it.d}</div>
        </div>
      ))}
    </div>
  )
}

export default function ExplainabilityPage() {
  return (
    <div className="content">
      <div className="stage-head">
        <h2>Explainability — why it flags what it flags</h2>
        <p>
          Five evaluation-only tests on the same model as the demo (no retraining). Each test was set up with its
          pass/fail rule written down before it ran, and every number on this page is read from the exported
          results file, not typed in.
        </p>
      </div>

      <Findings />

      <div className="section-tag">The five tests, in the order they were run</div>
      <div className="xp-tests">
        {TESTS.map((t, i) => (
          <div className="xp-test" key={t.id}>
            <div className="xp-test-head">
              <span className="xp-test-n mono">{i + 1}</span>
              <span className="xp-test-id mono">{t.id}</span>
              <span className="xp-test-name">{t.name}</span>
            </div>
            <div className="xp-test-q">{t.q}</div>
            <div className="xp-test-how">{t.how}</div>
          </div>
        ))}
      </div>

      <ExplainPanel />

      <div className="hero-panel xp-takeaway">
        <span className="tick mono">WHAT IT ADDS UP TO</span>
        <div className="cap">
          <div className="d">
            The model has learned what a pasted object looks like. Pastes score near 1 with over-confident edges whether
            or not that object was in training, while real objects in the same photos score far lower with
            under-confident edges. That is why calibration learned on pastes moves real edges the wrong way (Home page),
            and why the checks matter: an unchecked gradient map would have pointed at the wrong encoder stage.
          </div>
        </div>
      </div>
    </div>
  )
}
