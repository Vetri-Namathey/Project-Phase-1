import { useEffect, useMemo, useRef, useState } from 'react'

// Streams the backend's explanation (explain_live.py) like a chat reply: a typing
// indicator, then the text word by word, then the numbers and any picture. The text is
// built from measured numbers on the server; there is no language model.

const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

const WORD_MS = 26
const PAUSE_MS = 140          // extra pause after a sentence
const DOTS_MS = 620
const BETWEEN_MS = 380

function Avatar() {
  return <div className="chat-avatar" aria-hidden="true"><span /></div>
}

function Dots({ label }) {
  return (
    <div className="chat-row">
      <Avatar />
      <div className="chat-bubble chat-dots" role="status" aria-label={label || 'Thinking'}>
        <i /><i /><i />
        {label && <span className="chat-dots-label">{label}</span>}
      </div>
    </div>
  )
}

function Message({ msg, text, done, images, caret }) {
  return (
    <div className="chat-row chat-in">
      <Avatar />
      <div className="chat-bubble">
        <div className="chat-title">{msg.title}</div>
        <p className="chat-text">
          {text}
          {caret && <span className="chat-caret" />}
        </p>
        {done && msg.facts?.length > 0 && (
          <div className="chat-facts">
            {msg.facts.map(([k, v]) => (
              <div className="chat-fact" key={k}>
                <span className="k">{k}</span>
                <span className="v mono">{v}</span>
              </div>
            ))}
          </div>
        )}
        {done && msg.image && images?.[msg.image] && (
          <img className="chat-image" alt={msg.title} src={`data:image/png;base64,${images[msg.image]}`} />
        )}
      </div>
    </div>
  )
}

export default function AnalystChat({ state, data, question, error, loadingLabel, expanded = false, onToggleExpanded }) {
  const messages = useMemo(() => (state === 'ready' && data ? data.messages : []), [state, data])
  // The parent remounts this component (key) for every new explanation, so state starts fresh.
  const instant = prefersReducedMotion()
  const [idx, setIdx] = useState(instant ? messages.length : 0)
  const [stage, setStage] = useState(instant ? 'done' : 'dots')   // dots | typing | done
  const [words, setWords] = useState(0)
  const [skipped, setSkipped] = useState(instant)
  const scroller = useRef(null)

  const n = messages.length
  const finished = n > 0 && (skipped || idx >= n)

  useEffect(() => {
    if (!n || skipped || idx >= n) return
    let t
    if (stage === 'dots') {
      t = setTimeout(() => { setStage('typing'); setWords(0) }, idx === 0 ? DOTS_MS + 300 : DOTS_MS)
    } else if (stage === 'typing') {
      const all = messages[idx].text.split(' ')
      if (words < all.length) {
        const w = all[words]
        t = setTimeout(() => setWords(words + 1), WORD_MS + (/[.;:]$/.test(w) ? PAUSE_MS : 0))
      } else {
        t = setTimeout(() => setStage('done'), 120)
      }
    } else if (stage === 'done') {
      t = setTimeout(() => { setIdx(idx + 1); setStage('dots'); setWords(0) }, BETWEEN_MS + 200)
    }
    return () => clearTimeout(t)
  }, [stage, words, idx, n, skipped, messages])

  const atStart = words === 0
  useEffect(() => {
    const el = scroller.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [idx, stage, atStart, finished])

  useEffect(() => {
    if (!expanded) return
    const onKey = e => { if (e.key === 'Escape') onToggleExpanded?.() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [expanded, onToggleExpanded])

  const replay = () => { setIdx(0); setStage('dots'); setWords(0); setSkipped(false) }

  return (
    <>
    {expanded && <div className="chat-backdrop" onClick={onToggleExpanded} aria-hidden="true" />}
    <aside className={`chat${expanded ? ' expanded' : ''}`} aria-label="Explanation of the flagged regions">
      <header className="chat-head">
        <Avatar />
        <div className="chat-who">
          <div className="chat-name">TwinGuard analyst</div>
          <div className="chat-sub">Built from measured numbers on this frame. Not a language model.</div>
        </div>
        {state === 'ready' && n > 0 && (finished
          ? <button className="chat-btn" onClick={replay}>Replay</button>
          : <button className="chat-btn" onClick={() => setSkipped(true)}>Skip</button>)}
        <button className="chat-btn chat-expand" onClick={onToggleExpanded}
          aria-label={expanded ? 'Close the large view' : 'Open the explanation in a large view'}>
          {expanded ? 'Close' : 'Expand'}
        </button>
      </header>

      <div className="chat-scroll" ref={scroller} aria-live="polite">
        {state === 'idle' && (
          <div className="chat-empty">
            <Avatar />
            <p>Upload a road image or video. When something is flagged I explain why: what the segmenter sees there, whether the score survives removing it, which part of the network carries it, and how sure the three heads are.</p>
          </div>
        )}

        {state !== 'idle' && question && (
          <div className="chat-row chat-user chat-in"><div className="chat-bubble">{question}</div></div>
        )}

        {state === 'loading' && <Dots label={loadingLabel || 'Running the removal, ablation and uncertainty tests…'} />}
        {state === 'error' && (
          <div className="chat-row chat-in"><Avatar /><div className="chat-bubble chat-error">{error}</div></div>
        )}

        {state === 'ready' && messages.map((m, i) => {
          if (!skipped && i > idx) return null
          const current = !skipped && i === idx
          if (current && stage === 'dots') return <Dots key={m.id} />
          const typing = current && stage === 'typing'
          const text = typing ? m.text.split(' ').slice(0, words).join(' ') : m.text
          const done = skipped || i < idx || (current && stage === 'done')
          return <Message key={m.id} msg={m} text={text} done={done} images={data.images} caret={typing} />
        })}

        {state === 'ready' && finished && (
          <div className="chat-foot">
            Threshold {data.threshold.toFixed(3)} (fitted on validation data) · analysed in {(data.timing_ms / 1000).toFixed(1)} s.
            Without ground truth these are explanations of the model, not proof that it is right.
          </div>
        )}
      </div>
    </aside>
    </>
  )
}
