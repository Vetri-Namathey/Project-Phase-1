import { Component } from 'react'

// A crash in one component must never blank the whole site.
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('TwinGuard UI error:', error, info?.componentStack)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    return (
      <div className="content">
        <div className="err-card" role="alert">
          <div className="err-title">Something went wrong showing this page</div>
          <p className="err-text">{String(error.message || error)}</p>
          <p className="err-hint">
            If you just changed the backend, restart <span className="mono">server.py</span> and reload this page.
          </p>
          <button className="btn-soft" onClick={() => this.setState({ error: null })}>Try again</button>
        </div>
      </div>
    )
  }
}
