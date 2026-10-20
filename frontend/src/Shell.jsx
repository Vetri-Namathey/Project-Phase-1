import { useEffect, useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import ErrorBoundary from './components/ErrorBoundary'

const LINKS = [
  { to: '/', label: 'Home', end: true },
  { to: '/demo', label: 'Demo' },
]

export default function Shell() {
  const location = useLocation()
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => { window.scrollTo(0, 0) }, [location.pathname])

  return (
    <div className="page">
      <div className={`navbar${scrolled ? ' scrolled' : ''}`}>
        <Link to="/" className="brand">
          <span className="brand-mark" />
          <span className="brand-name">TwinGuard</span>
        </Link>
        <nav className="navlinks">
          {LINKS.map(l => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.end}
              className={({ isActive }) => `navlink${isActive ? ' active' : ''}`}
            >
              {l.label}
            </NavLink>
          ))}
        </nav>
      </div>
      <div key={location.pathname} className="route-fade">
        <ErrorBoundary><Outlet /></ErrorBoundary>
      </div>
      <footer className="site-footer">
        <span>TwinGuard — road anomaly detection prototype</span>
        <span>SegFormer-B5 frozen encoder · 3 independently seeded heads · Fishyscapes Lost&amp;Found</span>
      </footer>
    </div>
  )
}
