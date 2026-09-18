import { NavLink, Outlet, useLocation } from 'react-router-dom'

export default function Shell() {
  const location = useLocation()

  return (
    <div className="page">
      <div className="navbar">
        <div className="brand">
          <span className="brand-mark">▣</span>
          <span className="brand-name">TWINGUARD</span>
        </div>
        <nav className="navlinks">
          <NavLink to="/" end className={({ isActive }) => `navlink${isActive ? ' active' : ''}`}>
            Detection Demo
          </NavLink>
          <NavLink to="/runs" className={({ isActive }) => `navlink${isActive ? ' active' : ''}`}>
            Training Runs
          </NavLink>
          <NavLink to="/training" className={({ isActive }) => `navlink${isActive ? ' active' : ''}`}>
            Training Data
          </NavLink>
        </nav>
      </div>
      <div key={location.pathname} className="route-fade">
        <Outlet />
      </div>
      <footer className="site-footer">
        <span>TwinGuard — OOD detection prototype</span>
        <span>Fishyscapes Lost&amp;Found · mit-b5 frozen encoder · 3 independently-seeded heads</span>
      </footer>
    </div>
  )
}
