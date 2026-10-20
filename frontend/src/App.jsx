import { Navigate, Route, Routes } from 'react-router-dom'
import Shell from './Shell'
import HomePage from './pages/HomePage'
import DemoPage from './pages/DemoPage'

// Only Home and Demo remain. Explainability moves into the demo (DISCUSSION_NOTES.md, section 3);
// training runs and training data go to the report. Old links land on the demo / home.
export default function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/explainability" element={<Navigate to="/demo" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
