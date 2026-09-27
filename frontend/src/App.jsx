import { Route, Routes } from 'react-router-dom'
import Shell from './Shell'
import HomePage from './pages/HomePage'
import DemoPage from './pages/DemoPage'
import HistoryPage from './pages/HistoryPage'
import TrainingPage from './pages/TrainingPage'

export default function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/runs" element={<HistoryPage />} />
        <Route path="/training" element={<TrainingPage />} />
      </Route>
    </Routes>
  )
}
