import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
import Navbar from "./components/Navbar"
import HeroSection from "./components/HeroSection"
import Dashboard from "./dashboard/App"

function LandingPage() {
  return (
    <div className="bg-hero-bg min-h-screen relative font-sora selection:bg-primary selection:text-primary-foreground">
      {/* Navigation Bar */}
      <Navbar />

      {/* Hero Section with 3D Background */}
      <HeroSection />
    </div>
  )
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </Router>
  )
}
