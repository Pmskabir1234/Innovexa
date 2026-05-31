import Navbar from "./components/Navbar"
import HeroSection from "./components/HeroSection"

export default function App() {
  return (
    <div className="bg-hero-bg min-h-screen relative font-sora selection:bg-primary selection:text-primary-foreground">
      {/* Navigation Bar */}
      <Navbar />

      {/* Hero Section with 3D Background */}
      <HeroSection />
    </div>
  )
}
