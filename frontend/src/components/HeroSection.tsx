import React, { Suspense } from "react"
import { Link } from "react-router-dom"

// Lazy load the Spline 3D component to optimize the initial page load speed.
const Spline = React.lazy(() => import("@splinetool/react-spline"))

export default function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-end bg-hero-bg overflow-hidden select-none">
      {/* Absolute container for the 3D Spline scene */}
      <div className="absolute inset-0 w-full h-full">
        <Suspense fallback={<div className="absolute inset-0 bg-hero-bg animate-pulse" />}>
          <Spline
            scene="https://prod.spline.design/Slk6b8kz3LRlKiyk/scene.splinecode"
            className="w-full h-full"
          />
        </Suspense>
      </div>

      {/* Dark overlay ensuring high readability of the text content */}
      <div className="absolute inset-0 bg-black/30 z-[1] pointer-events-none" />

      {/* Content Container (Content is placed in bottom-left and clicks are passed to Spline except for CTA buttons) */}
      <div className="relative z-10 pointer-events-none w-full max-w-[90%] sm:max-w-md lg:max-w-2xl px-6 md:px-10 pb-10 md:pb-16 pt-32 flex flex-col items-start text-left">
        
        {/* Title Heading: Delay 0.2s */}
        <h2 
          className="opacity-0 animate-fade-up text-[clamp(2.75rem,8vw,5.5rem)] font-bold leading-[1.05] tracking-[-0.05em] text-foreground mb-2 md:mb-4 uppercase"
          style={{ animationDelay: "0.2s" }}
        >
          Core <span className="text-primary">Insight</span>
        </h2>

        {/* Subheading: Delay 0.4s */}
        <p 
          className="opacity-0 animate-fade-up text-foreground/80 text-[clamp(1.125rem,2.5vw,1.875rem)] font-light mb-3 md:mb-6"
          style={{ animationDelay: "0.4s" }}
        >
          Stop Guessing. Start Predicting.
        </p>

        {/* Description Text: Delay 0.55s */}
        <p 
          className="opacity-0 animate-fade-up text-muted-foreground text-[clamp(0.875rem,1.5vw,1.15rem)] leading-relaxed font-light mb-4 md:mb-8"
          style={{ animationDelay: "0.55s" }}
        >
          Python MVP for anomaly detection, failure risk estimation, root-cause hints, and explainable maintenance decisions. Includes synthetic industrial sensor data, FastAPI endpoints, and a Streamlit chat shell.
        </p>

        {/* CTA Buttons: Delay 0.7s (pointer-events-auto is set to re-enable clicks) */}
        <div 
          className="opacity-0 animate-fade-up flex flex-wrap gap-3 font-bold pointer-events-auto"
          style={{ animationDelay: "0.7s" }}
        >
          <Link 
            to="/dashboard"
            className="bg-[#22c55e] text-black px-6 py-3 md:px-8 md:py-4 text-xs tracking-widest uppercase font-bold rounded-sm cursor-pointer hover:brightness-110 active:scale-[0.97] transition-all duration-200 outline-none inline-block text-center border-2 border-[#22c55e] shadow-[0_0_15px_rgba(34,197,94,0.3)]"
          >
            Demo
          </Link>
          
          <a 
            href="https://github.com/Pmskabir1234/CoreInsight"
            target="_blank"
            rel="noopener noreferrer"
            className="bg-white text-background px-6 py-3 md:px-8 md:py-4 text-xs tracking-widest uppercase font-semibold rounded-sm cursor-pointer hover:brightness-90 active:scale-[0.97] transition-all duration-200 outline-none inline-block text-center"
          >
            Github
          </a>
        </div>

        {/* Trust/Metadata line: Delay 0.85s */}
        <p 
          className="opacity-0 animate-fade-up text-muted-foreground/60 text-[10px] md:text-xs tracking-widest uppercase font-light mt-6 md:mt-10"
          style={{ animationDelay: "0.85s" }}
        >
          Detect failures before they happen. Powered by AI, built by Deadline Sprinters.
        </p>
      </div>
    </section>
  )
}
