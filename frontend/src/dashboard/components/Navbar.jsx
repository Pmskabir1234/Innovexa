import { motion } from 'framer-motion'
import { Sun, Moon, Activity, WifiOff, Cpu, Menu } from 'lucide-react'
import { Link } from 'react-router-dom'
import clsx from 'clsx'

export function Navbar({ theme, onToggleTheme, health, healthLoading, onOpenSidebar }) {
  const isOnline = health?.status === 'ok'

  return (
    <header className="fixed top-0 left-0 right-0 z-[100] h-14 glass-panel border-x-0 border-t-0 bg-white/[0.01]">
      <div className="max-w-screen-2xl mx-auto px-6 h-full flex items-center justify-between gap-4">

          <div className="flex items-center gap-4">
            {/* Mobile Hamburger */}
            <button
              onClick={onOpenSidebar}
              className="lg:hidden p-2 rounded-lg text-[var(--text-muted)] hover:bg-white/[0.05] transition-colors"
              aria-label="Open sidebar"
            >
              <Menu size={20} />
            </button>

            {/* Logo */}
            <Link to="/">
              <motion.div
                className="flex items-center gap-3.5 cursor-pointer group"
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <div className="relative">
                  <div
                    className="w-9 h-9 rounded-xl flex items-center justify-center shadow-xl group-hover:scale-105 transition-transform duration-300"
                    style={{
                      background: 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)',
                      boxShadow: '0 0 20px rgba(34,197,94,0.3)',
                    }}
                  >
                    <Activity size={18} className="text-black" />
                  </div>
                  {/* Pulse ring when online */}
                  {isOnline && (
                    <span className="absolute -top-0.5 -right-0.5 w-3 h-3 rounded-full bg-emerald-400 border-2 border-[#030711] animate-pulse" />
                  )}
                </div>
                <div className="leading-tight">
                  <div className="flex items-center gap-2">
                    <span
                      className="font-bold text-[var(--text-primary)] text-base tracking-tight uppercase"
                      style={{ fontFamily: 'var(--font-heading)' }}
                    >
                      Core<span className="text-primary">Insight</span>
                    </span>
                    <span
                      className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-md text-[9px] font-black tracking-[0.1em] uppercase shadow-sm"
                      style={{
                        background: 'var(--color-primary-dim)',
                        border: '1px solid var(--color-primary-border)',
                        color: 'var(--color-primary)',
                      }}
                    >
                      AI Engine
                    </span>
                  </div>
                  <span className="hidden sm:block text-[10px] font-medium tracking-wider uppercase opacity-50" style={{ color: 'var(--text-muted)' }}>
                    Predictive Intelligence
                  </span>
                </div>
              </motion.div>
            </Link>
          </div>

          {/* Center — model info */}
          {health?.failure_model && (
            <motion.div
              className="hidden lg:flex items-center gap-4 px-5 py-1.5 rounded-full bg-white/[0.02] border border-white/[0.05]"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <div className="flex items-center gap-2">
                <Cpu size={12} className="text-primary/60" />
                <span className="text-[10px] font-black uppercase tracking-[0.15em] text-[var(--text-muted)]">
                  Neural Model:
                </span>
                <span className="text-xs font-mono font-bold text-primary/90">
                  {health.failure_model}
                </span>
              </div>
              <div className="w-px h-3 bg-white/10" />
              {health.failure_model_accuracy && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-black uppercase tracking-[0.15em] text-[var(--text-muted)]">Accuracy:</span>
                  <span className="text-xs font-mono font-bold text-emerald-400">
                    {(health.failure_model_accuracy * 100).toFixed(1)}%
                  </span>
                </div>
              )}
            </motion.div>
          )}

          {/* Right */}
          <motion.div
            className="flex items-center gap-3"
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            {/* Backend status pill */}
            <div
              className={clsx(
                'hidden sm:flex items-center gap-2 px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-[0.1em]',
                'border transition-all duration-500 shadow-sm'
              )}
              style={{
                background: healthLoading
                  ? 'rgba(255,255,255,0.03)'
                  : isOnline
                  ? 'rgba(34,197,94,0.08)'
                  : 'rgba(239,68,68,0.08)',
                borderColor: healthLoading
                  ? 'rgba(255,255,255,0.08)'
                  : isOnline
                  ? 'rgba(34,197,94,0.2)'
                  : 'rgba(239,68,68,0.2)',
                color: healthLoading
                  ? 'var(--text-muted)'
                  : isOnline
                  ? '#22c55e'
                  : '#ef4444',
              }}
            >
              {healthLoading ? (
                <div className="w-1.5 h-1.5 rounded-full bg-slate-500 animate-pulse" />
              ) : isOnline ? (
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_#10b981]" />
              ) : (
                <WifiOff size={10} />
              )}
              {healthLoading ? 'Syncing…' : isOnline ? 'System Live' : 'Link Lost'}
            </div>

            <div className="w-px h-4 bg-white/10 mx-1 hidden sm:block" />

            {/* Theme toggle */}
            <button
              onClick={onToggleTheme}
              className="p-2.5 rounded-xl hover:bg-white/[0.05] transition-all duration-200 group relative"
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark'
                ? <Sun size={16} className="text-slate-400 group-hover:text-amber-400 group-hover:rotate-45 transition-all duration-500" />
                : <Moon size={16} className="text-slate-400 group-hover:text-indigo-400 group-hover:-rotate-12 transition-all duration-500" />
              }
            </button>
          </motion.div>
        </div>
      </header>


  )
}
