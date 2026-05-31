import { motion } from 'framer-motion'
import { Sun, Moon, Activity, WifiOff, Cpu, Menu } from 'lucide-react'
import { Link } from 'react-router-dom'
import clsx from 'clsx'

export function Navbar({ theme, onToggleTheme, health, healthLoading, onOpenSidebar }) {
  const isOnline = health?.status === 'ok'

  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-14">
      {/* Gradient border bottom */}
      <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-[var(--border-subtle)] to-transparent" />

      <div
        className="h-full"
        style={{
          background: 'var(--surface-elevated)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
        }}
      >
        <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 h-full flex items-center justify-between gap-4">

          <div className="flex items-center gap-4">
            {/* Mobile Hamburger */}
            <button
              onClick={onOpenSidebar}
              className="lg:hidden p-2 rounded-lg text-[var(--text-muted)] hover:bg-[var(--surface-elevated)]"
              aria-label="Open sidebar"
            >
              <Menu size={20} />
            </button>

            {/* Logo */}
            <Link to="/">
              <motion.div
                className="flex items-center gap-3 cursor-pointer"
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4 }}
              >
                <div className="relative">
                  <div
                    className="w-8 h-8 rounded-xl flex items-center justify-center shadow-lg"
                    style={{
                      background: 'linear-gradient(135deg, var(--color-primary) 0%, hsl(119,99%,36%) 100%)',
                      boxShadow: '0 4px 14px var(--color-primary-dim)',
                    }}
                  >
                    <Activity size={15} className="text-white" />
                  </div>
                  {/* Pulse ring when online */}
                  {isOnline && (
                    <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-emerald-400 border-2 border-[var(--bg)]" />
                  )}
                </div>
                <div className="leading-tight">
                  <div className="flex items-center gap-1.5">
                    <span
                      className="font-bold text-[var(--text-primary)] text-sm tracking-tight"
                      style={{ fontFamily: 'var(--font-heading)' }}
                    >
                      CoreInsight
                    </span>
                    <span
                      className="hidden sm:inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold tracking-wider"
                      style={{
                        background: 'var(--color-primary-dim)',
                        border: '1px solid var(--color-primary-border)',
                        color: 'var(--color-primary)',
                      }}
                    >
                      AI
                    </span>
                  </div>
                  <span className="hidden sm:block text-[11px]" style={{ color: 'var(--text-muted)' }}>
                    Predictive Maintenance
                  </span>
                </div>
              </motion.div>
            </Link>
          </div>

          {/* Center — model info */}
          {health?.failure_model && (
            <motion.div
              className="hidden lg:flex items-center gap-2"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              <Cpu size={12} style={{ color: 'var(--text-muted)' }} />
              <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
                {health.failure_model}
              </span>
              {health.failure_model_accuracy && (
                <span
                  className="text-xs font-semibold font-mono px-1.5 py-0.5 rounded"
                  style={{
                    background: 'var(--color-primary-dim)',
                    color: 'var(--color-primary)',
                  }}
                >
                  {(health.failure_model_accuracy * 100).toFixed(1)}%
                </span>
              )}
            </motion.div>
          )}

          {/* Right */}
          <motion.div
            className="flex items-center gap-2"
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4 }}
          >
            {/* Backend status pill */}
            <div
              className={clsx(
                'hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium',
                'border transition-all duration-300'
              )}
              style={{
                background: healthLoading
                  ? 'var(--input-bg)'
                  : isOnline
                  ? 'var(--color-primary-dim)'
                  : 'rgba(239,68,68,0.06)',
                borderColor: healthLoading
                  ? 'var(--border-subtle)'
                  : isOnline
                  ? 'var(--color-primary-border)'
                  : 'rgba(239,68,68,0.2)',
                color: healthLoading
                  ? 'var(--text-muted)'
                  : isOnline
                  ? 'var(--color-primary)'
                  : '#ef4444',
              }}
            >
              {healthLoading ? (
                <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: 'var(--text-muted)' }} />
              ) : isOnline ? (
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              ) : (
                <WifiOff size={10} />
              )}
              {healthLoading ? 'Connecting…' : isOnline ? 'Online' : 'Offline'}
            </div>

            {/* Theme toggle */}
            <button
              onClick={onToggleTheme}
              className="btn-ghost p-2 rounded-lg"
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark'
                ? <Sun size={15} className="text-slate-400 hover:text-amber-400 transition-colors" />
                : <Moon size={15} className="text-slate-400 transition-colors" />
              }
            </button>
          </motion.div>
        </div>
      </div>
    </header>

  )
}
