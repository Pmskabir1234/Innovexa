import { motion } from 'framer-motion'
import clsx from 'clsx'

const ACCENT_STYLES = {
  cyan:    { icon: 'rgba(6,182,212,0.15)',   text: '#22d3ee',  border: 'rgba(6,182,212,0.25)'   },
  orange:  { icon: 'rgba(249,115,22,0.15)',  text: '#fb923c',  border: 'rgba(249,115,22,0.25)'  },
  green:   { icon: 'rgba(16,185,129,0.15)',  text: '#34d399',  border: 'rgba(16,185,129,0.25)'  },
  red:     { icon: 'rgba(239,68,68,0.15)',   text: '#f87171',  border: 'rgba(239,68,68,0.25)'   },
  amber:   { icon: 'rgba(245,158,11,0.15)',  text: '#fbbf24',  border: 'rgba(245,158,11,0.25)'  },
  purple:  { icon: 'rgba(168,85,247,0.15)',  text: '#c084fc',  border: 'rgba(168,85,247,0.25)'  },
  default: { icon: 'var(--border-subtle)', text: 'var(--text-muted)',  border: 'var(--border-subtle)' },
}

export function MetricCard({ label, value, sub, accent, icon: Icon, className, delay = 0 }) {
  const style = ACCENT_STYLES[accent] || ACCENT_STYLES.default

  return (
    <motion.div
      className={clsx('card p-5 flex flex-col gap-3 relative overflow-hidden group', className)}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -4, transition: { duration: 0.3 } }}
      style={{ borderColor: style.border }}
    >
      {/* Dynamic Background Glow */}
      <div 
        className="absolute -right-4 -bottom-4 w-24 h-24 blur-[40px] opacity-0 group-hover:opacity-20 transition-opacity duration-500 rounded-full"
        style={{ background: style.text }}
      />

      {/* Premium top accent line */}
      <div
        className="absolute inset-x-0 top-0 h-[2px] opacity-0 group-hover:opacity-100 transition-opacity duration-300"
        style={{ background: `linear-gradient(90deg, transparent, ${style.text}, transparent)` }}
      />

      <div className="flex items-center justify-between relative z-10">
        <span className="text-[10px] font-black uppercase tracking-[0.2em] text-[var(--text-muted)] group-hover:text-[var(--text-secondary)] transition-colors">{label}</span>
        {Icon && (
          <div
            className="w-8 h-8 rounded-xl flex items-center justify-center transition-transform duration-300 group-hover:scale-110 group-hover:rotate-6"
            style={{ 
              background: style.icon,
              border: `1px solid ${style.text}20`
            }}
          >
            <Icon size={14} style={{ color: style.text }} />
          </div>
        )}
      </div>

      <div className="space-y-1 relative z-10">
        <span
          className="text-2xl font-black leading-none tracking-tighter block"
          style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-heading)' }}
        >
          {value}
        </span>
        {sub && (
          <span className="text-[10px] font-medium tracking-wide block uppercase opacity-60" style={{ color: 'var(--text-muted)' }}>{sub}</span>
        )}
      </div>
    </motion.div>
  )
}
