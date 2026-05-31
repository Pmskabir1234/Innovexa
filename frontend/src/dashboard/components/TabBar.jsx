import { motion } from 'framer-motion'
import clsx from 'clsx'

export function TabBar({ tabs, activeTab, hasData, onTabChange }) {
  return (
    <div
      className="flex items-center gap-1 p-1 rounded-2xl overflow-x-auto bg-[var(--surface)] border border-[var(--border-subtle)]"
    >
      {tabs.map((tab) => {
        const Icon = tab.icon
        const active = activeTab === tab.id
        const hasDot = hasData[tab.id] && !active
        return (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className="relative flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold whitespace-nowrap transition-all duration-200"
            style={{
              color: active ? 'var(--text-primary)' : 'var(--text-muted)',
              background: active
                ? `linear-gradient(135deg, ${tab.accent}25, ${tab.accent}15)`
                : 'transparent',
              border: active ? `1px solid ${tab.accent}30` : '1px solid transparent',
              boxShadow: active ? `0 0 16px ${tab.accent}15` : 'none',
            }}
          >
            <Icon size={14} style={{ color: active ? tab.accent : 'var(--text-faint)' }} />
            {tab.label}
            {hasDot && (
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{
                  background: tab.accent,
                  boxShadow: `0 0 4px ${tab.accent}`,
                  marginLeft: '2px'
                }}
              />
            )}
            {active && (
              <motion.div
                layoutId="activeTabGlow"
                className="absolute inset-0 rounded-xl pointer-events-none"
                style={{ border: `1px solid ${tab.accent}40` }}
                initial={false}
                transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
              />
            )}
          </button>
        )
      })}
    </div>
  )
}
