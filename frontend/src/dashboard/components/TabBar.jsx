import { motion } from 'framer-motion'
import clsx from 'clsx'

export function TabBar({ tabs, activeTab, hasData, onTabChange }) {
  return (
    <div
      className="flex items-center gap-1.5 p-1.5 rounded-2xl overflow-x-auto glass-panel bg-white/[0.01] border-white/[0.05] shadow-inner"
    >
      {tabs.map((tab) => {
        const Icon = tab.icon
        const active = activeTab === tab.id
        const hasDot = hasData[tab.id] && !active
        return (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className="relative flex items-center gap-2.5 px-5 py-2.5 rounded-xl text-[11px] font-bold uppercase tracking-[0.1em] whitespace-nowrap transition-all duration-300 group"
            style={{
              color: active ? 'var(--text-primary)' : 'var(--text-muted)',
              background: active
                ? `linear-gradient(135deg, ${tab.accent}20, ${tab.accent}05)`
                : 'transparent',
            }}
          >
            <Icon 
              size={14} 
              style={{ color: active ? tab.accent : 'var(--text-faint)' }} 
              className="group-hover:scale-110 transition-transform"
            />
            {tab.label}
            {hasDot && (
              <span
                className="w-2 h-2 rounded-full absolute -top-0.5 -right-0.5"
                style={{
                  background: tab.accent,
                  boxShadow: `0 0 10px ${tab.accent}`,
                }}
              />
            )}
            {active && (
              <motion.div
                layoutId="activeTabGlow"
                className="absolute inset-0 rounded-xl pointer-events-none"
                style={{ 
                  border: `1px solid ${tab.accent}40`,
                  boxShadow: `0 0 20px ${tab.accent}15`
                }}
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
