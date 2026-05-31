import { motion } from 'framer-motion'
import { MetricCard } from './ui/MetricCard'
import { GaugeChart } from './ui/GaugeChart'
import { BarChart2, AlertTriangle, Zap } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

const stagger = {
  container: { animate: { transition: { staggerChildren: 0.08 } } },
  item: {
    initial: { opacity: 0, y: 14 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } },
  },
}

const BAR_COLORS = [
  'var(--color-primary)', 'hsl(119,99%,40%)', 'hsl(119,99%,35%)', 'hsl(119,99%,30%)',
  'hsl(119,99%,50%)', 'hsl(119,99%,60%)', 'hsl(119,99%,70%)', 'hsl(119,99%,80%)',
]

export function PredictResult({ data }) {
  if (!data) return null
  const { failure_probability_percent, risk_category, feature_importance } = data

  const chartData = (feature_importance || []).slice(0, 6).map((f) => ({
    name: f.feature || f.name || 'Unknown',
    value: parseFloat((f.importance || f.value || 0).toFixed(4)),
  }))

  const riskAccent =
    risk_category === 'Critical' ? 'red'
    : risk_category === 'High' ? 'orange'
    : risk_category === 'Medium' ? 'amber'
    : 'green'

  return (
    <motion.div
      className="space-y-4"
      variants={stagger.container}
      initial="initial"
      animate="animate"
    >
      {/* Metrics */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          label="Failure Probability"
          value={`${failure_probability_percent?.toFixed(2)}%`}
          icon={AlertTriangle}
          accent="orange"
          delay={0}
        />
        <MetricCard
          label="Risk Category"
          value={risk_category}
          icon={BarChart2}
          accent={riskAccent}
          delay={0.08}
        />
      </div>

      {/* Gauge */}
      <motion.div variants={stagger.item} className="card-elevated p-8 flex justify-center">
        <GaugeChart value={failure_probability_percent} title="Failure Risk %" size={200} variant="risk" />
      </motion.div>

      {/* Feature importance */}
      {chartData.length > 0 && (
        <motion.div variants={stagger.item} className="card p-5">
          <div className="flex items-center gap-2.5 mb-4">
            <div className="w-6 h-6 rounded-lg flex items-center justify-center" style={{ background: 'var(--color-primary-dim)' }}>
              <Zap size={12} style={{ color: 'var(--color-primary)' }} />
            </div>
            <span className="section-title" style={{ color: 'var(--text-muted)' }}>Top Contributing Factors</span>
            <div className="flex-1 h-px" style={{ background: 'var(--border-subtle)' }} />
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 4, right: 20, top: 4, bottom: 4 }}>
              <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} width={140} tickLine={false} axisLine={false} />
              <Tooltip
                cursor={{ fill: 'var(--border-subtle)' }}
                contentStyle={{
                  background: 'var(--surface-elevated)',
                  border: '1px solid var(--border-accent)',
                  borderRadius: '0.75rem',
                  fontSize: 12,
                  color: 'var(--text-secondary)',
                }}
                formatter={(v) => [v.toFixed(4), 'Importance']}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {chartData.map((_, i) => (
                  <Cell
                    key={i}
                    fill={BAR_COLORS[i % BAR_COLORS.length]}
                    style={{ filter: `drop-shadow(0 0 4px color-mix(in srgb, ${BAR_COLORS[i % BAR_COLORS.length]}, transparent 60%))` }}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      )}
    </motion.div>
  )
}
