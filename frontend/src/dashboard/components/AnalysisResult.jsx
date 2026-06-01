import { motion } from 'framer-motion'
import { GaugeChart } from './ui/GaugeChart'
import { StatusBanner } from './ui/StatusBanner'
import { MetricCard } from './ui/MetricCard'
import { Accordion } from './ui/Accordion'
import {
  Activity, BarChart2, TrendingUp, TrendingDown, Minus,
  FileText, AlertTriangle, Clock, Zap, Brain,
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
} from 'recharts'
import clsx from 'clsx'

/* ── helpers ── */
const stagger = {
  container: { animate: { transition: { staggerChildren: 0.07 } } },
  item: {
    initial: { opacity: 0, y: 16 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } },
  },
}

const STATUS_CFG = {
  Normal:   { bg: 'rgba(16,185,129,0.08)',  border: 'rgba(16,185,129,0.2)',  text: '#34d399', bar: '#10b981' },
  Warning:  { bg: 'rgba(245,158,11,0.08)',  border: 'rgba(245,158,11,0.2)',  text: '#fbbf24', bar: '#f59e0b' },
  Critical: { bg: 'rgba(239,68,68,0.08)',   border: 'rgba(239,68,68,0.2)',   text: '#f87171', bar: '#ef4444' },
}

const TREND_CFG = {
  Stable:   { icon: Minus,        color: '#64748b', label: 'Stable'   },
  Rising:   { icon: TrendingUp,   color: '#fb923c', label: 'Rising'   },
  Falling:  { icon: TrendingDown, color: '#22d3ee', label: 'Falling'  },
  Volatile: { icon: Activity,     color: '#c084fc', label: 'Volatile' },
}

/* ── sub-components ── */
function SectionHeader({ icon: Icon, title, accent = 'var(--color-primary)' }) {
  return (
    <div className="flex items-center gap-3 mb-6 group">
      <div
        className="w-8 h-8 rounded-xl flex items-center justify-center transition-transform group-hover:scale-110 duration-300"
        style={{ 
          background: `linear-gradient(135deg, ${accent}30, ${accent}10)`,
          border: `1px solid ${accent}40`,
          boxShadow: `0 0 20px ${accent}15`
        }}
      >
        <Icon size={14} style={{ color: accent }} />
      </div>
      <span className="text-[10px] font-black uppercase tracking-[0.2em]" style={{ color: 'var(--text-muted)' }}>{title}</span>
      <div className="flex-1 h-px bg-gradient-to-r from-white/[0.1] to-transparent ml-2" />
    </div>
  )
}

function DiagnosticsGrid({ diagnostics }) {
  if (!diagnostics?.length) return (
    <p className="text-sm font-light italic" style={{ color: 'var(--text-faint)' }}>No diagnostics available for this period.</p>
  )

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      {diagnostics.map((d, i) => {
        const pct = Math.min(100, Math.abs(d.deviation_percent || 0))
        const cfg = STATUS_CFG[d.status] || STATUS_CFG.Normal

        return (
          <motion.div
            key={d.parameter}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="rounded-2xl p-4 space-y-3 glass-panel hover:bg-white/[0.04] transition-colors"
            style={{
              borderColor: `${cfg.bar}30`,
              boxShadow: pct > 80 ? `0 0 20px ${cfg.bar}10` : 'none'
            }}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold tracking-tight uppercase" style={{ color: 'var(--text-secondary)' }}>
                {d.parameter}
              </span>
              <span
                className="badge !px-2 !py-0.5"
                style={{
                  background: `${cfg.bar}15`,
                  color: cfg.text,
                  border: `1px solid ${cfg.bar}25`,
                }}
              >
                {d.status}
              </span>
            </div>

            {/* Progress bar */}
            <div className="space-y-1.5">
              <div
                className="h-1.5 rounded-full overflow-hidden bg-white/[0.05]"
              >
                <motion.div
                  className="h-full rounded-full"
                  style={{ 
                    background: `linear-gradient(90deg, ${cfg.bar}80, ${cfg.bar})`, 
                    boxShadow: `0 0 10px ${cfg.bar}40` 
                  }}
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 1, delay: i * 0.1, ease: [0.16, 1, 0.3, 1] }}
                />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono font-medium" style={{ color: 'var(--text-faint)' }}>
                  {d.actual_value?.toFixed(2)} <span className="opacity-40">vs</span> {d.safe_max}
                </span>
                <span className="text-[10px] font-black font-mono tracking-wider" style={{ color: cfg.text }}>
                  {pct.toFixed(1)}% DEVIATION
                </span>
              </div>
            </div>

            {d.explanation && (
              <p className="text-[11px] leading-relaxed font-light italic border-t border-white/[0.03] pt-2" style={{ color: 'var(--text-muted)' }}>
                "{d.explanation}"
              </p>
            )}
          </motion.div>
        )
      })}
    </div>
  )
}

function FeatureImportanceChart({ importance }) {
  if (!importance?.length) return (
    <p className="text-sm italic font-light" style={{ color: 'var(--text-faint)' }}>Feature analysis not available.</p>
  )

  const data = importance.slice(0, 8).map((f) => ({
    name: (f.feature || f.name || 'Unknown').toUpperCase(),
    value: parseFloat((f.importance || f.value || 0).toFixed(4)),
  }))

  const BAR_COLORS = [
    '#22c55e', '#16a34a', '#15803d', '#166534',
    '#4ade80', '#86efac', '#bbf7d0', '#dcfce7',
  ]

  return (
    <div className="h-[250px] w-full pt-2">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30, top: 0, bottom: 0 }}>
          <XAxis
            type="number"
            tick={{ fontSize: 9, fill: 'var(--text-faint)', fontWeight: 700 }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fontSize: 9, fill: 'var(--text-muted)', fontWeight: 800, letterSpacing: '0.05em' }}
            width={120}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            cursor={{ fill: 'rgba(255,255,255,0.03)' }}
            contentStyle={{
              background: 'rgba(15, 23, 42, 0.9)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '1rem',
              fontSize: 10,
              boxShadow: '0 10px 25px -5px rgba(0,0,0,0.5)'
            }}
            itemStyle={{ fontWeight: 700, textTransform: 'uppercase' }}
            formatter={(v) => [v.toFixed(4), 'WEIGHT']}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={12}>
            {data.map((_, i) => (
              <Cell
                key={i}
                fill={BAR_COLORS[i % BAR_COLORS.length]}
                style={{ filter: `drop-shadow(0 0 5px ${BAR_COLORS[i % BAR_COLORS.length]}40)` }}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function TrendInsightRow({ t, i }) {
  const cfg = TREND_CFG[t.trend] || TREND_CFG.Stable
  const Icon = cfg.icon

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: i * 0.08, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="flex items-start gap-4 p-4 rounded-2xl transition-all duration-300 group glass-panel border-white/[0.03]"
      whileHover={{ scale: 1.01, background: 'rgba(255,255,255,0.04)', borderColor: 'rgba(255,255,255,0.08)' }}
    >
      <div
        className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0 mt-0.5 group-hover:rotate-6 transition-transform"
        style={{ background: `${cfg.color}15`, border: `1px solid ${cfg.color}20` }}
      >
        <Icon size={16} style={{ color: cfg.color }} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-3 mb-1 flex-wrap">
          <span className="text-xs font-black uppercase tracking-wider" style={{ color: 'var(--text-primary)' }}>{t.metric}</span>
          <span
            className="badge !text-[9px] !px-2 !py-0.5"
            style={{ background: `${cfg.color}10`, color: cfg.color, border: `1px solid ${cfg.color}20` }}
          >
            {cfg.label.toUpperCase()}
          </span>
        </div>
        <p className="text-xs leading-relaxed font-light" style={{ color: 'var(--text-muted)' }}>{t.detail}</p>
      </div>
    </motion.div>
  )
}

/* ── main export ── */
export function AnalysisResult({ data }) {
  if (!data) return null

  const {
    failure_probability_percent,
    anomaly_score,
    decision_priority,
    risk_category,
    health_score,
    parameter_diagnostics,
    feature_importance,
    trend_insights,
    comparison_note,
    engineering_report,
    structured_analysis,
  } = data

  const visuals = structured_analysis?.visualizations || []
  const rootCause = structured_analysis?.root_cause_analysis || []
  const historicalComparison = structured_analysis?.historical_comparison || []

  return (
    <motion.div
      className="space-y-6"
      variants={stagger.container}
      initial="initial"
      animate="animate"
    >
      {/* Status banner */}
      <motion.div variants={stagger.item} className="mb-2">
        <StatusBanner risk={risk_category} priority={decision_priority} />
      </motion.div>

      {/* Top metrics row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="FAILURE RISK"
          value={`${failure_probability_percent?.toFixed(1)}%`}
          icon={AlertTriangle}
          accent="orange"
          delay={0.1}
        />
        <MetricCard
          label="ANOMALY SCORE"
          value={anomaly_score?.toFixed(3)}
          icon={Activity}
          accent="cyan"
          delay={0.15}
        />
        <MetricCard
          label="PRIORITY"
          value={decision_priority.toUpperCase()}
          icon={Zap}
          accent="amber"
          delay={0.2}
        />
        <MetricCard
          label="RISK LEVEL"
          value={risk_category.toUpperCase()}
          icon={BarChart2}
          accent={risk_category === 'Critical' ? 'red' : risk_category === 'High' ? 'orange' : risk_category === 'Medium' ? 'amber' : 'green'}
          delay={0.25}
        />
      </div>

      {/* Gauges */}
      <motion.div variants={stagger.item} className="card-elevated p-8 relative overflow-hidden group">
        <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-[80px] -mr-32 -mt-32 pointer-events-none" />
        <SectionHeader icon={Activity} title="Predictive Health Indices" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 pt-4">
          <div className="flex flex-col items-center">
            <GaugeChart value={health_score} title="System Health Index" size={200} variant="health" />
            <p className="text-[10px] font-black uppercase tracking-[0.2em] text-primary/40 mt-4">Real-time Stability</p>
          </div>
          <div className="flex flex-col items-center">
            <GaugeChart value={failure_probability_percent} title="Failure Probability" size={200} variant="risk" />
            <p className="text-[10px] font-black uppercase tracking-[0.2em] text-orange-500/40 mt-4">Risk Estimation</p>
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Diagnostics */}
        <motion.div variants={stagger.item} className="card p-6">
          <SectionHeader icon={Activity} title="Parameter Diagnostics" accent="var(--color-primary)" />
          <DiagnosticsGrid diagnostics={parameter_diagnostics} />
        </motion.div>

        <div className="space-y-6">
          {/* Feature importance */}
          <motion.div variants={stagger.item} className="card p-6">
            <SectionHeader icon={BarChart2} title="Neural Network Weights" accent="#c084fc" />
            <FeatureImportanceChart importance={feature_importance} />
          </motion.div>

          {/* Trend insights */}
          {trend_insights?.length > 0 && (
            <motion.div variants={stagger.item} className="card p-6">
              <SectionHeader icon={TrendingUp} title="Time-Series Insights" accent="#fbbf24" />
              <div className="space-y-3">
                {trend_insights.map((t, i) => (
                  <TrendInsightRow key={i} t={t} i={i} />
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </div>

      {/* Comparison note */}
      {comparison_note && (
        <motion.div
          variants={stagger.item}
          className="rounded-2xl p-6 flex items-start gap-5 glass-panel border-primary/20 bg-primary/[0.03] relative overflow-hidden group"
        >
          <div className="absolute inset-y-0 left-0 w-1 bg-primary group-hover:w-1.5 transition-all" />
          <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center shrink-0">
            <Clock size={18} className="text-primary" />
          </div>
          <div className="space-y-1">
            <span className="text-[10px] font-black uppercase tracking-widest text-primary/70">Historical Context</span>
            <p className="text-sm leading-relaxed font-medium text-[var(--text-secondary)]">{comparison_note}</p>
          </div>
        </motion.div>
      )}

      {/* Visualizations */}
      {visuals.length > 0 && (
        <motion.div variants={stagger.item} className="card p-6">
          <SectionHeader icon={BarChart2} title="AI-Generated Visualizations" accent="var(--color-primary)" />
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {visuals.slice(0, 3).map((v, i) => (
              <motion.div
                key={i}
                className="rounded-2xl overflow-hidden glass-panel border-white/[0.05] group cursor-zoom-in"
                whileHover={{ scale: 1.02, borderColor: 'var(--color-primary-border)' }}
                transition={{ duration: 0.3 }}
              >
                <div
                  className="px-4 py-2 text-[9px] font-black uppercase tracking-widest border-b border-white/[0.05] bg-white/[0.02] text-[var(--text-muted)] group-hover:text-primary transition-colors"
                >
                  {v.title || v.metric || `Sensor Stream ${i + 1}`}
                </div>
                <div className="p-1">
                  <img
                    src={`data:image/png;base64,${v.image_base64}`}
                    alt={v.title || 'Visualization'}
                    className="w-full h-auto rounded-xl grayscale-[40%] group-hover:grayscale-0 transition-all duration-500"
                  />
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Engineering report */}
      <motion.div variants={stagger.item} className="card-elevated p-8 relative overflow-hidden group">
        <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
          <Brain size={120} />
        </div>
        <SectionHeader icon={Brain} title="AI Executive Summary" accent="#c084fc" />
        <div
          className="text-sm leading-relaxed whitespace-pre-wrap font-light first-letter:text-3xl first-letter:font-bold first-letter:text-primary first-letter:mr-1 first-letter:float-left"
          style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-sans)' }}
        >
          {engineering_report || 'No qualitative report generated.'}
        </div>
      </motion.div>

      {/* Expandable sections */}
      <motion.div variants={stagger.item} className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {rootCause.length > 0 && (
          <Accordion title="Neural Root Cause Analysis" icon={Brain}>
            <ul className="space-y-3 p-4">
              {rootCause.map((line, i) => (
                <li key={i} className="flex items-start gap-3 text-xs font-light leading-relaxed group" style={{ color: 'var(--text-secondary)' }}>
                  <div className="mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 bg-primary/40 group-hover:bg-primary transition-colors" />
                  {line}
                </li>
              ))}
            </ul>
          </Accordion>
        )}
        {historicalComparison.length > 0 && (
          <Accordion title="Peer Asset Comparison" icon={Clock}>
            <ul className="space-y-3 p-4">
              {historicalComparison.map((item, i) => (
                <li key={i} className="flex items-start gap-3 text-xs font-light leading-relaxed group" style={{ color: 'var(--text-secondary)' }}>
                  <div className="mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 bg-amber-500/40 group-hover:bg-amber-500 transition-colors" />
                  {item.detail || JSON.stringify(item)}
                </li>
              ))}
            </ul>
          </Accordion>
        )}
      </motion.div>
    </motion.div>
  )
}
