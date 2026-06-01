import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Cpu, Thermometer, Zap, Gauge, ChevronDown, RotateCcw } from 'lucide-react'

const DEFAULT_PARAMS = {
  vibration_rms: 4.8,
  rpm: 2900,
  torque_nm: 175,
  bearing_temp_c: 78,
  ambient_temp_c: 32,
  motor_current_a: 58,
  voltage_v: 415,
  flow_rate_l_min: 470,
  pressure_bar: 6.2,
  humidity_percent: 54,
}

const PARAM_GROUPS = [
  {
    label: 'Mechanical',
    icon: Cpu,
    accent: '#22d3ee',
    fields: [
      { key: 'vibration_rms',  label: 'Vibration RMS', unit: 'mm/s', min: 0,    max: 50,    step: 0.1 },
      { key: 'rpm',            label: 'RPM',            unit: 'rpm',  min: 100,  max: 10000, step: 10  },
      { key: 'torque_nm',      label: 'Torque',         unit: 'Nm',   min: 0,    max: 5000,  step: 1   },
    ],
  },
  {
    label: 'Thermal',
    icon: Thermometer,
    accent: '#fb923c',
    fields: [
      { key: 'bearing_temp_c', label: 'Bearing Temp', unit: '°C', min: -20, max: 220, step: 0.5 },
      { key: 'ambient_temp_c', label: 'Ambient Temp', unit: '°C', min: -30, max: 80,  step: 0.5 },
    ],
  },
  {
    label: 'Electrical',
    icon: Zap,
    accent: '#fbbf24',
    fields: [
      { key: 'motor_current_a', label: 'Motor Current', unit: 'A', min: 0,   max: 500,  step: 0.5 },
      { key: 'voltage_v',       label: 'Voltage',       unit: 'V', min: 100, max: 1000, step: 1   },
    ],
  },
  {
    label: 'Process',
    icon: Gauge,
    accent: '#c084fc',
    fields: [
      { key: 'flow_rate_l_min',  label: 'Flow Rate', unit: 'L/min', min: 0, max: 3000, step: 1   },
      { key: 'pressure_bar',     label: 'Pressure',  unit: 'bar',   min: 0, max: 100,  step: 0.1 },
      { key: 'humidity_percent', label: 'Humidity',  unit: '%',     min: 0, max: 100,  step: 0.5 },
    ],
  },
]

function GroupSection({ group, params, onChange }) {
  const [open, setOpen] = useState(true)
  const Icon = group.icon

  return (
    <div
      className="rounded-2xl overflow-hidden transition-all duration-300"
      style={{ 
        border: open ? '1px solid var(--border-accent)' : '1px solid var(--border-subtle)', 
        background: open ? 'rgba(255,255,255,0.02)' : 'transparent' 
      }}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-white/[0.04] group"
      >
        <div
          className="w-8 h-8 rounded-xl flex items-center justify-center shrink-0 transition-transform group-hover:scale-110"
          style={{ 
            background: open ? `${group.accent}20` : 'rgba(255,255,255,0.05)',
            boxShadow: open ? `0 0 15px ${group.accent}15` : 'none'
          }}
        >
          <Icon size={14} style={{ color: open ? group.accent : 'var(--text-muted)' }} />
        </div>
        <span
          className="text-[10px] font-black tracking-[0.15em] uppercase flex-1"
          style={{ color: open ? 'var(--text-primary)' : 'var(--text-muted)' }}
        >
          {group.label}
        </span>
        <motion.div animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}>
          <ChevronDown size={14} style={{ color: 'var(--text-faint)' }} />
        </motion.div>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="fields"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            style={{ overflow: 'hidden' }}
          >
            <div
              className="px-4 pb-5 pt-2 space-y-4"
              style={{ borderTop: '1px solid var(--border-subtle)' }}
            >
              {group.fields.map((f) => (
                <div key={f.key} className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <label
                      className="label !mb-0"
                      htmlFor={f.key}
                    >
                      {f.label}
                    </label>
                    <span className="text-[10px] font-bold text-primary/60 font-mono">
                      {params[f.key]} {f.unit}
                    </span>
                  </div>
                  <input
                    id={f.key}
                    type="number"
                    className="input-field !py-2 !px-3 font-mono text-xs focus:ring-1 focus:ring-primary/30"
                    value={params[f.key]}
                    min={f.min}
                    max={f.max}
                    step={f.step}
                    onChange={(e) => onChange(f.key, parseFloat(e.target.value) || 0)}
                  />
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export function InputPanel({ params, onParamsChange, machineId, onMachineIdChange }) {
  function handleChange(key, value) {
    onParamsChange({ ...params, [key]: value })
  }

  function handleReset() {
    onParamsChange(DEFAULT_PARAMS)
  }

  return (
    <div className="space-y-5">
      {/* Machine ID */}
      <div className="space-y-2">
        <label className="label" htmlFor="machine-id">System Identifier</label>
        <div className="relative group">
          <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none text-primary/40 group-focus-within:text-primary transition-colors">
            <Cpu size={14} />
          </div>
          <input
            id="machine-id"
            type="text"
            className="input-field font-mono !pl-10 !py-2.5 tracking-wider uppercase"
            value={machineId}
            onChange={(e) => onMachineIdChange(e.target.value)}
            placeholder="MOTOR-LINE-07"
          />
        </div>
      </div>

      <div className="divider opacity-30" />

      {/* Parameter groups */}
      <div className="space-y-3">
        {PARAM_GROUPS.map((g) => (
          <GroupSection key={g.label} group={g} params={params} onChange={handleChange} />
        ))}
      </div>

      <button
        onClick={handleReset}
        className="btn-ghost w-full justify-center gap-2 text-[10px] font-bold uppercase tracking-[0.1em] hover:bg-white/[0.03]"
        style={{ color: 'var(--text-faint)' }}
      >
        <RotateCcw size={12} />
        Reset to Baseline
      </button>
    </div>
  )
}

export { DEFAULT_PARAMS }
