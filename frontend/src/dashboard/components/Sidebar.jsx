import { useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Activity, Zap, Clock } from 'lucide-react'
import { InputPanel } from './InputPanel'
import { Spinner } from './ui/Spinner'
import clsx from 'clsx'

export function Sidebar({
  open,
  onClose,
  params,
  machineId,
  onMachineIdChange,
  onParamsChange,
  onAnalyze,
  onPredict,
  onHistory,
  analyzeLoading,
  predictLoading,
  historyLoading
}) {
  const sidebarRef = useRef(null)

  // Escape key handler
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') onClose()
    }
    if (open) {
      window.addEventListener('keydown', handleEscape)
    }
    return () => window.removeEventListener('keydown', handleEscape)
  }, [open, onClose])

  // Simple focus trap for mobile
  useEffect(() => {
    if (open && sidebarRef.current && window.innerWidth < 1024) {
      const focusable = sidebarRef.current.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      )
      if (focusable.length === 0) return

      const first = focusable[0]
      const last = focusable[focusable.length - 1]

      const handleTab = (e) => {
        if (e.key !== 'Tab') return
        if (e.shiftKey) {
          if (document.activeElement === first) {
            last.focus()
            e.preventDefault()
          }
        } else {
          if (document.activeElement === last) {
            first.focus()
            e.preventDefault()
          }
        }
      }

      window.addEventListener('keydown', handleTab)
      return () => window.removeEventListener('keydown', handleTab)
    }
  }, [open])

  return (
    <>
      <AnimatePresence>
        {open && (
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[140] bg-black/80 backdrop-blur-md lg:hidden"
            onClick={onClose}
          />
        )}
      </AnimatePresence>

      <aside
        ref={sidebarRef}
        className={clsx(
          'sidebar flex flex-col z-[150]',
          open && 'open'
        )}
      >
        <div className="lg:hidden flex items-center justify-between px-6 py-5 border-b border-white/[0.05]">
          <span className="font-bold text-xs uppercase tracking-[0.2em] text-[var(--text-primary)]">Control Terminal</span>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/[0.05] text-[var(--text-muted)]"
            aria-label="Close sidebar"
          >
            <X size={20} />
          </button>
        </div>

        <div className="flex-1 p-6 space-y-8 overflow-y-auto scrollbar-hide">
          <div className="space-y-6">
            <InputPanel
              params={params}
              machineId={machineId}
              onMachineIdChange={onMachineIdChange}
              onParamsChange={onParamsChange}
            />
          </div>

          <div className="divider opacity-50" />

          <div className="space-y-3">
            <button
              onClick={onAnalyze}
              disabled={analyzeLoading}
              className="btn-primary w-full justify-center group relative overflow-hidden"
            >
              <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 pointer-events-none" />
              {analyzeLoading ? <Spinner size="sm" /> : <Activity size={14} className="relative z-10" />}
              <span className="relative z-10">Run Full Analysis</span>
            </button>
            
            <button
              onClick={onPredict}
              disabled={predictLoading}
              className="btn-secondary w-full justify-center hover:scale-[1.02] active:scale-[0.98]"
            >
              {predictLoading ? <Spinner size="sm" /> : <Zap size={14} className="text-amber-400" />}
              Quick Predict
            </button>
            
            <button
              onClick={onHistory}
              disabled={historyLoading}
              className="btn-secondary w-full justify-center hover:scale-[1.02] active:scale-[0.98]"
            >
              {historyLoading ? <Spinner size="sm" /> : <Clock size={14} className="text-emerald-400" />}
              Load History
            </button>
          </div>
        </div>

        <div className="p-6 mt-auto border-t border-white/[0.05] bg-white/[0.01]">
          <div className="flex items-center gap-4 p-4 rounded-2xl bg-white/[0.03] border border-white/[0.05]">
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary shadow-[0_0_15px_rgba(34,197,94,0.1)]">
              <Activity size={20} />
            </div>
            <div>
              <div className="text-[9px] font-black uppercase tracking-[0.2em] text-[var(--text-muted)] mb-0.5">Engine Status</div>
              <div className="text-[11px] font-bold text-primary tracking-wide">Ready for Input</div>
            </div>
          </div>
        </div>
      </aside>
    </>
  )
}
