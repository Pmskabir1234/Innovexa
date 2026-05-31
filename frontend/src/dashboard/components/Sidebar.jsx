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
            className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm lg:hidden"
            onClick={onClose}
          />
        )}
      </AnimatePresence>

      <aside
        ref={sidebarRef}
        className={clsx(
          'sidebar flex flex-col',
          open && 'open'
        )}
      >
        <div className="lg:hidden flex items-center justify-between p-4 border-b border-[var(--border-subtle)]">
          <span className="font-bold text-sm text-[var(--text-primary)]">Control Panel</span>
          <button
            onClick={onClose}
            className="btn-ghost p-1.5"
            aria-label="Close sidebar"
          >
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 p-4 space-y-6 overflow-y-auto">
          <div className="space-y-4">
            <InputPanel
              params={params}
              machineId={machineId}
              onMachineIdChange={onMachineIdChange}
              onParamsChange={onParamsChange}
            />
          </div>

          <div className="h-px bg-gradient-to-r from-transparent via-[var(--border-subtle)] to-transparent" />

          <div className="space-y-2">
            <button
              onClick={onAnalyze}
              disabled={analyzeLoading}
              className="btn-primary w-full justify-center"
            >
              {analyzeLoading ? <Spinner size="sm" /> : <Activity size={14} />}
              Run Full Analysis
            </button>
            <button
              onClick={onPredict}
              disabled={predictLoading}
              className="btn-secondary w-full justify-center"
            >
              {predictLoading ? <Spinner size="sm" /> : <Zap size={14} />}
              Quick Predict
            </button>
            <button
              onClick={onHistory}
              disabled={historyLoading}
              className="btn-secondary w-full justify-center"
            >
              {historyLoading ? <Spinner size="sm" /> : <Clock size={14} />}
              Load History
            </button>
          </div>
        </div>

        <div className="p-4 mt-auto border-t border-[var(--border-subtle)] bg-[var(--surface)]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-[var(--color-primary-dim)] flex items-center justify-center text-[var(--color-primary)]">
              <Activity size={16} />
            </div>
            <div>
              <div className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Status</div>
              <div className="text-xs font-semibold text-[var(--text-secondary)]">Ready for Input</div>
            </div>
          </div>
        </div>
      </aside>
    </>
  )
}
