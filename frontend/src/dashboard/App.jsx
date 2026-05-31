import { useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useTheme } from './hooks/useTheme'
import { useApi } from './hooks/useApi'
import { api } from './api/client'
import { Navbar } from './components/Navbar'
import { Sidebar } from './components/Sidebar'
import { TabBar } from './components/TabBar'
import { DEFAULT_PARAMS } from './components/InputPanel'
import { AnalysisResult } from './components/AnalysisResult'
import { PredictResult } from './components/PredictResult'
import { SimulationPanel } from './components/SimulationPanel'
import { HistoryPanel } from './components/HistoryPanel'
import { Toast } from './components/ui/Toast'
import { Spinner } from './components/ui/Spinner'
import { Activity, Zap, FlaskConical, Database, ChevronRight } from 'lucide-react'
import clsx from 'clsx'

const TABS = [
  { id: 'analyze',  label: 'Analysis',   icon: Activity,     accent: 'hsl(119,99%,46%)' },
  { id: 'predict',  label: 'Predict',    icon: Zap,          accent: '#fbbf24' },
  { id: 'simulate', label: 'Simulate',   icon: FlaskConical, accent: '#c084fc' },
  { id: 'history',  label: 'History',    icon: Database,     accent: '#34d399' },
]

function SkeletonCard({ h = 'h-32' }) {
  return <div className={clsx('skeleton rounded-2xl', h)} />
}

function LoadingState({ message }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3 px-1">
        <Spinner size="sm" />
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{message}</span>
      </div>
      <SkeletonCard h="h-16" />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[...Array(4)].map((_, i) => <SkeletonCard key={i} h="h-24" />)}
      </div>
      <SkeletonCard h="h-64" />
      <SkeletonCard h="h-48" />
    </div>
  )
}

function EmptyState({ icon: Icon, title, description, action, actionLabel, loading, accent = 'var(--color-primary)' }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className="card-elevated p-16 flex flex-col items-center gap-5 text-center"
    >
      <motion.div
        className="relative"
        animate={{ y: [0, -6, 0] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
      >
        <div
          className="w-16 h-16 rounded-2xl flex items-center justify-center"
          style={{
            background: `color-mix(in srgb, ${accent}, transparent 90%)`,
            border: `1px solid color-mix(in srgb, ${accent}, transparent 80%)`,
            boxShadow: `0 0 30px color-mix(in srgb, ${accent}, transparent 85%)`,
          }}
        >
          <Icon size={26} style={{ color: accent }} />
        </div>
        <div
          className="absolute inset-0 rounded-2xl animate-ping"
          style={{ background: `color-mix(in srgb, ${accent}, transparent 92%)`, animationDuration: '3s' }}
        />
      </motion.div>

      <div>
        <h3 className="font-semibold text-sm mb-1.5 text-[var(--text-secondary)]">{title}</h3>
        <p className="text-xs leading-relaxed max-w-xs text-[var(--text-muted)]">{description}</p>
      </div>

      <button onClick={action} disabled={loading} className="btn-primary">
        {loading ? <Spinner size="sm" /> : <ChevronRight size={13} />}
        {actionLabel}
      </button>
    </motion.div>
  )
}

export default function App() {
  const { theme, toggle: toggleTheme } = useTheme()
  const [activeTab, setActiveTab] = useState('analyze')
  const [machineId, setMachineId] = useState('MOTOR-LINE-07')
  const [params, setParams] = useState(DEFAULT_PARAMS)
  const [toast, setToast] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const healthApi   = useApi(useCallback((sig) => api.health(sig), []))
  const analyzeApi  = useApi(useCallback((payload, sig) => api.analyze(payload, sig), []))
  const predictApi  = useApi(useCallback((payload, sig) => api.predict(payload, sig), []))
  const simulateApi = useApi(useCallback((payload, sig) => api.simulate(payload, sig), []))
  const historyApi  = useApi(useCallback((id, limit, sig) => api.history(id, limit, sig), []))

  useEffect(() => {
    healthApi.execute()
    const interval = setInterval(() => healthApi.execute(), 30000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const err = analyzeApi.error || predictApi.error || simulateApi.error || historyApi.error
    if (err) setToast({ message: err, type: 'error' })
  }, [analyzeApi.error, predictApi.error, simulateApi.error, historyApi.error])

  function buildPayload() {
    return { machine_id: machineId, parameters: params }
  }

  async function handleAnalyze() {
    await analyzeApi.execute(buildPayload())
    setActiveTab('analyze')
    setSidebarOpen(false)
  }

  async function handlePredict() {
    await predictApi.execute(buildPayload())
    setActiveTab('predict')
    setSidebarOpen(false)
  }

  async function handleHistory() {
    await historyApi.execute(machineId, 10)
    setActiveTab('history')
    setSidebarOpen(false)
  }

  async function handleSimulate(payload) {
    await simulateApi.execute(payload)
  }

  const anyLoading = analyzeApi.loading || predictApi.loading || simulateApi.loading || historyApi.loading

  const hasData = {
    analyze:  !!analyzeApi.data,
    predict:  !!predictApi.data,
    simulate: !!simulateApi.data,
    history:  !!historyApi.data,
  }

  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--text-primary)]">
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse 80% 50% at 50% -20%, var(--color-primary-dim) 0%, transparent 60%)',
        }}
      />

      <Navbar
        theme={theme}
        onToggleTheme={toggleTheme}
        health={healthApi.data}
        healthLoading={healthApi.loading}
        onOpenSidebar={() => setSidebarOpen(true)}
      />

      <div className="layout-root">
        <Sidebar
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          params={params}
          machineId={machineId}
          onMachineIdChange={setMachineId}
          onParamsChange={setParams}
          onAnalyze={handleAnalyze}
          onPredict={handlePredict}
          onHistory={handleHistory}
          analyzeLoading={analyzeApi.loading}
          predictLoading={predictApi.loading}
          historyLoading={historyApi.loading}
        />

        <main className="content-area">
          <div className="flex flex-col gap-6">
            <div className="flex items-center justify-between gap-4">
              <TabBar
                tabs={TABS}
                activeTab={activeTab}
                hasData={hasData}
                onTabChange={setActiveTab}
              />
              {anyLoading && (
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[var(--surface)] border border-[var(--border-subtle)] animate-pulse">
                  <Spinner size="sm" />
                  <span className="text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">Processing</span>
                </div>
              )}
            </div>

            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
              >
                {activeTab === 'analyze' && (
                  analyzeApi.loading ? (
                    <LoadingState message="Running full analysis pipeline…" />
                  ) : analyzeApi.data ? (
                    <AnalysisResult data={analyzeApi.data} />
                  ) : (
                    <EmptyState
                      icon={Activity}
                      title="No analysis yet"
                      description="Configure your machine parameters in the sidebar and click Run Full Analysis."
                      action={handleAnalyze}
                      actionLabel="Run Analysis"
                      loading={analyzeApi.loading}
                      accent="hsl(119,99%,46%)"
                    />
                  )
                )}

                {activeTab === 'predict' && (
                  predictApi.loading ? (
                    <LoadingState message="Running failure prediction…" />
                  ) : predictApi.data ? (
                    <PredictResult data={predictApi.data} />
                  ) : (
                    <EmptyState
                      icon={Zap}
                      title="No prediction yet"
                      description="Click Quick Predict for a fast failure probability estimate."
                      action={handlePredict}
                      actionLabel="Quick Predict"
                      loading={predictApi.loading}
                      accent="#fbbf24"
                    />
                  )
                )}

                {activeTab === 'simulate' && (
                  <SimulationPanel
                    params={params}
                    machineId={machineId}
                    onSimulate={handleSimulate}
                    loading={simulateApi.loading}
                    result={simulateApi.data}
                  />
                )}

                {activeTab === 'history' && (
                  historyApi.loading ? (
                    <LoadingState message="Loading history…" />
                  ) : (
                    <HistoryPanel items={historyApi.data?.items} />
                  )
                )}
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  )
}

