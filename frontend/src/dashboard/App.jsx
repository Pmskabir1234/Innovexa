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
  return <div className={clsx('bg-white/[0.03] animate-pulse rounded-2xl border border-white/[0.05]', h)} />
}

function LoadingState({ message }) {
  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div className="flex items-center gap-4 px-1">
        <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        <span className="text-xs font-bold uppercase tracking-[0.2em] text-primary/80">{message}</span>
      </div>
      <div className="space-y-4">
        <SkeletonCard h="h-24" />
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => <SkeletonCard key={i} h="h-32" />)}
        </div>
        <SkeletonCard h="h-80" />
        <SkeletonCard h="h-64" />
      </div>
    </div>
  )
}

function EmptyState({ icon: Icon, title, description, action, actionLabel, loading, accent = 'var(--color-primary)' }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="card-elevated p-12 md:p-20 flex flex-col items-center gap-8 text-center max-w-2xl mx-auto relative overflow-hidden"
    >
      <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent pointer-events-none" />
      
      <motion.div
        className="relative"
        animate={{ y: [0, -10, 0] }}
        transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
      >
        <div
          className="w-20 h-20 rounded-3xl flex items-center justify-center shadow-2xl relative z-10"
          style={{
            background: `linear-gradient(135deg, ${accent}30, ${accent}10)`,
            border: `1px solid ${accent}40`,
            boxShadow: `0 0 40px ${accent}20`,
          }}
        >
          <Icon size={32} style={{ color: accent }} />
        </div>
        <div
          className="absolute inset-0 rounded-3xl animate-ping opacity-20"
          style={{ background: accent, animationDuration: '3s' }}
        />
      </motion.div>

      <div className="space-y-3 relative z-10">
        <h3 className="font-bold text-xl md:text-2xl tracking-tight text-[var(--text-primary)]">{title}</h3>
        <p className="text-sm md:text-base leading-relaxed text-[var(--text-muted)] max-w-sm mx-auto font-light">{description}</p>
      </div>

      <button onClick={action} disabled={loading} className="btn-primary relative z-10 min-w-[160px] justify-center">
        {loading ? <Spinner size="sm" /> : <ChevronRight size={14} className="group-hover:translate-x-1 transition-transform" />}
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
    <div className="min-h-screen bg-[var(--bg)] text-[var(--text-primary)] selection:bg-primary/20 selection:text-primary transition-colors duration-500">
      {/* Premium Atmosphere Background */}
      <div className="atmosphere" />
      
      {/* Interactive Global Glow */}
      <div
        className="fixed inset-0 pointer-events-none z-0 overflow-hidden"
        style={{
          background: 'radial-gradient(circle at 50% -10%, var(--color-primary-dim) 0%, transparent 50%)',
        }}
      />

      <Navbar
        theme={theme}
        onToggleTheme={toggleTheme}
        health={healthApi.data}
        healthLoading={healthApi.loading}
        onOpenSidebar={() => setSidebarOpen(true)}
      />

      <div className="layout-root relative z-10">
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
          <div className="flex flex-col gap-8 max-w-6xl mx-auto">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
              <TabBar
                tabs={TABS}
                activeTab={activeTab}
                hasData={hasData}
                onTabChange={setActiveTab}
              />
              <AnimatePresence>
                {anyLoading && (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    className="flex items-center gap-3 px-4 py-2 rounded-full glass-panel"
                  >
                    <div className="w-2 h-2 rounded-full bg-primary animate-pulse shadow-[0_0_8px_var(--color-primary)]" />
                    <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-primary">AI Engine Active</span>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 20, filter: 'blur(10px)' }}
                animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
                exit={{ opacity: 0, y: -20, filter: 'blur(10px)' }}
                transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              >
                {activeTab === 'analyze' && (
                  analyzeApi.loading ? (
                    <LoadingState message="Running full analysis pipeline…" />
                  ) : analyzeApi.data ? (
                    <AnalysisResult data={analyzeApi.data} />
                  ) : (
                    <EmptyState
                      icon={Activity}
                      title="No Analysis Data"
                      description="Configure your machine parameters in the sidebar and trigger the AI analysis engine."
                      action={handleAnalyze}
                      actionLabel="Initiate Analysis"
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
                      title="Predict Failure Risk"
                      description="Use our predictive models to estimate the probability of system failure."
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
                    <LoadingState message="Loading historical data…" />
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


