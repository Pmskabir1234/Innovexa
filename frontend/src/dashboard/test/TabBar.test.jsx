import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { TabBar } from '../components/TabBar'
import { Activity, Zap } from 'lucide-react'

const TABS = [
  { id: 't1', label: 'Tab 1', icon: Activity, accent: '#ff0000' },
  { id: 't2', label: 'Tab 2', icon: Zap,      accent: '#00ff00' },
]

describe('TabBar', () => {
  it('renders all tabs', () => {
    render(<TabBar tabs={TABS} activeTab="t1" hasData={{}} onTabChange={vi.fn()} />)
    expect(screen.getByText('Tab 1')).toBeInTheDocument()
    expect(screen.getByText('Tab 2')).toBeInTheDocument()
  })

  it('calls onTabChange when a tab is clicked', () => {
    const onTabChange = vi.fn()
    render(<TabBar tabs={TABS} activeTab="t1" hasData={{}} onTabChange={onTabChange} />)
    
    fireEvent.click(screen.getByText('Tab 2'))
    expect(onTabChange).toHaveBeenCalledWith('t2')
  })

  it('shows dot indicator when data is loaded for inactive tab', () => {
    const { container } = render(
      <TabBar tabs={TABS} activeTab="t1" hasData={{ t2: true }} onTabChange={vi.fn()} />
    )
    // The dot is a span with specific style
    const dot = container.querySelector('span[style*="background: rgb(0, 255, 0)"]')
    expect(dot).toBeInTheDocument()
  })

  it('hides dot indicator for active tab even if data is loaded', () => {
    const { container } = render(
      <TabBar tabs={TABS} activeTab="t1" hasData={{ t1: true }} onTabChange={vi.fn()} />
    )
    const dot = container.querySelector('span[style*="background: rgb(255, 0, 0)"]')
    expect(dot).not.toBeInTheDocument()
  })
})
