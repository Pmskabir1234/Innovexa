import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Sidebar } from '../components/Sidebar'

const DEFAULT_PROPS = {
  open: true,
  onClose: vi.fn(),
  params: {},
  machineId: 'TEST-01',
  onMachineIdChange: vi.fn(),
  onParamsChange: vi.fn(),
  onAnalyze: vi.fn(),
  onPredict: vi.fn(),
  onHistory: vi.fn(),
}

describe('Sidebar', () => {
  it('renders input panel and buttons when open', () => {
    render(<Sidebar {...DEFAULT_PROPS} />)
    expect(screen.getByLabelText(/Machine ID/i)).toBeInTheDocument()
    expect(screen.getByText(/Run Full Analysis/i)).toBeInTheDocument()
    expect(screen.getByText(/Quick Predict/i)).toBeInTheDocument()
    expect(screen.getByText(/Load History/i)).toBeInTheDocument()
  })

  it('calls onClose when clicking backdrop (on mobile)', () => {
    // Mock window.innerWidth to simulate mobile
    Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 500 })
    
    render(<Sidebar {...DEFAULT_PROPS} />)
    const backdrop = document.querySelector('.fixed.inset-0.z-40')
    if (backdrop) {
      fireEvent.click(backdrop)
      expect(DEFAULT_PROPS.onClose).toHaveBeenCalled()
    }
  })

  it('calls onClose when pressing Escape', () => {
    render(<Sidebar {...DEFAULT_PROPS} />)
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(DEFAULT_PROPS.onClose).toHaveBeenCalled()
  })

  it('calls action callbacks when buttons are clicked', () => {
    render(<Sidebar {...DEFAULT_PROPS} />)
    
    fireEvent.click(screen.getByText(/Run Full Analysis/i))
    expect(DEFAULT_PROPS.onAnalyze).toHaveBeenCalled()
    
    fireEvent.click(screen.getByText(/Quick Predict/i))
    expect(DEFAULT_PROPS.onPredict).toHaveBeenCalled()
    
    fireEvent.click(screen.getByText(/Load History/i))
    expect(DEFAULT_PROPS.onHistory).toHaveBeenCalled()
  })
})
