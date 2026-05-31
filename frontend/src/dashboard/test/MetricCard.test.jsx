import { describe, it, expect } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { MetricCard } from '../components/ui/MetricCard'
import { Activity } from 'lucide-react'
import * as fc from 'fast-check'

describe('MetricCard', () => {
  it('renders label and value', () => {
    render(<MetricCard label="Test Label" value="123.45" />)
    expect(screen.getByText('Test Label')).toBeInTheDocument()
    expect(screen.getByText('123.45')).toBeInTheDocument()
    cleanup()
  })

  it('renders sub text when provided', () => {
    render(<MetricCard label="Test" value="123" sub="subtext" />)
    expect(screen.getByText('subtext')).toBeInTheDocument()
    cleanup()
  })

  it('renders icon when provided', () => {
    const { container } = render(<MetricCard label="Test" value="123" icon={Activity} />)
    expect(container.querySelector('svg')).toBeInTheDocument()
    cleanup()
  })

  // Feature: dashboard-ui-redesign, Property 3: KPI card content completeness
  it('Property 3: KPI card content completeness', () => {
    fc.assert(
      fc.property(
        fc.string({ minLength: 1 }).map(s => s.trim()).filter(s => s.length > 0),
        fc.string({ minLength: 1 }).map(s => s.trim()).filter(s => s.length > 0),
        (label, value) => {
          render(<MetricCard label={label} value={value} icon={Activity} />)
          expect(screen.getByText(label)).toBeInTheDocument()
          expect(screen.getByText(value)).toBeInTheDocument()
          expect(document.querySelector('svg')).toBeInTheDocument()
          cleanup()
        }
      ),
      { numRuns: 50 }
    )
  })
})
