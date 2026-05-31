import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'
import { Navbar } from '../components/Navbar'
import { Sidebar } from '../components/Sidebar'
import * as fc from 'fast-check'

describe('Accessibility', () => {
  // Feature: dashboard-ui-redesign, Property 4: icon-only buttons have aria-labels
  it('Property 4: icon-only buttons have aria-labels in Navbar and Sidebar', () => {
    // For Navbar
    const { container: navContainer } = render(
      <Navbar theme="dark" onToggleTheme={vi.fn()} health={{}} healthLoading={false} onOpenSidebar={vi.fn()} />
    )
    const navButtons = navContainer.querySelectorAll('button')
    navButtons.forEach(btn => {
      const hasText = btn.textContent.trim().length > 0
      const hasIcon = !!btn.querySelector('svg')
      if (hasIcon && !hasText) {
        expect(btn).toHaveAttribute('aria-label')
        expect(btn.getAttribute('aria-label')).not.toBe('')
      }
    })

    // For Sidebar
    const { container: sideContainer } = render(
      <Sidebar 
        open={true} 
        onClose={vi.fn()} 
        params={{}} 
        machineId="M1" 
        onMachineIdChange={vi.fn()} 
        onParamsChange={vi.fn()}
        onAnalyze={vi.fn()}
        onPredict={vi.fn()}
        onHistory={vi.fn()}
      />
    )
    const sideButtons = sideContainer.querySelectorAll('button')
    sideButtons.forEach(btn => {
      const hasText = btn.textContent.trim().length > 0
      const hasIcon = !!btn.querySelector('svg')
      if (hasIcon && !hasText) {
        expect(btn).toHaveAttribute('aria-label')
        expect(btn.getAttribute('aria-label')).not.toBe('')
      }
    })
  })
})
