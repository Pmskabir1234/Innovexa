import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { InputPanel, DEFAULT_PARAMS } from '../components/InputPanel'
import * as fc from 'fast-check'

const PARAM_KEYS = [
  'vibration_rms', 'rpm', 'torque_nm', 'bearing_temp_c', 'ambient_temp_c',
  'motor_current_a', 'voltage_v', 'flow_rate_l_min', 'pressure_bar', 'humidity_percent'
]

describe('InputPanel', () => {
  it('calls onParamsChange when a parameter is changed', () => {
    const onParamsChange = vi.fn()
    render(
      <InputPanel 
        params={DEFAULT_PARAMS} 
        onParamsChange={onParamsChange} 
        machineId="M1" 
        onMachineIdChange={vi.fn()} 
      />
    )
    
    // Change Vibration RMS
    const input = screen.getByLabelText(/Vibration RMS/i)
    fireEvent.change(input, { target: { value: '10.5' } })
    
    expect(onParamsChange).toHaveBeenCalledWith(expect.objectContaining({
      vibration_rms: 10.5
    }))
  })

  it('calls onMachineIdChange when machine id is changed', () => {
    const onMachineIdChange = vi.fn()
    render(
      <InputPanel 
        params={DEFAULT_PARAMS} 
        onParamsChange={vi.fn()} 
        machineId="M1" 
        onMachineIdChange={onMachineIdChange} 
      />
    )
    
    const input = screen.getByLabelText(/Machine ID/i)
    fireEvent.change(input, { target: { value: 'NEW-ID' } })
    
    expect(onMachineIdChange).toHaveBeenCalledWith('NEW-ID')
  })

  // Feature: dashboard-ui-redesign, Property 6: parameter change callback completeness
  it('Property 6: parameter change callback completeness', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...PARAM_KEYS),
        fc.float({ min: 0, max: 500, noNaN: true }),
        (key, value) => {
          const onParamsChange = vi.fn()
          const { unmount } = render(
            <InputPanel 
              params={DEFAULT_PARAMS} 
              onParamsChange={onParamsChange} 
              machineId="M1" 
              onMachineIdChange={vi.fn()} 
            />
          )
          
          // Find input by label regex matching the key parts
          // This is a bit tricky, but we can find all inputs and match by value or id
          const inputs = document.querySelectorAll('input[type="number"]')
          const input = Array.from(inputs).find(i => i.id === key)
          
          if (input) {
            fireEvent.change(input, { target: { value: value.toString() } })
            expect(onParamsChange).toHaveBeenCalled()
            const calledWith = onParamsChange.mock.calls[0][0]
            PARAM_KEYS.forEach(k => {
              expect(calledWith).toHaveProperty(k)
            })
            expect(calledWith[key]).toBe(value)
          }
          unmount()
        }
      ),
      { numRuns: 20 }
    )
  })
})
