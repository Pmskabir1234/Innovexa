import { describe, it, expect, beforeEach, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useTheme } from '../hooks/useTheme'
import * as fc from 'fast-check'

class LocalStorageMock {
  constructor() {
    this.store = {}
  }
  clear() {
    this.store = {}
  }
  getItem(key) {
    return this.store[key] || null
  }
  setItem(key, value) {
    this.store[key] = String(value)
  }
  removeItem(key) {
    delete this.store[key]
  }
}

const localStorageMock = new LocalStorageMock()
vi.stubGlobal('localStorage', localStorageMock)

describe('useTheme', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.clearAllMocks()
    
    // Mock matchMedia
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: vi.fn().mockImplementation(query => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    })
  })

  it('initializes to light by default if no storage and no dark preference', () => {
    const { result } = renderHook(() => useTheme())
    expect(result.current.theme).toBe('light')
  })

  it('initializes to dark if system preference is dark', () => {
    window.matchMedia.mockImplementation(query => ({
      matches: query === '(prefers-color-scheme: dark)',
      media: query,
    }))
    const { result } = renderHook(() => useTheme())
    expect(result.current.theme).toBe('dark')
  })

  it('initializes to stored value regardless of system preference', () => {
    localStorage.setItem('ci-theme', 'dark')
    window.matchMedia.mockImplementation(query => ({
      matches: false, // system is light
      media: query,
    }))
    const { result } = renderHook(() => useTheme())
    expect(result.current.theme).toBe('dark')
  })

  it('toggles theme correctly', () => {
    const { result } = renderHook(() => useTheme())
    expect(result.current.theme).toBe('light')
    act(() => {
      result.current.toggle()
    })
    expect(result.current.theme).toBe('dark')
    act(() => {
      result.current.toggle()
    })
    expect(result.current.theme).toBe('light')
  })

  // Feature: dashboard-ui-redesign, Property 1: theme persistence round-trip
  it('Property 1: theme persistence round-trip', () => {
    fc.assert(
      fc.property(fc.constantFrom('dark', 'light'), (targetTheme) => {
        localStorage.clear()
        const { result } = renderHook(() => useTheme())
        
        act(() => {
          if (result.current.theme !== targetTheme) {
            result.current.toggle()
          }
        })
        
        expect(result.current.theme).toBe(targetTheme)
        expect(localStorage.getItem('ci-theme')).toBe(targetTheme)
      }),
      { numRuns: 100 }
    )
  })

  // Feature: dashboard-ui-redesign, Property 2: theme initialization priority
  it('Property 2: theme initialization priority', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('dark', 'light'), // stored
        fc.boolean(), // system preference is dark
        (stored, systemIsDark) => {
          localStorage.setItem('ci-theme', stored)
          window.matchMedia.mockImplementation(query => ({
            matches: query === '(prefers-color-scheme: dark)' ? systemIsDark : !systemIsDark,
            media: query,
          }))
          
          const { result } = renderHook(() => useTheme())
          expect(result.current.theme).toBe(stored)
        }
      ),
      { numRuns: 100 }
    )
  })
})
