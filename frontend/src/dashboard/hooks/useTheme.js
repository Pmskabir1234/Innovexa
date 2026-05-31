import { useState, useEffect } from 'react'

function getStoredTheme() {
  try {
    return localStorage.getItem('ci-theme')
  } catch {
    return null
  }
}

export function useTheme() {
  const [theme, setTheme] = useState(() => {
    const stored = getStoredTheme()
    if (stored) return stored
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })

  useEffect(() => {
    const root = document.documentElement
    root.dataset.theme = theme
    try {
      localStorage.setItem('ci-theme', theme)
    } catch {
      // storage unavailable (e.g. private browsing with strict settings)
    }
  }, [theme])

  const toggle = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
  return { theme, toggle }
}
