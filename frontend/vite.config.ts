import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Load-balanced backend targets — Vite dev proxy round-robins across these.
const BACKEND_TARGETS = [
  'http://127.0.0.1:8000',
]

let rrIndex = 0
function nextTarget() {
  const t = BACKEND_TARGETS[rrIndex % BACKEND_TARGETS.length]
  rrIndex++
  return t
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    proxy: {
      '/api': {
        target: BACKEND_TARGETS[0],
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            const target = nextTarget()
            const url = new URL(target)
            proxyReq.host = url.host
          })
        },
      },
    },
  },
})
