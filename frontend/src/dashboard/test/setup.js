import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Do not attempt to overwrite global.localStorage if it's read-only.
// If it's missing clear, we might need to investigate why jsdom is incomplete.
