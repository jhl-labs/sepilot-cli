export type CliLocale = 'en' | 'ko'

const FORCE_ENV_KEYS = ['SEPILOT_LOCALE', 'SEPILOT_LANG'] as const
const SYSTEM_ENV_KEYS = ['LC_ALL', 'LC_MESSAGES', 'LANG'] as const

export function detectCliLocale(env: NodeJS.ProcessEnv = process.env): CliLocale {
  for (const key of FORCE_ENV_KEYS) {
    const value = env[key]?.toLowerCase()
    if (value === 'en' || value === 'ko') return value
  }
  for (const key of SYSTEM_ENV_KEYS) {
    const value = env[key]?.toLowerCase()
    if (value && value.startsWith('ko')) return 'ko'
  }
  return 'en'
}
