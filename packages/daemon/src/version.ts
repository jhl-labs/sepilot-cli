import { readFileSync } from 'node:fs'

export function loadDaemonVersion(
  packageJsonUrl: URL = new URL('../package.json', import.meta.url),
): string {
  const fromEnv = process.env.SEPILOT_VERSION?.trim()
  if (fromEnv) return fromEnv

  try {
    const raw = readFileSync(packageJsonUrl, 'utf8')
    const parsed = JSON.parse(raw) as { version?: string }
    return parsed.version?.trim() || '0.0.0'
  } catch {
    return '0.0.0'
  }
}

export const DAEMON_VERSION = loadDaemonVersion()
