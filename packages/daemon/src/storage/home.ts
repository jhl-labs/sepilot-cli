import { homedir } from 'node:os'
import { join, resolve } from 'node:path'

interface DaemonStorageConfiguration {
  dataDir: string
  domainHome: string
  generation: number
}

let configuredStorage: DaemonStorageConfiguration | undefined
let storageGeneration = 0

function envPath(name: 'SEPILOTD_DATA_DIR' | 'SEPILOTD_HOME'): string | undefined {
  const value = process.env[name]?.trim()
  return value ? resolve(value) : undefined
}

export function daemonDataDir(): string {
  return configuredStorage?.dataDir
    ?? envPath('SEPILOTD_DATA_DIR')
    ?? envPath('SEPILOTD_HOME')
    ?? join(homedir(), '.sepilotd')
}

export function sepilotdHome(): string {
  return envPath('SEPILOTD_HOME')
    ?? configuredStorage?.domainHome
    ?? daemonDataDir()
}

export function configureDaemonStorage(input: {
  dataDir: string
  domainHome: string
}): () => void {
  const generation = ++storageGeneration
  configuredStorage = {
    dataDir: resolve(input.dataDir),
    domainHome: resolve(input.domainHome),
    generation,
  }

  return () => {
    if (configuredStorage?.generation === generation) {
      configuredStorage = undefined
    }
  }
}
