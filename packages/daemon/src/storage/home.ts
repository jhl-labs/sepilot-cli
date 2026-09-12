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

/**
 * Canonical daemon data root. Runtime bootstrap configures this from its
 * resolved `dataDir` option so embedded callers do not have to mutate the
 * process environment merely to keep every storage consumer on one profile.
 */
export function daemonDataDir(): string {
  return configuredStorage?.dataDir
    ?? envPath('SEPILOTD_DATA_DIR')
    ?? envPath('SEPILOTD_HOME')
    ?? join(homedir(), '.sepilotd')
}

export function sepilotdHome(): string {
  // SEPILOTD_HOME remains an explicit compatibility override. When it is not
  // supplied, bootstrap points this legacy helper at the canonical data dir
  // only after the legacy domain-state migration has committed.
  return envPath('SEPILOTD_HOME')
    ?? configuredStorage?.domainHome
    ?? daemonDataDir()
}

/**
 * Install process-scoped storage roots for a running daemon. The returned
 * release is generation-aware so a late shutdown from an older embedded
 * handle cannot clear a newer runtime's configuration.
 */
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
