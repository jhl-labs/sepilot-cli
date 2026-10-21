import { realpathSync } from 'node:fs'

export interface StandaloneDaemonHooks {
  selfExec: boolean
  selfExecCommand?: string
  selfExecPrefixArgs?: string[]
  selfBinaryPath?: string
  foregroundMain: (argv: string[]) => Promise<void>
  embeddedFactory?: (options: {
    dataDir: string
    host: string
    port: number
    autoApproveCliFlag?: boolean
  }) => Promise<unknown>
}

let hooks: StandaloneDaemonHooks | undefined

export function registerStandaloneDaemon(value: StandaloneDaemonHooks): void {
  hooks = value
}

export function getStandaloneDaemon(): StandaloneDaemonHooks | undefined {
  return hooks
}

export function __resetStandaloneDaemonForTests(): void {
  hooks = undefined
}

export function resolveStandaloneDaemonInvocation():
  | { command: string; args: string[] }
  | undefined {
  const standalone = getStandaloneDaemon()
  if (!standalone?.selfExec) return undefined
  const command = standalone.selfExecCommand ?? process.execPath
  const prefixArgs =
    standalone.selfExecPrefixArgs ??
    (() => {
      const entryScript = process.argv[1]
      const isInterpreterExec =
        !!entryScript &&
        (() => {
          try {
            return realpathSync(entryScript) !== realpathSync(process.execPath)
          } catch {
            return entryScript !== process.execPath
          }
        })()
      return isInterpreterExec ? [entryScript] : []
    })()
  return { command, args: [...prefixArgs, '__daemon'] }
}
