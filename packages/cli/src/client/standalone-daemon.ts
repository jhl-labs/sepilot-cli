import { realpathSync } from 'node:fs'

/** Hooks a single-file bundle installs so the CLI can act as / launch the daemon. */
export interface StandaloneDaemonHooks {
  /** When true, ensure-daemon may re-launch the current executable with `__daemon`. */
  selfExec: boolean
  /**
   * Exact executable to spawn when re-launching ourselves as the daemon. The
   * bundle entry knows whether it is a `bun --compile` single-file binary (in
   * which case this is the binary itself, `process.execPath`) or a plain
   * `dist/main.js` running under a Node/Bun interpreter (in which case this is
   * the interpreter). When omitted, ensure-daemon falls back to a heuristic.
   */
  selfExecCommand?: string
  /**
   * Argv to put *before* `__daemon` when re-launching. For a compiled binary
   * this is `[]` (the binary re-enters itself); for `node dist/main.js` it is
   * `[<path to main.js>]` so the interpreter re-runs the bundle, not the bare
   * REPL. When omitted, ensure-daemon falls back to a heuristic.
   */
  selfExecPrefixArgs?: string[]
  /**
   * Absolute path to *this* running executable when it is a `bun build
   * --compile` single-file binary (i.e. `process.execPath`). Left `undefined`
   * for the npm-installed CLI running under a `node`/`bun` interpreter. The
   * `sepilot upgrade` command uses this both to detect "I am a standalone
   * binary" and as the file to atomically replace.
   */
  selfBinaryPath?: string
  /** Foreground daemon entry (the `__daemon` subcommand calls this). */
  foregroundMain: (argv: string[]) => Promise<void>
  /** In-process daemon factory, used only as a last-resort fallback. */
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

/** Test-only reset. */
export function __resetStandaloneDaemonForTests(): void {
  hooks = undefined
}

/**
 * If this process is the standalone single-file bundle, returns how to launch
 * the daemon by re-entering ourselves (`<command> <...prefixArgs> __daemon`).
 *
 * - `bun --compile` single-file binary → `{ command: <binary>, args: ['__daemon'] }`
 * - plain `node dist/main.js` (interpreter) → `{ command: <node>, args: [<main.js>, '__daemon'] }`
 *
 * Returns `undefined` for the ordinary npm-installed CLI; the caller should
 * fall back to `resolveDaemonInvocation()` in that case. The bundle entry
 * normally registers `selfExecCommand` / `selfExecPrefixArgs` verbatim; when it
 * doesn't, this falls back to the same heuristic ensure-daemon uses (compare
 * `process.argv[1]` against `process.execPath` — equal ⇒ compiled binary,
 * different ⇒ interpreter exec).
 */
export function resolveStandaloneDaemonInvocation():
  | { command: string; args: string[] }
  | undefined {
  const standalone = getStandaloneDaemon()
  if (!standalone?.selfExec) return undefined
  const command = standalone.selfExecCommand ?? process.execPath
  const prefixArgs =
    standalone.selfExecPrefixArgs ??
    (() => {
      // Heuristic: in a `bun --compile` single-file build `process.execPath`
      // IS the executable, so `<exec> __daemon` re-enters it. When the same
      // bundle runs as a plain script under a Node/Bun interpreter,
      // `process.execPath` is the interpreter and the script path is
      // `process.argv[1]`, which must be passed through.
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
