import { spawn } from 'node:child_process'
import type {
  HookEvent,
  HookPayload,
  HookResult,
  IHookHandler,
  IHookRegistry,
  ToolCall,
} from '@sepilotd/core'
import { createLogger } from '../logger.js'
import type { HookRegistry } from '../hook/registry.js'

const logger = createLogger('command-hook')

const DEFAULT_COMMAND_HOOK_TIMEOUT_MS = 10_000
const MAX_CAPTURED_OUTPUT_BYTES = 64 * 1024

export interface CommandHookConfig {
  enabled?: boolean
  event: HookEvent
  /** Regex matched against the tool name for tool-scoped events. */
  toolMatcher?: string
  /** Shell command. Receives the hook payload as JSON on stdin. */
  command: string
  timeoutMs?: number
}

/**
 * User-configured shell command run as a hook handler (Claude Code
 * settings-hook parity). Contract:
 *  - the payload is written to stdin as one newline-terminated JSON record
 *    (`{event, data}\n`) so append-only consumers produce valid JSONL,
 *  - exit 2 aborts the gated operation; stderr becomes the reason,
 *  - exit 0 with a JSON object on stdout may return
 *    `{action, reason}` or `{toolArguments}` (pre:tool:execute only),
 *  - timeouts, spawn errors, and other exit codes count as `continue`
 *    so a broken hook never takes down tool execution.
 */
export class CommandHookHandler implements IHookHandler {
  readonly id: string
  readonly priority = 50
  private readonly config: CommandHookConfig
  private readonly matcher?: RegExp

  constructor(config: CommandHookConfig) {
    this.config = config
    this.id = `command-hook:${config.event}:${config.command.slice(0, 40)}`
    if (config.toolMatcher) {
      try {
        this.matcher = new RegExp(config.toolMatcher)
      } catch {
        logger.warn('invalid toolMatcher regex; hook will match no tools', {
          toolMatcher: config.toolMatcher,
        })
        this.matcher = /$^/
      }
    }
  }

  async handle(payload: HookPayload): Promise<HookResult> {
    if (this.matcher) {
      const toolName = resolveToolName(payload)
      if (!toolName || !this.matcher.test(toolName)) {
        return { action: 'continue' }
      }
    }

    const run = await this.runCommand(payload)
    if (!run) return { action: 'continue' }

    if (run.exitCode === 2) {
      const reason = (run.stderr.trim() || run.stdout.trim())
        || `command hook exited 2 (${this.config.command})`
      return { action: 'abort', reason }
    }
    if (run.exitCode !== 0) {
      logger.warn('command hook exited with unexpected code; continuing', {
        event: this.config.event,
        exitCode: run.exitCode,
      })
      return { action: 'continue' }
    }

    return this.interpretStdout(payload, run.stdout)
  }

  private interpretStdout(payload: HookPayload, stdout: string): HookResult {
    const trimmed = stdout.trim()
    if (!trimmed.startsWith('{')) return { action: 'continue' }
    let parsed: Record<string, unknown>
    try {
      parsed = JSON.parse(trimmed) as Record<string, unknown>
    } catch {
      return { action: 'continue' }
    }

    if (parsed.action === 'abort' || parsed.action === 'skip') {
      return {
        action: parsed.action,
        ...(typeof parsed.reason === 'string' ? { reason: parsed.reason } : {}),
      }
    }

    const toolCall = payload.data.toolCall as ToolCall | undefined
    if (
      toolCall
      && parsed.toolArguments
      && typeof parsed.toolArguments === 'object'
      && !Array.isArray(parsed.toolArguments)
    ) {
      return {
        action: 'continue',
        modifiedPayload: {
          ...payload,
          data: {
            ...payload.data,
            toolCall: {
              ...toolCall,
              arguments: parsed.toolArguments as Record<string, unknown>,
            },
          },
        },
      }
    }
    return { action: 'continue' }
  }

  private runCommand(
    payload: HookPayload,
  ): Promise<{ exitCode: number; stdout: string; stderr: string } | null> {
    const timeoutMs = this.config.timeoutMs ?? DEFAULT_COMMAND_HOOK_TIMEOUT_MS
    return new Promise((resolvePromise) => {
      let settled = false
      const settle = (value: { exitCode: number; stdout: string; stderr: string } | null) => {
        if (settled) return
        settled = true
        clearTimeout(timer)
        resolvePromise(value)
      }

      let child
      try {
        child = spawn(this.config.command, {
          shell: true,
          stdio: ['pipe', 'pipe', 'pipe'],
          env: { ...process.env, SEPILOTD_HOOK_EVENT: this.config.event },
        })
      } catch (error) {
        logger.warn('command hook spawn failed; continuing', {
          event: this.config.event,
          error: error instanceof Error ? error.message : String(error),
        })
        settle(null)
        return
      }

      const timer = setTimeout(() => {
        logger.warn('command hook timed out; continuing', {
          event: this.config.event,
          timeoutMs,
        })
        child.kill('SIGKILL')
        settle(null)
      }, timeoutMs)

      let stdout = ''
      let stderr = ''
      child.stdout.on('data', (chunk: Buffer) => {
        if (stdout.length < MAX_CAPTURED_OUTPUT_BYTES) stdout += chunk.toString('utf8')
      })
      child.stderr.on('data', (chunk: Buffer) => {
        if (stderr.length < MAX_CAPTURED_OUTPUT_BYTES) stderr += chunk.toString('utf8')
      })
      child.on('error', (error) => {
        logger.warn('command hook process error; continuing', {
          event: this.config.event,
          error: error.message,
        })
        settle(null)
      })
      child.on('close', (code) => {
        settle({ exitCode: code ?? 0, stdout, stderr })
      })

      child.stdin.on('error', () => {
        // The command may exit without reading stdin; ignore EPIPE.
      })
      child.stdin.end(`${JSON.stringify({ event: payload.event, data: payload.data })}\n`)
    })
  }
}

/** Register command hooks from config. */
export function registerCommandHooks(
  hookRegistry: IHookRegistry,
  hooks: readonly CommandHookConfig[],
): void {
  for (const config of hooks) {
    if (config.enabled === false) continue
    hookRegistry.register(config.event, new CommandHookHandler(config))
  }
}

/** Replace all command hooks (config rebind path). */
export function replaceCommandHooks(
  hookRegistry: HookRegistry,
  hooks: readonly CommandHookConfig[],
): void {
  hookRegistry.removeHandlers((_event, handler) => handler instanceof CommandHookHandler)
  registerCommandHooks(hookRegistry, hooks)
}

function resolveToolName(payload: HookPayload): string | undefined {
  const toolCall = payload.data.toolCall as ToolCall | undefined
  if (toolCall?.name) return toolCall.name
  return typeof payload.data.tool === 'string' ? payload.data.tool : undefined
}
