import type {
  HookEvent,
  HookPayload,
  HookResult,
  IHookHandler,
  ToolCall,
} from '@sepilotd/core'
import { createLogger } from '../logger.js'
import type { HookRegistry } from '../hook/registry.js'
import { runBoundedCommand } from '../utils/bounded-command.js'
import { createHash } from 'node:crypto'

const logger = createLogger('command-hook')

const DEFAULT_COMMAND_HOOK_TIMEOUT_MS = 10_000
const MAX_CAPTURED_OUTPUT_BYTES = 64 * 1024

export interface CommandHookConfig {
  async?: boolean
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

  constructor(config: CommandHookConfig, private readonly background?: (payload: HookPayload, execute: (signal: AbortSignal) => Promise<unknown>) => void) {
    if (config.async && !config.event.startsWith('post:')) throw new Error('Async command hooks cannot gate pre events')
    this.config = config
    this.id = `command-hook:${config.event}:${createHash('sha256').update(JSON.stringify(config)).digest('hex').slice(0, 16)}`
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

  async handle(payload: HookPayload, signal?: AbortSignal): Promise<HookResult> {
    signal?.throwIfAborted()
    if (this.matcher) {
      const toolName = resolveToolName(payload)
      if (!toolName || !this.matcher.test(toolName)) {
        return { action: 'continue' }
      }
    }

    if (this.config.async) {
      if (!this.background) throw new Error('Background hook job service is unavailable')
      const ownedPayload = JSON.parse(JSON.stringify(payload)) as HookPayload
      this.background(ownedPayload, async (ownedSignal) => {
        const result = await runBoundedCommand({
          command: this.config.command, shell: true,
          env: { ...process.env, SEPILOTD_HOOK_EVENT: ownedPayload.event },
          stdin: `${JSON.stringify({ event: ownedPayload.event, data: ownedPayload.data })}\n`,
          signal: ownedSignal, timeoutMs: this.config.timeoutMs ?? DEFAULT_COMMAND_HOOK_TIMEOUT_MS,
          maxOutputBytes: MAX_CAPTURED_OUTPUT_BYTES,
        })
        if (result.exitCode !== 0) throw new Error(`Background hook exited ${result.exitCode ?? 'by signal'}: ${result.stderr.slice(0, 2000)}`)
        // Observational output, never interpreted as a retroactive allow/abort decision.
        return result
      })
      return { action: 'continue' }
    }
    const run = await this.runCommand(payload, signal)
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

  private async runCommand(
    payload: HookPayload,
    signal?: AbortSignal,
  ): Promise<{ exitCode: number; stdout: string; stderr: string } | null> {
    const timeoutMs = this.config.timeoutMs ?? DEFAULT_COMMAND_HOOK_TIMEOUT_MS
    try {
      const result = await runBoundedCommand({
        command: this.config.command,
        shell: true,
        env: { ...process.env, SEPILOTD_HOOK_EVENT: this.config.event },
        stdin: `${JSON.stringify({ event: payload.event, data: payload.data })}\n`,
        signal,
        timeoutMs,
        maxOutputBytes: MAX_CAPTURED_OUTPUT_BYTES,
      })
      // Signal termination is not exit 0: never interpret partial hook output.
      return result.exitCode === null ? null : { ...result, exitCode: result.exitCode }
    } catch (error) {
      logger.warn('command hook interrupted; continuing', {
        event: this.config.event,
        error: error instanceof Error ? error.message : String(error),
      })
      return null
    }
  }
}

/** Register command hooks from config. */
export function registerCommandHooks(
  hookRegistry: HookRegistry,
  hooks: readonly CommandHookConfig[],
): void {
  for (const config of hooks) {
    if (config.enabled === false) continue
    const handler = new CommandHookHandler(config, (payload, execute) => hookRegistry.startBackground(handler.id, payload, execute))
    hookRegistry.register(config.event, handler)
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
