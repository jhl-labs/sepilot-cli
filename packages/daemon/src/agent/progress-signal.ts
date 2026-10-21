import type { Message } from '@sepilotd/core'
import { observeWorkProgress, type WorkProgress } from './work-progress.js'

export const CONTINUATION_MARKER_METADATA_KEY = 'continuationCycle'
export const CURRENT_TURN_USER_METADATA_KEY = 'currentAgentTurnUserInput'
const TOOL_RESULT_STATUS_KEY = 'toolResultStatus'
const TOOL_RESULT_BLOCK_SOURCE_KEY = 'toolResultBlockSource'
const ARTIFACT_MUTATION_TOOLS: ReadonlySet<string> = new Set([
  'fs.write',
  'fs.append',
  'fs.edit',
  'apply_patch',
])

export interface ProgressSignal {
  newToolEvidence: number
  newArtifactMutations: number
  newVerifiedEvidence: number
  progressed: boolean
}

export function isArtifactMutationTool(name: string | undefined): boolean {
  return name !== undefined && ARTIFACT_MUTATION_TOOLS.has(name)
}

export function findProgressMarkerIndex(messages: readonly Message[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const cycle = messages[index]?.metadata?.[CONTINUATION_MARKER_METADATA_KEY]
    if (typeof cycle === 'number') return index
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message?.role === 'user' && message.metadata?.[CURRENT_TURN_USER_METADATA_KEY] === true) {
      return index
    }
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') return index
  }
  return -1
}

function messageText(message: Message): string {
  if (typeof message.content === 'string') return message.content
  return message.content
    .map((part) => (part.type === 'text' ? part.text : ''))
    .join('')
}

export function evaluateProgressSince(
  messages: readonly Message[],
  marker: number = findProgressMarkerIndex(messages),
): ProgressSignal {
  const toolNamesById = new Map<string, string>()
  const toolInputsById = new Map<string, Record<string, unknown>>()
  for (const message of messages) {
    for (const call of message.toolCalls ?? []) {
      toolNamesById.set(call.id, call.name)
      toolInputsById.set(call.id, call.arguments ?? {})
    }
  }
  let newToolEvidence = 0
  let newArtifactMutations = 0
  const newVerifiedEvidence = 0
  let progress: WorkProgress = { revision: 0, seen: [] }
  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index]!
    if (message.role === 'tool') {
      const metadata = message.metadata ?? {}
      if (metadata[TOOL_RESULT_BLOCK_SOURCE_KEY] !== undefined) continue
      const name = message.name ?? (message.toolCallId ? toolNamesById.get(message.toolCallId) : undefined)
      const next = observeWorkProgress(progress, {
        tool: name ?? '', input: message.toolCallId ? toolInputsById.get(message.toolCallId) : undefined, output: messageText(message),
        status: String(metadata[TOOL_RESULT_STATUS_KEY] ?? 'unknown'),
        executionObserved: metadata.toolResultExecutionObserved === true,
        securityEffect: String(metadata.toolResultSecurityEffect ?? 'unknown'),
      })
      const novel = next.revision > progress.revision
      progress = next
      if (index > marker && novel) {
        newToolEvidence += 1
        if (metadata[TOOL_RESULT_STATUS_KEY] === 'success' && isArtifactMutationTool(name)) newArtifactMutations += 1
      }
    }
  }
  return {
    newToolEvidence,
    newArtifactMutations,
    newVerifiedEvidence,
    progressed: newToolEvidence > 0 || newArtifactMutations > 0 || newVerifiedEvidence > 0,
  }
}

export function parseRunWallClockBudgetMs(raw: string | undefined): number | undefined {
  if (raw === undefined || raw.trim() === '') return undefined
  const parsed = Number(raw)
  if (!Number.isFinite(parsed)) return undefined
  return parsed > 0 ? Math.floor(parsed) : undefined
}

export function resolveWallClockSynthesisReserveMs(
  budgetMs: number,
  raw: string | undefined = process.env.SEPILOTD_RUN_WALL_SYNTHESIS_RESERVE_MS,
): number {
  const configured = raw === undefined || raw.trim() === '' ? Number.NaN : Number(raw)
  const reserve = Number.isFinite(configured)
    ? Math.max(0, Math.floor(configured))
    : Math.min(Math.max(Math.floor(budgetMs * 0.15), 60_000), 300_000)
  return Math.min(reserve, Math.floor(budgetMs / 2))
}

export function resolveRunWallClock(
  startedAt: number,
  deadlineAt?: number,
  raw: string | undefined = process.env.SEPILOTD_RUN_MAX_WALL_MS,
): { startedAt: number; budgetMs?: number } {
  const host = parseRunWallClockBudgetMs(raw)
  if (deadlineAt !== undefined && (!Number.isSafeInteger(deadlineAt) || deadlineAt <= 0)) throw new RangeError('deadlineAt must be a positive safe integer')
  const caller = deadlineAt === undefined ? undefined : Math.max(0, deadlineAt - startedAt)
  return { startedAt, budgetMs: caller === undefined ? host : host === undefined ? caller : Math.min(host, caller) }
}

export function runRequestTimeoutMs(clock: { startedAt: number; budgetMs?: number }, preferred?: number, now = Date.now()): number | undefined {
  if (clock.budgetMs === undefined) return preferred
  const remaining = Math.max(1, Math.floor(clock.startedAt + clock.budgetMs - now))
  return preferred !== undefined && preferred > 0 ? Math.min(preferred, remaining) : remaining
}
