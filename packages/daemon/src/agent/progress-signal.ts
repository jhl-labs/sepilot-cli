import type { Message } from '@sepilotd/core'
import { observeWorkProgress, type WorkProgress } from './work-progress.js'

/**
 * Pure progress accounting shared by the react engine and (later) the graph
 * runtime. Dependency-free on purpose: only message metadata conventions are
 * read, so callers can pass any transcript slice.
 *
 * Conventions read here (all daemon-owned, never copied from tool output):
 * - tool results tagged `toolResultStatus: 'success'` are evidence;
 * - results tagged `toolResultBlockSource: 'policy' | 'approval'` are
 *   friction, never evidence;
 * - artifact mutation tools are the file-edit family;
 * - model-authored completion prose is never execution evidence.
 */

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

/** Default absolute wall-clock ceiling for one run when the env is unset. */
export const DEFAULT_RUN_MAX_WALL_MS = 30 * 60 * 1000

export interface ProgressSignal {
  newToolEvidence: number
  newArtifactMutations: number
  newVerifiedEvidence: number
  progressed: boolean
}

export function isArtifactMutationTool(name: string | undefined): boolean {
  return name !== undefined && ARTIFACT_MUTATION_TOOLS.has(name)
}

/**
 * Index of the message that starts the current progress window: the latest
 * continuation marker, else the current-turn user message, else the latest
 * user message, else -1 (whole transcript).
 */
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

/**
 * Count fresh progress after `marker` (a message index, exclusive). Pass the
 * result of `findProgressMarkerIndex` or an explicit index.
 */
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

/**
 * Parse `SEPILOTD_RUN_MAX_WALL_MS`: unset/invalid → 30 minutes, `0` (or a
 * negative value) → disabled (`undefined`).
 */
export function parseRunWallClockBudgetMs(raw: string | undefined): number | undefined {
  if (raw === undefined || raw.trim() === '') return DEFAULT_RUN_MAX_WALL_MS
  const parsed = Number(raw)
  if (!Number.isFinite(parsed)) return DEFAULT_RUN_MAX_WALL_MS
  return parsed > 0 ? Math.floor(parsed) : undefined
}
