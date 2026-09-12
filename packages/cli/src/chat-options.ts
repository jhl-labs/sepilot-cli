import type { ChatSkillRef, DaemonAgentMode } from '@sepilotd/api-client'
import {
  DEFAULT_CHAT_OPTION_DEFAULTS,
  sanitizeChatOptions,
  type ChatStreamOptions,
} from '@sepilotd/api-client'

export const DEFAULT_CLI_AGENT_MODE = 'react' as const

export interface CliChatOptionInput {
  model?: string
  provider?: string
  persona?: string
  mode?: string | DaemonAgentMode
  thinkingLevel?: string
  projectId?: string | null
  fileIds?: string[]
  personaIds?: string[]
  maxTokens?: number | null
  cwd?: string
  workspaceRoot?: string
  skillRefs?: ChatSkillRef[]
  /** Explicit per-turn autonomy selection; daemon policy still applies. */
  autonomy?: string
}

export function parsePositiveIntegerCliOption(
  value: string | number | null | undefined,
  flagName: string,
): number | undefined {
  if (value === undefined || value === null || value === '') return undefined

  const parsed = typeof value === 'number' ? value : Number(value)
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${flagName} must be a positive integer (got: ${value})`)
  }

  return parsed
}

const CLI_AUTONOMY_LEVELS = new Set([
  'readonly',
  'accept-edits',
  'workspace-write',
  'supervised',
  'autonomous',
])

const CLI_THINKING_LEVELS = new Set(['off', 'low', 'medium', 'high', 'max'])

function normalizeAutonomy(
  value?: string,
): 'readonly' | 'accept-edits' | 'workspace-write' | 'supervised' | 'autonomous' | undefined {
  const trimmed = value?.trim().toLowerCase()
  if (!trimmed) return undefined
  if (!CLI_AUTONOMY_LEVELS.has(trimmed)) {
    throw new Error(
      `--autonomy must be one of readonly, accept-edits, workspace-write, supervised, autonomous (got: ${value})`,
    )
  }
  return trimmed as 'readonly' | 'accept-edits' | 'workspace-write' | 'supervised' | 'autonomous'
}

function normalizeMode(mode?: string | DaemonAgentMode): DaemonAgentMode | undefined {
  const trimmed = mode?.trim()
  return trimmed ? trimmed as DaemonAgentMode : undefined
}

function normalizeThinkingLevel(value?: string): string | undefined {
  const trimmed = value?.trim().toLowerCase()
  if (!trimmed) return undefined
  if (!CLI_THINKING_LEVELS.has(trimmed)) {
    throw new Error(
      `--thinking-level must be one of off, low, medium, high, max (got: ${value})`,
    )
  }
  return trimmed
}

export function buildCliChatOptions(
  options?: CliChatOptionInput,
): ChatStreamOptions | undefined {
  if (!options) return undefined
  const projectId = options.projectId?.trim()
  const cwd = options.cwd ?? (projectId ? undefined : process.cwd())
  // A launch directory is not an OS sandbox request. Explicit roots remain
  // strict; otherwise the daemon's configured tool/approval policy applies.
  const workspaceRoot = options.workspaceRoot
  const mode = normalizeMode(options.mode) ?? DEFAULT_CLI_AGENT_MODE

  const sanitized = sanitizeChatOptions(
    {
      model: options.model,
      provider: options.provider,
      persona: options.persona,
      mode,
      thinkingLevel: normalizeThinkingLevel(options.thinkingLevel),
      projectId: options.projectId ?? undefined,
      fileIds: options.fileIds,
      personaIds: options.personaIds,
      maxTokens: options.maxTokens ?? undefined,
      cwd,
      workspaceRoot,
      skillRefs: options.skillRefs,
      autonomy: normalizeAutonomy(options.autonomy),
    },
    { ...DEFAULT_CHAT_OPTION_DEFAULTS, thinkingLevel: undefined },
  )
  // Even `auto` is an explicit CLI choice; omission inherits shared daemon UI defaults.
  return { ...sanitized, mode }
}
