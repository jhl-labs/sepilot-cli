import { z } from 'zod'
import { resolveAgentMaxIterations } from '../../agent/iteration-budget.js'
import { MAX_CHAT_ATTACHMENTS } from '../../media/pipeline.js'
import { safeIdSchema } from '../../utils/safe-id.js'

export const chatAttachmentSchema = z.object({
  type: z.string(),
  path: z.string().optional(),
  filename: z.string().optional(),
  url: z.string().optional(),
})

export const chatSkillRefSchema = z.object({
  name: z.string().min(1),
})

export const chatAutonomySchema = z.enum([
  'readonly',
  'accept-edits',
  'workspace-write',
  'supervised',
  'autonomous',
])

export const chatRequestSchema = z.object({
  message: z.string(),
  sessionId: safeIdSchema.optional(),
  messageId: z.string().trim().min(1).max(256).optional(),
  lastEventId: z.string().trim().min(1).max(256).optional(),
  model: z.string().optional(),
  provider: z.string().optional(),
  tags: z.array(z.string().trim().min(1).max(64)).max(16).optional(),
  persona: z.string().optional(),
  /**
   * Multi-persona panel mode. When non-empty and `mode === 'persona-panel'`,
   * each id becomes a panelist that answers the same user turn in sequence.
   * Capped at 6 to keep latency / token cost predictable.
   */
  personaIds: z.array(z.string().min(1)).max(6).optional(),
  /**
   * `sequential` is the deterministic multi-perspective contract: every
   * resolved persona answers exactly once in request order while seeing the
   * transcript accumulated so far. `moderated` retains the facilitated
   * meeting workflow with dynamic floor selection.
   */
  panelStrategy: z.enum(['sequential', 'moderated']).optional(),
  thinkingLevel: z.enum(['off', 'low', 'medium', 'high', 'max']).optional(),
  maxTokens: z.number().int().positive().max(2_000_000).optional(),
  /**
   * Agent-loop iteration cap for this turn. One iteration is one LLM round
   * (which may emit one or more parallel tool calls). Defaults to the
   * `SEPILOTD_CHAT_MAX_ITERATIONS` env (or 50 if unset). Multi-pass skills
   * such as `software-architect` reverse-engineering a non-trivial codebase
   * routinely need more; the upper bound is `500` to leave room while
   * keeping a stop-the-bleeding hard cap on runaway loops.
   */
  maxIterations: z.number().int().min(1).max(500).optional(),
  temperature: z.number().min(0).max(2).optional(),
  mode: z.string().trim().min(1).optional(),
  writingDocId: safeIdSchema.optional(),
  projectId: safeIdSchema.optional(),
  fileIds: z.array(z.string()).max(MAX_CHAT_ATTACHMENTS).optional(),
  attachments: z.array(chatAttachmentSchema).max(MAX_CHAT_ATTACHMENTS).optional(),
  cwd: z.string().optional(),
  workspaceRoot: z.string().optional(),
  skillRefs: z.array(chatSkillRefSchema).optional(),
  /**
   * Optional per-request tool allowlist. Omit to expose the normal direct-API
   * registry; pass [] when the caller needs a tool-free model turn.
   */
  toolNames: z.array(z.string().trim().min(1).max(128)).max(128).optional(),
  // Composer-level toggles surfaced by the desktop chat shell. `ragEnabled`
  // controls document retrieval for this turn; image generation remains a
  // forwarded intent for routes that know how to use it.
  ragEnabled: z.boolean().optional(),
  imageGenEnabled: z.boolean().optional(),
  /**
   * Streaming-only opt-in. Omit it to keep the historical buffered text_delta
   * contract used by CLI, mobile, Telegram, and existing API consumers. `live`
   * streams only final-answer text that has passed the answer-protocol gate.
   */
  textDeltaMode: z.enum(['buffered', 'live']).optional(),
  inputTrustLevel: z.enum(['trusted', 'untrusted']).optional(),
  /**
   * Explicit per-turn autonomy selection. Independent channel/policy ceilings
   * still apply after this local session value is resolved.
   * This lets automation choose readonly and an interactive authenticated
   * client choose workspace-write without changing the configured default.
   */
  autonomy: chatAutonomySchema.optional(),
  /**
   * Per-turn HITL guard. Policy-allowed side effects still require a fresh
   * approval; read-only tools and hard policy denials are unchanged.
   */
  requireToolApproval: z.boolean().optional(),
  /**
   * Per-request kill switch for the IntentRouter. An explicit concrete mode
   * is always authoritative; auto/omitted mode lets the router choose while
   * persona and skill values remain routing hints. Pass `{enabled: false}`
   * to use the body's `mode`/`persona`/`skillRefs` verbatim for this turn.
   */
  intentRouting: z.object({
    enabled: z.boolean().optional(),
  }).optional(),
})

export type ChatBody = z.infer<typeof chatRequestSchema>

const desktopExternalAgentModes = new Set([
  'external-claude',
  'external-codex',
  'external-gemini',
  'external-opencode',
])

export function isDesktopExternalAgentMode(mode: unknown): boolean {
  return typeof mode === 'string' && desktopExternalAgentModes.has(mode)
}

export function resolveChatMaxIterations(body: Pick<ChatBody, 'maxIterations'>): number {
  return resolveAgentMaxIterations(body)
}
