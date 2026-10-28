import { modelRoutingHintSchema } from '../../providers/model-routing-policy.js'
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

export const CHAT_TOOL_NAMES_MAX = 128

export const chatRequestSchema = z.object({
  message: z.string(),
  deadlineAt: z.number().int().positive().max(Number.MAX_SAFE_INTEGER).optional(),
  completionRequest: z.string().trim().min(1).max(100_000).optional(),
  sessionId: safeIdSchema.optional(),
  messageId: z.string().trim().min(1).max(256).optional(),
  lastEventId: z.string().trim().min(1).max(256).optional(),
  model: z.string().optional(),
  provider: z.string().optional(),
  modelRouting: modelRoutingHintSchema.optional(),
  tags: z.array(z.string().trim().min(1).max(64)).max(16).optional(),
  persona: z.string().optional(),
  bot: z.string().trim().min(1).max(64).optional(),
  bots: z.array(z.string().trim().min(1).max(64)).max(6).optional(),
  personaIds: z.array(z.string().min(1)).max(6).optional(),
  panelStrategy: z.enum(['sequential', 'moderated']).optional(),
  thinkingLevel: z.enum(['auto', 'off', 'low', 'medium', 'high', 'max']).optional(),
  maxTokens: z.number().int().positive().max(2_000_000).optional(),
  maxIterations: z.number().int().min(1).max(500).optional(),
  hardMaxIterations: z.boolean().optional(),
  temperature: z.number().min(0).max(2).optional(),
  mode: z.string().trim().min(1).optional(),
  writingDocId: safeIdSchema.optional(),
  projectId: safeIdSchema.optional(),
  fileIds: z.array(z.string()).max(MAX_CHAT_ATTACHMENTS).optional(),
  attachments: z.array(chatAttachmentSchema).max(MAX_CHAT_ATTACHMENTS).optional(),
  cwd: z.string().optional(),
  workspaceRoot: z.string().optional(),
  skillRefs: z.array(chatSkillRefSchema).optional(),
  toolNames: z.array(z.string().trim().min(1).max(128)).max(CHAT_TOOL_NAMES_MAX).optional(),
  memoryPolicy: z.object({ read: z.boolean().optional(), write: z.boolean().optional() }).strict().optional(),
  ragEnabled: z.boolean().optional(),
  imageGenEnabled: z.boolean().optional(),
  textDeltaMode: z.enum(['buffered', 'live']).optional(),
  inputTrustLevel: z.enum(['trusted', 'untrusted']).optional(),
  localContext: z.enum(['default', 'none']).optional(),
  autonomy: chatAutonomySchema.optional(),
  requireToolApproval: z.boolean().optional(),
  intentRouting: z.object({
    enabled: z.boolean().optional(),
  }).optional(),
})

export type ChatBody = z.infer<typeof chatRequestSchema>

export function excludesLocalContext(body: { localContext?: unknown }): boolean {
  return body.localContext === 'none'
}

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

export function resolveChatHardMaxIterations(
  body: Pick<ChatBody, 'hardMaxIterations'>,
): boolean {
  return body.hardMaxIterations === true
}
