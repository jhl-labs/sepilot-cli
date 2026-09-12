import { z } from 'zod'

export const SWARM_AGENTS = ['claude', 'codex', 'gemini', 'opencode'] as const

export const createRunSchema = z.object({
  goal: z.string().min(1),
  cwd: z.string().min(1).optional(),
  worktree: z.string().min(1).optional(),
  autonomy: z
    .enum(['readonly', 'accept-edits', 'workspace-write', 'supervised', 'autonomous'])
    .optional(),
  autoApproveAgents: z.boolean().optional().default(true),
  warmPool: z.array(z.enum(SWARM_AGENTS)).optional().default(['claude']),
  /**
   * When true, the daemon spawns the warm pool but does NOT start the
   * supervisor agent loop. The run lives until DELETE; the user drives the
   * agents directly via the keys endpoint or attach mode. Useful for tests,
   * scripts, and "human-in-the-loop" sessions where you don't want an LLM
   * supervisor making decisions for you.
   */
  noSupervisor: z.boolean().optional().default(false),
})

export const sendKeysSchema = z.union([
  z.object({
    keys: z.string(),
    enter: z.boolean().optional(),
  }),
  z.object({
    keyName: z.string().min(1),
  }),
  z.object({
    resize: z.object({
      cols: z.number().int().positive(),
      rows: z.number().int().positive(),
    }),
  }),
])

export const interruptAgentSchema = z.object({
  escape: z.boolean().optional(),
})

export const attachAgentSchema = z.object({
  renew: z.boolean().optional(),
})

export const driveAgentSchema = z.object({
  prompt: z.string().min(1),
  followups: z.array(z.string().min(1)).optional(),
  continue_prompt: z.string().min(1).optional(),
  max_turns: z.number().int().min(1).max(100).optional(),
  stop_patterns: z.array(z.string().min(1)).optional(),
  timeout_sec: z.number().min(0.1).max(1800).optional(),
  stable_sec: z.number().min(0.1).max(60).optional(),
  min_wait_ms: z.number().min(0).max(60000).optional(),
  poll_ms: z.number().min(25).max(5000).optional(),
  timeout_retries: z.number().int().min(0).max(5).optional(),
})
