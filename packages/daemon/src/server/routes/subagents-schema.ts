import { z } from 'zod'

/**
 * Request body for `POST /api/v1/subagents/dispatch`.
 *
 * The route is a thin HTTP wrapper over `SubagentDispatcher.dispatch`.
 * Schema mirrors `SubagentDispatchInput` and applies minimal validation
 * (non-empty prompt, positive integer maxIterations, non-empty tool
 * names) — privilege-escalation rejection happens inside the dispatcher
 * and surfaces as a 400 with `code: 'SUBAGENT_TOOL_ESCALATION'`.
 */
export const subagentDispatchRequestSchema = z.object({
  prompt: z.string().min(1, 'prompt required'),
  system: z.string().optional(),
  category: z.string().min(1).optional(),
  agentId: z.string().min(1).optional(),
  maxIterations: z.number().int().positive().optional(),
  tools: z.array(z.string().min(1)).optional(),
  model: z.string().min(1).optional(),
  parentSessionId: z.string().min(1).optional(),
})
