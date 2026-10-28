import { z } from 'zod'

export const subagentDispatchRequestSchema = z.object({
  prompt: z.string().refine((value) => value.trim().length > 0, 'prompt required'),
  system: z.string().optional(),
  category: z.string().min(1).optional(),
  agentId: z.string().min(1).optional(),
  maxIterations: z.number().int().positive().optional(),
  tools: z.array(z.string().min(1)).optional(),
  model: z.string().min(1).optional(),
  provider: z.string().min(1).optional(),
  autonomy: z.enum(['readonly', 'supervised', 'accept-edits', 'workspace-write', 'autonomous']).optional(),
  parentSessionId: z.string().min(1).optional(),
  contextPacket: z.string().optional(),
  cwd: z.string().min(1).optional(),
  isolation: z.literal('worktree').optional(),
  worktreeBase: z.enum(['head', 'working-tree']).optional(),
  contextMode: z.enum(['packet', 'fork']).optional(),
  background: z.boolean().optional(),
  requestKey: z.string().min(1).max(128).optional(),
})
