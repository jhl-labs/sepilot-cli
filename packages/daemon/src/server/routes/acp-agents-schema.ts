import { z } from 'zod'
import { EXTERNAL_ACP_AGENT_NAMES } from '../../acp/external-agent.js'

export const externalAcpDispatchRequestSchema = z.object({
  prompt: z.string().min(1),
  agent: z.enum(EXTERNAL_ACP_AGENT_NAMES).optional(),
  cwd: z.string().min(1).optional(),
  sessionId: z.string().min(1).optional(),
  timeoutMs: z.number().int().positive().max(600_000).optional(),
})
