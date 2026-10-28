import { randomUUID } from 'node:crypto'
import type { BackgroundSubagents } from '../../jobs/subagent.js'
import { AutonomyLevel } from '@sepilotd/core'
import type { DelegatedAgentExecutionPolicy } from '../../tools/registry.js'
import type { subagentDispatchRequestSchema } from '../routes/subagents-schema.js'

export interface SubagentRequestAuthority {
  callerExecutionPolicy?: () => DelegatedAgentExecutionPolicy | null
  parentSelection?: (sessionId: string) => Promise<{
    provider: string
    model: string
    cwd?: string
    workspaceIsolation?: 'policy' | 'strict'
  } | null>
}

export async function prepareSubagentRequest(
  deps: SubagentRequestAuthority,
  input: ReturnType<typeof subagentDispatchRequestSchema.parse>,
) {
  const { autonomy, ...dispatchInput } = input
  const suppliedPolicy = deps.callerExecutionPolicy?.()
  const policy = suppliedPolicy
    ? { ...suppliedPolicy, allowedToolNames: [...suppliedPolicy.allowedToolNames] }
    : null
  if (policy && autonomy) {
    const order = ['readonly', 'supervised', 'accept-edits', 'workspace-write', 'autonomous']
    if (order.indexOf(autonomy) < order.indexOf(policy.autonomy))
      policy.autonomy = autonomy as AutonomyLevel
  }
  const parent =
    dispatchInput.parentSessionId && deps.parentSelection
      ? await deps.parentSelection(dispatchInput.parentSessionId)
      : null
  if (dispatchInput.parentSessionId && deps.parentSelection && !parent) {
    throw Object.assign(new Error('Parent session is unavailable'), { code: 'SUBAGENT_NOT_FOUND' })
  }
  return {
    ...dispatchInput,
    ...(parent
      ? {
          parentProvider: parent.provider,
          parentModel: parent.model,
          cwd: dispatchInput.cwd ?? parent.cwd,
          ...(parent.cwd && parent.workspaceIsolation !== 'policy'
            ? { workspaceRoot: parent.cwd }
            : {}),
        }
      : {}),
    ...(policy ? { parentExecutionPolicy: policy } : {}),
  }
}

export function admitHttpSubagent(
  service: BackgroundSubagents | null | undefined,
  input: Awaited<ReturnType<typeof prepareSubagentRequest>>,
) {
  const { background: _background, ...request } = input
  if (!service || !request.parentExecutionPolicy)
    throw Object.assign(new Error('Background subagent service not initialized'), {
      code: 'SERVICE_UNAVAILABLE',
      statusCode: 503,
    })
  if (request.contextMode === 'fork' && !request.parentSessionId)
    throw Object.assign(new Error('fork context requires parentSessionId'), {
      code: 'SUBAGENT_FORK_UNAVAILABLE',
      statusCode: 400,
    })
  if (request.requestKey && !request.parentSessionId)
    throw Object.assign(new Error('requestKey requires an explicit parentSessionId'), {
      code: 'INVALID_REQUEST',
      statusCode: 400,
    })
  const parentSessionId = request.parentSessionId ?? `api-subagent:${randomUUID()}`
  const job = service.start({
    ...request,
    parentSessionId,
    parentExecutionPolicy: request.parentExecutionPolicy,
  })
  return { ...job, total: 1, parentSessionId }
}

export function backgroundAdmissionError(
  err: unknown,
): { statusCode: 400 | 409 | 429 | 503; error: { code: string; message: string } } | null {
  const code = (err as { code?: string } | null)?.code
  const status =
    code === 'SUBAGENT_QUEUE_FULL'
      ? 429
      : code === 'SUBAGENT_REQUEST_CONFLICT'
        ? 409
        : code === 'SERVICE_UNAVAILABLE'
          ? 503
          : ['SUBAGENT_FORK_UNAVAILABLE', 'INVALID_REQUEST'].includes(code ?? '')
            ? 400
            : null
  return status && code
    ? {
        statusCode: status,
        error: { code, message: err instanceof Error ? err.message : String(err) },
      }
    : null
}
