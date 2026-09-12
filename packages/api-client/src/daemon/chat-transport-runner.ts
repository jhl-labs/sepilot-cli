import {
  getDaemonChatRunAbortMeta,
} from './chat-run-control.js'
import { createStreamEventController } from './chat-stream-controller.js'
import { createIncompleteDaemonChatStreamError } from './chat-transport-stream.js'
import type { Message as SurfaceMessage } from './chat-surface-types.js'
import type {
  ManagedSurfaceStreamRunParams,
  ManagedSurfaceStreamRunnerBaseParams,
  ManagedSurfaceStreamRunnerParams,
  SurfaceStreamRunParams,
} from './chat-transport-types.js'

export function createManagedSurfaceStreamRunner(
  base: ManagedSurfaceStreamRunnerBaseParams,
): (params: ManagedSurfaceStreamRunnerParams) => Promise<void> {
  return async (params: ManagedSurfaceStreamRunnerParams) => runManagedSurfaceChatStream({
    lifecycle: params.lifecycle,
    bindings: base.bindings,
    controllerOptions: {
      doneFallbackText: params.doneFallbackText,
      errorFallbackText: params.errorFallbackText,
      errorLabel: params.errorLabel,
      maxActivityItems: base.maxActivityItems,
      approvalResumeAvailable: params.approvalResumeAvailable,
      formatErrorMessage: base.formatErrorMessage,
      onEmptyCompletion: params.onEmptyCompletion,
    },
    createId: base.createId,
    createSeedMessages: params.createSeedMessages,
    pushActivity: base.pushActivity,
    onSession: base.onSession,
    onArtifacts: base.onArtifacts,
    tokenSpeedTracker: base.tokenSpeedTracker,
    formatRunError: base.formatRunError,
    errorAssistantFallbackText: params.errorAssistantFallbackText,
    rethrow: params.rethrow,
    run: params.run,
  })
}

export async function runManagedSurfaceChatStream(
  params: ManagedSurfaceStreamRunParams,
): Promise<void> {
  const { lifecycle, ...streamParams } = params

  await lifecycle.beforeStart?.()

  lifecycle.setLoading(true)
  if (lifecycle.clearError ?? true) {
    streamParams.bindings.setError(null)
  }
  streamParams.bindings.setStatus(lifecycle.statusText)
  streamParams.bindings.setAgentState(lifecycle.agentState)
  streamParams.pushActivity({
    id: streamParams.createId(),
    kind: lifecycle.activityKind ?? 'state',
    label: lifecycle.activityLabel,
    detail: lifecycle.activityDetail,
    status: lifecycle.activityStatus ?? 'running',
  })

  try {
    await runSurfaceChatStream(streamParams)
  } finally {
    lifecycle.setLoading(false)
    if (lifecycle.resetStatusOnFinish ?? true) {
      streamParams.bindings.setStatus(null)
    }
    await lifecycle.afterFinish?.()
  }
}

export async function runSurfaceChatStream(
  params: SurfaceStreamRunParams,
): Promise<void> {
  const assistantId = params.createId()
  const runStartedAt = Date.now()
  const seedMessages = params.createSeedMessages?.(assistantId) ?? [
    { id: assistantId, role: 'assistant', content: '' } satisfies SurfaceMessage,
  ]

  params.bindings.setMessages((prev) => [...prev, ...seedMessages])

  const streamController = createStreamEventController(
    params.bindings,
    {
      assistantId,
      createId: params.createId,
      ...params.controllerOptions,
    },
  )
  let sawTerminalEvent = false
  let assignedSessionId: string | null = null

  const assignSessionFromPayload = async (payload: unknown): Promise<boolean> => {
    if (!payload || typeof payload !== 'object' || !('sessionId' in payload)) {
      return false
    }

    const sessionId = (payload as { sessionId?: unknown }).sessionId
    if (typeof sessionId !== 'string' || !sessionId || sessionId === assignedSessionId) {
      return false
    }

    assignedSessionId = sessionId
    await params.onSession?.(sessionId)
    return true
  }

  try {
    await params.run(async (payload) => {
      const assignedFromPayload = await assignSessionFromPayload(payload)
      if (assignedFromPayload && !('type' in payload)) {
        return
      }

      if ('artifacts' in payload) {
        await params.onArtifacts?.(payload.artifacts)
        return
      }

      if (!('type' in payload)) {
        return
      }

      if (payload.type === 'done' || payload.type === 'error') {
        sawTerminalEvent = true
      }
      if (payload.type === 'done') {
        params.tokenSpeedTracker?.recordRun({
          startedAt: runStartedAt,
          finishedAt: Date.now(),
          usage: payload.usage,
        })
      }
      streamController.handleEvent(payload)
    })

    if (!sawTerminalEvent) {
      throw createIncompleteDaemonChatStreamError()
    }
  } catch (error) {
    const abortMeta = getDaemonChatRunAbortMeta(error)
    if (abortMeta) {
      params.bindings.setAgentState('done')
      params.bindings.setStatus(null)
      params.bindings.setError(null)
      if (!streamController.getAssistantContent()) {
        streamController.updateAssistant(abortMeta.assistantFallbackText)
      }
      params.pushActivity({
        id: params.createId(),
        kind: 'state',
        label: abortMeta.label,
        detail: abortMeta.detail,
        status: abortMeta.status,
      })
      return
    }

    const detail = params.formatRunError(error)
    params.bindings.setAgentState('error')
    params.bindings.setError(detail)
    streamController.updateAssistant(
      streamController.getAssistantContent() || params.errorAssistantFallbackText,
    )
    params.pushActivity({
      id: params.createId(),
      kind: 'error',
      label: params.controllerOptions.errorLabel,
      detail,
      status: 'error',
    })

    if (params.rethrow) {
      throw error
    }
  }
}
