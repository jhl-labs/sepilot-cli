export interface StreamingProgressInput {
  isStreaming: boolean
  streamStatus: string | null
  streamStartedAt: number | null
  now: number
  currentMessage: string
  liveOutputTokens: number
  hasPendingApproval: boolean
  hasPendingQuestion?: boolean
}

export interface StreamingProgress {
  indicator: string
  phase: string
  elapsedMs: number
  elapsedLabel: string
  estimatedOutputTokens: number
  tokensPerSecond: number | null
  tokensPerSecondLabel: string | null
  label: string
}

const PROGRESS_FRAMES = ['[-]', '[\\]', '[|]', '[/]'] as const

export function estimateTokensFromText(text: string): number {
  const normalized = text.trim()
  if (!normalized) {
    return 0
  }

  return Math.max(1, Math.ceil(normalized.length / 4))
}

export function formatElapsed(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60

  if (minutes <= 0) {
    return `${seconds}s`
  }

  return `${minutes}m ${String(seconds).padStart(2, '0')}s`
}

export function getProgressIndicator(now: number): string {
  const frameIndex = Math.max(0, Math.floor(now / 1000)) % PROGRESS_FRAMES.length
  return PROGRESS_FRAMES[frameIndex] ?? PROGRESS_FRAMES[0]
}

function normalizePhase(
  streamStatus: string | null,
  currentMessage: string,
  hasPendingApproval: boolean,
  hasPendingQuestion = false,
): string {
  if (hasPendingApproval) {
    return 'approval pending'
  }
  if (hasPendingQuestion) {
    return 'answer pending'
  }

  if (streamStatus?.startsWith('Thinking:')) {
    return 'thinking'
  }

  if (streamStatus?.startsWith('Running ')) {
    return streamStatus
      .replace(/^Running\s+/, 'running ')
      .replace(/…$/, '')
  }

  if (
    currentMessage.trim().length > 0
    || streamStatus?.startsWith('Streaming')
  ) {
    return 'responding'
  }

  if (streamStatus) {
    return streamStatus
      .replace(/[.:…]+$/g, '')
      .trim()
      .toLowerCase()
  }

  return 'working'
}

export function buildStreamingProgress(
  input: StreamingProgressInput,
): StreamingProgress | null {
  if (!input.isStreaming || input.streamStartedAt == null) {
    return null
  }

  const elapsedMs = Math.max(0, input.now - input.streamStartedAt)
  const estimatedOutputTokens = Math.max(
    input.liveOutputTokens,
    estimateTokensFromText(input.currentMessage),
  )
  const tokensPerSecond = (
    elapsedMs >= 1000 && estimatedOutputTokens > 0
      ? estimatedOutputTokens / (elapsedMs / 1000)
      : null
  )
  const tokensPerSecondLabel = tokensPerSecond == null
    ? null
    : `~${tokensPerSecond.toFixed(1)} tok/s`
  const phase = normalizePhase(
    input.streamStatus,
    input.currentMessage,
    input.hasPendingApproval,
    input.hasPendingQuestion,
  )
  const elapsedLabel = formatElapsed(elapsedMs)
  const indicator = getProgressIndicator(input.now)

  return {
    indicator,
    phase,
    elapsedMs,
    elapsedLabel,
    estimatedOutputTokens,
    tokensPerSecond,
    tokensPerSecondLabel,
    label: [
      indicator,
      'in progress',
      phase,
      elapsedLabel,
      tokensPerSecondLabel,
    ].filter(Boolean).join(' • '),
  }
}
