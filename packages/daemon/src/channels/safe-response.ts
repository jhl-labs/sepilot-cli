import {
  hasFinalAnswerStem,
  hasIncompleteAnswerStem,
  stripAnswerProtocolStem,
} from '../agent/interim-progress.js'

const INTERNAL_AGENT_FALLBACK_PATTERNS = [
  /^The run completed some tool work but the model returned an empty final reply\./i,
  /^The run ended with an empty final reply\./i,
  /^The run completed some tool work but ended with a progress-only reply instead of a finished answer\./i,
  /^The run ended with a progress-only reply instead of a finished answer\./i,
  /^Final reply missing after the run completed\./i,
]

const INTERNAL_AGENT_FALLBACK_PROGRESS_PATTERNS = [
  /^Latest progress update:\s*(.+)$/im,
  /\bLast progress update:\s*([^.\n]+(?:\.)?)/i,
]

function compactInternalFallbackLine(value: string): string {
  const normalized = value.trim().replace(/\s+/g, ' ')
  if (normalized.length <= 220) return normalized
  return `${normalized.slice(0, 219).trimEnd()}…`
}

function stripThinkingArtifactsForChannel(text: string): string {
  // Remove all well-formed <think>...</think> (or <thinking>...</thinking>) blocks.
  let remaining = text.replace(/<(think|thinking)>[\s\S]*?<\/\1>/gi, '')

  // Some models emit a closing </think> without an opening tag (the reasoning was
  // streamed before the wrapper was injected). Only strip such a leading close tag
  // when it is at the very start of the remaining text — never search-and-slice
  // mid-response, since a legitimate answer may quote `</think>` in code blocks,
  // documentation, or in a reply that echoes user-provided text.
  const trimmedLeading = remaining.trimStart()
  if (/^<\/(think|thinking)>/i.test(trimmedLeading)) {
    remaining = trimmedLeading.replace(/^<\/(?:think|thinking)>/i, '')
  }

  return remaining.trim()
}

function extractInternalFallbackProgress(responseText: string): string | undefined {
  for (const pattern of INTERNAL_AGENT_FALLBACK_PROGRESS_PATTERNS) {
    const match = responseText.match(pattern)
    const progress = match?.[1]?.trim()
    if (progress) return compactInternalFallbackLine(progress)
  }
  return undefined
}

function isInternalAgentFallbackResponse(responseText: string): boolean {
  const normalized = responseText.trim()
  return INTERNAL_AGENT_FALLBACK_PATTERNS.some((pattern) => pattern.test(normalized))
}

export function channelSafeAgentResponse(responseText: string): {
  text: string
  internalFallback: boolean
} {
  const visibleText = stripThinkingArtifactsForChannel(responseText)

  if (hasFinalAnswerStem(visibleText)) {
    return { text: stripAnswerProtocolStem(visibleText), internalFallback: false }
  }

  if (hasIncompleteAnswerStem(visibleText)) {
    const progress = compactInternalFallbackLine(stripAnswerProtocolStem(visibleText))
    return {
      internalFallback: true,
      text: [
        '요청을 완료하지 못했습니다.',
        progress ? `마지막 차단 사유: ${progress}` : undefined,
        '',
        '필요한 정보나 권한을 보완한 뒤 다시 요청해주세요.',
      ].filter((line): line is string => line !== undefined).join('\n'),
    }
  }

  if (!isInternalAgentFallbackResponse(visibleText)) {
    return { text: visibleText, internalFallback: false }
  }

  const progress = extractInternalFallbackProgress(visibleText)
  return {
    internalFallback: true,
    text: [
      '요청을 끝까지 답변으로 정리하지 못했습니다.',
      '조회나 도구 실행은 일부 진행됐지만, 모델이 사용자에게 보낼 최종 답변을 생성하지 못했습니다.',
      progress ? `마지막 진행 상황: ${progress}` : undefined,
      '',
      '같은 질문을 다시 보내면 새 실행으로 재시도합니다. 반복되면 /status로 모델/provider 상태를 확인해주세요.',
    ].filter((line): line is string => line !== undefined).join('\n'),
  }
}
