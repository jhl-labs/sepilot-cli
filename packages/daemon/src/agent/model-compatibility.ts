import { ThinkingLevel, type ModelInfo, type ModelToolTransport } from '@sepilotd/core'

export type ModelToolRunner = 'react' | 'graph'

export interface ResolvedModelToolTransport {
  configured: ModelToolTransport
  initial: 'native' | 'prompt-react'
  adaptiveFallback: boolean
}

const FINAL_ANSWER_STEM = 'ANSWER:'
const FINAL_TRANSPORT_OPEN_PATTERN = /^(\s*(?:<(?:final|answer)>\s*)?)([\s\S]*)$/i

export function resolveRepairThinkingLevel(
  model: ModelInfo | undefined,
  userThinkingLevel: string | undefined,
): ThinkingLevel {
  if (userThinkingLevel === ThinkingLevel.Off || !model?.capabilities.thinking) {
    return ThinkingLevel.Off
  }
  if ([ThinkingLevel.Low, ThinkingLevel.Medium, ThinkingLevel.High, ThinkingLevel.Max]
    .includes(userThinkingLevel as ThinkingLevel)) return userThinkingLevel as ThinkingLevel
  return (model.compatibility?.repairThinkingOffFallback as ThinkingLevel | undefined) ?? ThinkingLevel.Off
}

export function normalizeModelAnswerProtocol(
  content: string,
  model: ModelInfo | undefined,
): string {
  if (model?.compatibility?.answerProtocol !== 'repair-left-truncated-answer-stem') {
    return content
  }

  const transport = FINAL_TRANSPORT_OPEN_PATTERN.exec(content)
  if (!transport) return content
  const leading = transport[1] ?? ''
  const body = transport[2] ?? ''
  const upperBody = body.toUpperCase()
  if (upperBody.startsWith(FINAL_ANSWER_STEM) || upperBody.startsWith('INCOMPLETE:')) {
    return content
  }

  for (let offset = 1; offset < FINAL_ANSWER_STEM.length; offset += 1) {
    const suffix = FINAL_ANSWER_STEM.slice(offset)
    if (upperBody.startsWith(suffix)) {
      return `${leading}${FINAL_ANSWER_STEM}${body.slice(suffix.length)}`
    }
  }

  return content
}

function configuredToolTransport(model: ModelInfo | undefined): ModelToolTransport {
  const configured = model?.compatibility?.toolTransport
  if (configured && configured !== 'auto') return configured
  if (model?.capabilities.adaptivePromptReact === true) return 'adaptive'
  return configured ?? 'auto'
}

export function resolveModelToolTransport(
  model: ModelInfo | undefined,
  _runner: ModelToolRunner,
): ResolvedModelToolTransport {
  const configured = configuredToolTransport(model)
  if (model?.capabilities.toolUse === false || configured === 'prompt-react') {
    return { configured, initial: 'prompt-react', adaptiveFallback: false }
  }
  if (configured === 'adaptive') {
    return { configured, initial: 'native', adaptiveFallback: true }
  }
  return { configured, initial: 'native', adaptiveFallback: false }
}
