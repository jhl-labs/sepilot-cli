import type { ModelInfo, ModelToolTransport } from '@sepilotd/core'

export type ModelToolRunner = 'react' | 'graph'

export interface ResolvedModelToolTransport {
  configured: ModelToolTransport
  initial: 'native' | 'prompt-react'
  adaptiveFallback: boolean
}

const FINAL_ANSWER_STEM = 'ANSWER:'
const FINAL_TRANSPORT_OPEN_PATTERN = /^(\s*(?:<(?:final|answer)>\s*)?)([\s\S]*)$/i

/**
 * Repair only the structural response shape declared by an evidence-scoped
 * provider/model compatibility profile. The endpoint may remove any left
 * prefix of `ANSWER:` while keeping a suffix such as `WER:` or just `:`.
 * Complete stems, ordinary prose, and models without the override are left
 * byte-for-byte unchanged.
 */
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
  // Backward-compatible interpretation for configs written before the
  // compatibility profile existed. An explicit non-auto profile above wins.
  if (model?.capabilities.adaptivePromptReact === true) return 'adaptive'
  return configured ?? 'auto'
}

/**
 * Resolve transport policy from declared capability + operator compatibility.
 * There are deliberately no provider or model-name branches here.
 *
 * `toolUse: false` is a hard capability ceiling. A compatibility preference
 * can choose a safer protocol but cannot invent native support. The adaptive
 * graph runner starts native and switches on observed protocol anomalies; the
 * simpler react runner starts directly in prompt-react because it has no
 * mid-run transport switch.
 */
export function resolveModelToolTransport(
  model: ModelInfo | undefined,
  _runner: ModelToolRunner,
): ResolvedModelToolTransport {
  const configured = configuredToolTransport(model)
  if (model?.capabilities.toolUse === false || configured === 'prompt-react') {
    return { configured, initial: 'prompt-react', adaptiveFallback: false }
  }
  if (configured === 'adaptive') {
    // Adaptive declares an observed-failure fallback, not a disabled
    // transport. Starting prompt-react eagerly is unsafe for
    // OpenAI-compatible servers whose chat template consumes tool-envelope
    // sentinel tokens: the visible response then comes back empty even though
    // native function calling works. Both runners therefore start native and
    // switch only on structurally unusable native evidence.
    return { configured, initial: 'native', adaptiveFallback: true }
  }
  return { configured, initial: 'native', adaptiveFallback: false }
}
