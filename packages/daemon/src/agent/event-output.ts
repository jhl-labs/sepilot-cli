import type { AgentEvent } from '@sepilotd/core'
import { INCOMPLETE_EXECUTION_CONTINUATION } from './response-safety.js'
import {
  hasAnyAnswerProtocolStem,
  isLikelyInterimProgressUpdate,
  stripAnswerProtocolStem,
  stripFinalAnswerStem,
} from './interim-progress.js'

export interface AgentOutputTracker {
  consume: (event: AgentEvent) => void
  finalContent: () => string
  providerFailureContent: (error: { code?: string; message?: string }) => string | null
  syntheticMessageEvent: () => Extract<AgentEvent, { type: 'message' }> | null
}

export function createAgentOutputTracker(): AgentOutputTracker {
  let explicitMessage = ''
  let streamedText = ''
  let lastSupersededProgress: string | null = null
  let lastToolOutcome:
    | { toolName: string; status: 'success' | 'error'; detail: string }
    | null = null
  const successfulToolOutcomes: Array<{ toolName: string; detail: string }> = []
  const toolNamesById = new Map<string, string>()

  const summarizeText = (content: string): string => {
    const normalized = content.trim().replace(/\s+/g, ' ')
    if (!normalized) {
      return ''
    }

    return normalized.length > 180
      ? `${normalized.slice(0, 179)}…`
      : normalized
  }

  const trackSupersededProgress = () => {
    const currentContent = explicitMessage || streamedText
    if (!isLikelyInterimProgressUpdate(currentContent)) {
      return
    }

    lastSupersededProgress = summarizeText(stripAnswerProtocolStem(currentContent))
    explicitMessage = ''
    streamedText = ''
  }

    const buildCompletionContent = (): string => {
      const rawCurrentContent = explicitMessage || streamedText
      const currentContent = stripFinalAnswerStem(rawCurrentContent)
      const trimmedCurrentContent = currentContent.trim()
      const hasProtocolStem = hasAnyAnswerProtocolStem(rawCurrentContent)

      if (
        trimmedCurrentContent
        && (hasProtocolStem || !isLikelyInterimProgressUpdate(rawCurrentContent))
      ) {
        return currentContent
      }

    const progressSummary = rawCurrentContent.trim()
      ? summarizeText(stripAnswerProtocolStem(rawCurrentContent))
      : lastSupersededProgress

    if (lastToolOutcome) {
      if (successfulToolOutcomes.length > 0) {
        const evidence = successfulToolOutcomes
          .slice(-3)
          .map((outcome) => `- ${outcome.toolName}: ${outcome.detail}`)
          .join('\n')
        const failedFollowUp = lastToolOutcome.status === 'error'
          ? [
              `Failed follow-up check: ${lastToolOutcome.toolName}.`,
              lastToolOutcome.detail && lastToolOutcome.detail !== 'No details'
                ? lastToolOutcome.detail
                : '',
            ].filter(Boolean).join(' ')
          : ''
        const progress = progressSummary ? `Last progress update: ${progressSummary}.` : ''
        return [
          'Final reply missing after the run completed.',
          'Successful tool evidence:',
          evidence,
          failedFollowUp,
          progress,
        ].filter(Boolean).join('\n')
      }
      const outcomeLabel = lastToolOutcome.status === 'success'
        ? 'completed successfully'
        : 'completed with an error'
      const progress = progressSummary
        ? ` Last progress update: ${progressSummary}.`
        : ''
      const detail = lastToolOutcome.detail && lastToolOutcome.detail !== 'No details'
        ? ` Last tool output: ${lastToolOutcome.detail}`
        : ''
      return `Final reply missing after the run completed. ${lastToolOutcome.toolName} ${outcomeLabel}.${progress}${detail}`
    }

    if (progressSummary) {
      return `Final reply missing after the run completed. Last progress update: ${progressSummary}.`
    }

    return currentContent
  }

  const finalContent = (): string => {
    return buildCompletionContent()
  }

  const providerFailureContent = (error: { code?: string; message?: string }): string | null => {
    if (successfulToolOutcomes.length === 0) {
      return null
    }

    const evidence = successfulToolOutcomes
      .slice(-3)
      .map((outcome) => `- ${outcome.toolName}: ${outcome.detail}`)
      .join('\n')
    const failureIdentity = [error.code?.trim(), summarizeText(error.message ?? '')]
      .filter(Boolean)
      .join(': ')

    return [
      'INCOMPLETE: The model provider failed before a final answer could be synthesized.',
      failureIdentity ? `Provider failure: ${failureIdentity}` : 'Provider failure: unknown provider error',
      'Retained successful tool evidence:',
      evidence,
      INCOMPLETE_EXECUTION_CONTINUATION,
    ].join('\n')
  }

  return {
    consume(event) {
      if (event.type === 'text_delta') {
        streamedText += event.text ?? ''
        return
      }

      if (event.type === 'message') {
        explicitMessage = event.content ?? ''
        return
      }

      if (event.type === 'state_change' && (event.state === 'thinking' || event.state === 'acting')) {
        trackSupersededProgress()
        return
      }

      if (event.type === 'tool_call') {
        toolNamesById.set(event.toolCall.id, event.toolCall.name)
        trackSupersededProgress()
        return
      }

      if (event.type === 'approval_request') {
        toolNamesById.set(event.toolCall.id, event.toolCall.name)
        trackSupersededProgress()
        return
      }

      if (event.type === 'tool_result') {
        lastToolOutcome = {
          toolName: toolNamesById.get(event.toolCallId) ?? 'Tool',
          status: event.status === 'success' ? 'success' : 'error',
          detail: summarizeText(event.output),
        }
        if (lastToolOutcome.status === 'success') {
          successfulToolOutcomes.push({
            toolName: lastToolOutcome.toolName,
            detail: lastToolOutcome.detail,
          })
        }
      }
    },
    finalContent,
    providerFailureContent,
    syntheticMessageEvent() {
      const content = finalContent()
      if (!content) {
        return null
      }

      const normalizedExplicitMessage = stripFinalAnswerStem(explicitMessage)
      if (
        content === normalizedExplicitMessage
        && (
          hasAnyAnswerProtocolStem(explicitMessage)
          || !isLikelyInterimProgressUpdate(explicitMessage)
        )
      ) {
        return null
      }

      return {
        type: 'message',
        content,
      }
    },
  }
}
