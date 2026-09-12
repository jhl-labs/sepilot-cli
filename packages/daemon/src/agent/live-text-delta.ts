type LiveTextDeltaState = 'pending' | 'answer' | 'blocked'

export interface LiveTextDeltaEmitterOptions {
  /**
   * Prompt-ReAct providers often wrap final text in <final>...</final>.
   * When enabled, stream only the body of that final/answer tag and suppress
   * the closing tag if it arrives split across chunks.
   */
  allowTaggedFinal?: boolean
  /**
   * Some providers ignore the final-answer stem despite the system prompt.
   * When enabled, stream plain assistant text after it is no longer a possible
   * ANSWER:/INCOMPLETE: prefix instead of buffering the whole response.
   */
  allowUnmarkedFinal?: boolean
  /**
   * Optional guard for unmarked text. Prompt-ReAct can emit transport tags
   * such as <tool_call> or <think>; callers can reject those starts so live
   * streaming never exposes protocol/private text.
   */
  isAllowedUnmarkedFinalStart?: (text: string) => boolean
}

const LIVE_ANSWER_PROTOCOL_STEM_PATTERN = /^\s*(ANSWER|INCOMPLETE):/i
const TAGGED_FINAL_OPEN_PATTERN = /^\s*<(final|answer)>\s*/i

function matchLiveAnswerProtocolStem(text: string): {
  stem: 'answer' | 'incomplete'
  end: number
} | null {
  const match = LIVE_ANSWER_PROTOCOL_STEM_PATTERN.exec(text)
  if (!match?.[1]) {
    return null
  }

  return {
    stem: match[1].toUpperCase() === 'ANSWER' ? 'answer' : 'incomplete',
    end: match[0].length,
  }
}

function couldBecomeAnswerProtocolStem(text: string): boolean {
  const head = text.replace(/^\s+/, '').toUpperCase()
  if (head.length === 0) {
    return true
  }

  return 'ANSWER:'.startsWith(head) || 'INCOMPLETE:'.startsWith(head)
}

function couldBecomeTaggedFinalOpen(text: string): boolean {
  const head = text.replace(/^\s+/, '').toLowerCase()
  if (head.length === 0) {
    return true
  }

  return '<final>'.startsWith(head) || '<answer>'.startsWith(head)
}

function findClosingTagPrefixLength(text: string, closingTag: string): number {
  const lower = text.toLowerCase()
  const lowerClosing = closingTag.toLowerCase()
  const max = Math.min(lower.length, lowerClosing.length - 1)
  for (let length = max; length > 0; length -= 1) {
    if (lower.slice(-length) === lowerClosing.slice(0, length)) {
      return length
    }
  }
  return 0
}

export function createLiveTextDeltaEmitter(
  options: LiveTextDeltaEmitterOptions = {},
): {
  push: (chunk: string) => string
  flush: () => string
  hasEmitted: () => boolean
} {
  let state: LiveTextDeltaState = 'pending'
  let pending = ''
  let pendingBody = ''
  let trimLeadingBodyWhitespace = false
  let emitted = false
  let closingTag: string | null = null

  const emitBody = (body: string): string => {
    if (!body || state !== 'answer') {
      return ''
    }

    let next = body
    if (trimLeadingBodyWhitespace) {
      next = next.replace(/^\s+/, '')
      if (next.length === 0) {
        return ''
      }
      trimLeadingBodyWhitespace = false
    }

    if (!closingTag) {
      emitted = emitted || next.length > 0
      return next
    }

    pendingBody += next
    const lowerPending = pendingBody.toLowerCase()
    const lowerClosing = closingTag.toLowerCase()
    const closeIndex = lowerPending.indexOf(lowerClosing)
    if (closeIndex !== -1) {
      const out = pendingBody.slice(0, closeIndex)
      pendingBody = ''
      state = 'blocked'
      emitted = emitted || out.length > 0
      return out
    }

    const hold = findClosingTagPrefixLength(pendingBody, closingTag)
    const emitUntil = pendingBody.length - hold
    if (emitUntil <= 0) {
      return ''
    }

    const out = pendingBody.slice(0, emitUntil)
    pendingBody = pendingBody.slice(emitUntil)
    emitted = emitted || out.length > 0
    return out
  }

  const enterAnswer = (body: string, nextClosingTag: string | null): string => {
    state = 'answer'
    closingTag = nextClosingTag
    trimLeadingBodyWhitespace = true
    pending = ''
    return emitBody(body)
  }

  return {
    push(chunk: string): string {
      if (!chunk || state === 'blocked') {
        return ''
      }

      if (state === 'answer') {
        return emitBody(chunk)
      }

      pending += chunk
      const stem = matchLiveAnswerProtocolStem(pending)
      if (stem) {
        if (stem.stem === 'incomplete') {
          state = 'blocked'
          pending = ''
          return ''
        }

        return enterAnswer(pending.slice(stem.end), null)
      }

      if (options.allowTaggedFinal) {
        const tagMatch = TAGGED_FINAL_OPEN_PATTERN.exec(pending)
        if (tagMatch?.[1]) {
          const bodyStart = tagMatch[0].length
          const body = pending.slice(bodyStart)
          if (body.length === 0 || couldBecomeAnswerProtocolStem(body)) {
            return ''
          }

          const taggedStem = matchLiveAnswerProtocolStem(body)
          if (taggedStem) {
            if (taggedStem.stem === 'incomplete') {
              state = 'blocked'
              pending = ''
              return ''
            }
            return enterAnswer(
              body.slice(taggedStem.end),
              `</${tagMatch[1].toLowerCase()}>`,
            )
          }

          return enterAnswer(body, `</${tagMatch[1].toLowerCase()}>`)
        }
      }

      if (
        couldBecomeAnswerProtocolStem(pending)
        || (options.allowTaggedFinal && couldBecomeTaggedFinalOpen(pending))
      ) {
        return ''
      }

      if (options.allowUnmarkedFinal) {
        if (
          options.isAllowedUnmarkedFinalStart
          && !options.isAllowedUnmarkedFinalStart(pending)
        ) {
          state = 'blocked'
          pending = ''
          return ''
        }
        return enterAnswer(pending, null)
      }

      state = 'blocked'
      pending = ''
      return ''
    },
    flush() {
      if (state !== 'answer' || pendingBody.length === 0) {
        return ''
      }
      const out = pendingBody
      pendingBody = ''
      emitted = emitted || out.length > 0
      return out
    },
    hasEmitted() {
      return emitted
    },
  }
}
