import type { Message } from '@sepilotd/core'
import {
  CURRENT_AGENT_TURN_INSTRUCTION_METADATA_KEY,
  CURRENT_AGENT_TURN_USER_METADATA_KEY,
} from './turn-context.js'

export { CURRENT_AGENT_TURN_USER_METADATA_KEY } from './turn-context.js'

export const MEMORY_REMEMBER_TOOL_NAME = 'memory.remember'

export const TOOL_RESULT_STATUS_METADATA_KEY = 'toolResultStatus'
const MEMORY_WRITE_RECOVERY_REMINDER_KIND = 'memory_write_completion'
export type MemoryWriteOutcome = 'success' | 'failed' | 'missing'

type ToolCallHistoryEntry = {
  tool: string
  input?: Record<string, unknown>
  status: 'success' | 'error'
}

// This is deliberately an intent classifier only. It never extracts a memory
// payload and never writes anything itself: the model must still choose safe,
// durable content and call memory.remember through the normal policy/tool path.
const EXPLICIT_MEMORY_WRITE_PATTERNS = [
  /(?:^|[.!?]\s+|\band\s+)(?:please\s+)?remember\s+(?!to\b)(?:that\b|this\b|these\b|those\b|it\b|my\b|the\b)/iu,
  /\b(?:save|store|record|commit|add)\b[\s\S]{0,120}\b(?:in|to)\s+(?:your\s+)?(?:long[-\s]?term\s+)?memory\b/iu,
  /기억\s*(?:해|하여)\s*(?:줘|주세요|주십시오|둬|두세요|놓아\s*줘|놓으세요)(?=\s*(?:$|[.!?,]|그리고|또))/u,
  /(?:메모리|장기\s*기억)(?:에|로)\s*[\s\S]{0,80}?(?:저장|기록|등록|추가|남겨)(?:해\s*주세요|해\s*줘|해\s*두|해주세요|해둬|하여|주세요|줘|해)?(?=\s*(?:$|[.!?,]|그리고|또))/u,
  /(?:请记住|請記住)(?=[^。！？\n]*(?:$|[。！？]))/u,
  /覚えて(?:おいて|ください)(?=\s*(?:$|[。！？]))/u,
]

const NEGATED_MEMORY_WRITE_PATTERNS = [
  /\b(?:do\s+not|don't|never)\s+(?:remember|save|store|record)\b/iu,
  /(?:기억|저장|기록)(?:하|해)?지\s*마/u,
  /(?:记住|記住|覚えて)[^.!?\n]{0,12}(?:不要|しないで)/u,
]

// Mentions of a memory command are not themselves memory commands. These
// patterns intentionally prefer a false negative when a user is asking what a
// phrase means or reporting what happened after they typed it. The model may
// still choose memory.remember normally; only the mandatory completion guard
// is disabled for ambiguous meta-language.
const MEMORY_WRITE_META_MENTION_PATTERNS = [
  /(?:기억\s*(?:해|하여)\s*(?:줘|주세요|주십시오|둬|두세요)|(?:메모리|장기\s*기억)(?:에|로)[^.!?。！？\n]{0,80}(?:저장|기록|등록|추가|남겨))[.!?。！？]\s*(?:라고|하고)\s*(?:했|말|입력|요청|썼|보냈|하니|했더니)/u,
  /기억\s*(?:해|하여)\s*(?:줘|주세요|주십시오|둬|두세요)\s*\?\s*(?:무슨|어떤)\s*(?:뜻|의미)|기억해줘\s*\?\s*(?:뭐|왜|어떻게)/u,
  /\b(?:please\s+)?remember\b[^.!?\n]{0,120}[.!?]\s*(?:(?:is|was)\s+what\s+)?(?:I|we|the user|a user)\s+(?:said|typed|wrote|asked|entered)\b/iu,
  /\b(?:please\s+)?remember\b[^.!?\n]{0,120}[.!?]\s*(?:but|and)\s+(?:it|that|the request)\s+(?:did\s+not|didn't|wasn't)\s+(?:save|saved|remembered|stored)\b/iu,
  /\b(?:what\s+does|what\s+is\s+the\s+meaning\s+of|explain\s+the\s+phrase)\b[^.!?\n]{0,100}\bremember\b/iu,
  /\bremember\s+(?:this|that|it)\s+(?:means?|meaning|refers?\s+to)\b/iu,
  /(?:请记住|請記住)[^。！？\n]{0,24}(?:是什么意思|是什麼意思|什么意思|什麼意思|的意思(?:是|呢|吗|嗎)?)/u,
  /(?:请记住|請記住)[^。！？\n]{0,60}[。！？]\s*(?:(?:是我|我|用户|使用者)?(?:说|說|输入|輸入|写|寫|请求|請求))/u,
  /覚えて(?:おいて|ください)[^。！？\n]{0,24}(?:とは|って|の)?\s*(?:どういう意味|何を意味|意味(?:は|です|だ))/u,
  /覚えて(?:おいて|ください)[。！？]\s*(?:と\s*)?(?:言|入力|書|頼)(?:った|いました|いた|くと|んだ)/u,
]

export function isExplicitMemoryWriteRequest(input: string): boolean {
  const normalized = input.normalize('NFKC').trim()
  if (!normalized) return false
  // A bug report or explanation often quotes the exact phrase that triggered
  // the behavior (for example: `"기억해줘"라고 하니...`). Quoted/code text is
  // mention data, not an instruction to the current agent, so exclude it from
  // this conservative completion guard.
  const directiveText = normalized
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`[^`]*`/g, ' ')
    .replace(/"[^"\r\n]*"/g, ' ')
    .replace(/'[^'\r\n]*'/g, ' ')
    .replace(/“[^”]*”|‘[^’]*’|「[^」]*」|『[^』]*』|《[^》]*》/gu, ' ')
    .trim()
  if (!directiveText) return false
  if (NEGATED_MEMORY_WRITE_PATTERNS.some((pattern) => pattern.test(directiveText))) {
    return false
  }
  if (MEMORY_WRITE_META_MENTION_PATTERNS.some((pattern) => pattern.test(directiveText))) {
    return false
  }
  return EXPLICIT_MEMORY_WRITE_PATTERNS.some((pattern) => pattern.test(directiveText))
}

export function latestUserText(messages: readonly Message[]): string {
  const index = currentTurnUserMessageIndex(messages)
  const message = index >= 0 ? messages[index] : undefined
  if (!message) return ''
  const structuredInstruction = message.metadata?.[CURRENT_AGENT_TURN_INSTRUCTION_METADATA_KEY]
  if (typeof structuredInstruction === 'string') return structuredInstruction
  if (typeof message.content === 'string') return message.content
  return message.content
    .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

export function memoryWriteOutcomeFromHistory(
  history: readonly ToolCallHistoryEntry[] | undefined,
): MemoryWriteOutcome {
  const attempts = (history ?? []).filter((entry) => entry.tool === MEMORY_REMEMBER_TOOL_NAME)
  if (attempts.length === 0) return 'missing'
  const latestStatusByPayload = new Map<string, 'success' | 'error'>()
  attempts.forEach((entry, index) => {
    latestStatusByPayload.set(memoryWritePayloadKey(entry.input, `history:${index}`), entry.status)
  })
  return [...latestStatusByPayload.values()].every((status) => status === 'success')
    ? 'success'
    : 'failed'
}

export function memoryWriteOutcomeFromMessages(
  messages: readonly Message[],
): MemoryWriteOutcome {
  const currentTurnMessages = messagesAfterLatestUser(messages)
  const inputByToolCallId = new Map<string, Record<string, unknown>>()
  for (const message of currentTurnMessages) {
    if (message.role !== 'assistant') continue
    for (const toolCall of message.toolCalls ?? []) {
      if (toolCall.name === MEMORY_REMEMBER_TOOL_NAME) {
        inputByToolCallId.set(toolCall.id, toolCall.arguments ?? {})
      }
    }
  }
  const latestStatusByPayload = new Map<string, 'success' | 'error'>()
  for (const message of currentTurnMessages) {
    if (message.role !== 'tool' || message.name !== MEMORY_REMEMBER_TOOL_NAME) continue
    const fallbackKey = `tool-call:${message.toolCallId ?? latestStatusByPayload.size}`
    const key = memoryWritePayloadKey(
      message.toolCallId ? inputByToolCallId.get(message.toolCallId) : undefined,
      fallbackKey,
    )
    latestStatusByPayload.set(
      key,
      message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] === 'success' ? 'success' : 'error',
    )
  }
  if (latestStatusByPayload.size === 0) return 'missing'
  return [...latestStatusByPayload.values()].every((status) => status === 'success')
    ? 'success'
    : 'failed'
}

function memoryWritePayloadKey(
  input: Record<string, unknown> | undefined,
  fallback: string,
): string {
  const content = input?.content
  return typeof content === 'string' && content.trim()
    ? `content:${content.trim()}`
    : fallback
}

/**
 * Persist structured result evidence on the matching tool message. This keeps
 * the completion decision robust across run checkpoints without parsing a
 * human-readable tool output.
 */
export function recordToolResultStatus(
  messages: Message[],
  toolCallId: string,
  status: 'success' | 'error',
): void {
  const message = [...messages].reverse().find((candidate) => (
    candidate.role === 'tool' && candidate.toolCallId === toolCallId
  ))
  if (!message) return
  message.metadata = {
    ...message.metadata,
    [TOOL_RESULT_STATUS_METADATA_KEY]: status,
  }
}

export function buildMemoryWriteRecoveryMessage(): Message {
  return {
    role: 'system',
    metadata: { reminderKind: MEMORY_WRITE_RECOVERY_REMINDER_KIND },
    content: [
      '[Memory write completion guard]',
      'The user explicitly requested a durable memory write, but no successful memory.remember result exists in this run.',
      'The previous assistant draft was withheld from the user. After the tool succeeds, answer the original request in full again (including any requested explanation or other non-memory work); do not return only a save acknowledgement.',
      'Call memory.remember now through the normal tool path. Semantically select only the durable fact or preference the user asked to retain; do not copy the request wrapper and do not store secrets or volatile data.',
      'Do not claim that anything was remembered or saved unless memory.remember returns success. If the tool cannot be called or fails, report INCOMPLETE and say the memory was not confirmed saved.',
    ].join('\n'),
  }
}

export function countMemoryWriteRecoveryPrompts(messages: readonly Message[]): number {
  return messagesAfterLatestUser(messages).filter((message) => (
    message.role === 'system'
    && message.metadata?.reminderKind === MEMORY_WRITE_RECOVERY_REMINDER_KIND
  )).length
}

function messagesAfterLatestUser(messages: readonly Message[]): readonly Message[] {
  const latestUserIndex = currentTurnUserMessageIndex(messages)
  return latestUserIndex >= 0 ? messages.slice(latestUserIndex + 1) : messages
}

function currentTurnUserMessageIndex(messages: readonly Message[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (
      message?.role === 'user'
      && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true
    ) {
      return index
    }
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') {
      return index
    }
  }
  return -1
}

export interface MemoryWriteFailureOutputOptions {
  candidate?: string
  userInput?: string
}

const MEMORY_WRITE_SUCCESS_CLAIM_PATTERNS: Array<{
  pattern: RegExp
  replacement?: string
}> = [
  { pattern: /(?:네|예|알겠습니다|좋아요)?\s*[,;:]?\s*(?:(?:이|그|해당)\s*)?(?:(?:내용|설명|정보|사실|선호|용어)(?:은|는|을|를|도)?\s*)?(?:기억|저장|기록)(?:해\s*(?:두었|뒀|둘게|놓았)|했|하였|되었|됐|완료했|완료됐|하겠)(?:습니다|어요|요)?/giu },
  { pattern: /(?:저장|기억)(?:을|를|도)?\s*(?:완료|확인)(?:했|하였|됐|되었)(?:습니다|어요)?/gu },
  { pattern: /\bI(?:'ve|\s+have)?\s+(?:now\s+|successfully\s+)?(?:remembered|saved|stored|recorded)\b(?:\s+(?:it|this|that|the\s+(?:information|explanation|preference|term)))?(?:\s+(?:for\s+later|in\s+(?:my\s+)?(?:long[- ]term\s+)?memory))?/giu },
  { pattern: /\b(?:it|this|that|the\s+(?:information|explanation|preference|term))\s+(?:has\s+been|was|is\s+now)\s+(?:remembered|saved|stored|recorded)\b(?:\s+(?:for\s+later|in\s+memory))?/giu },
  {
    pattern: /(^|[.!?]\s+)(?:yes[,;:]?\s+|okay[,;:]?\s+|successfully\s+|now\s+)?(?:remembered|saved|stored|recorded)\s+(?:it|this|that|to\s+memory|in\s+memory)\b/gimu,
    replacement: '$1',
  },
  { pattern: /(?:我)?(?:已经|已)?(?:记住|記住|保存|存储|儲存|记录|記錄)(?:了|好了|成功)?/gu },
  { pattern: /(?:覚えました|記憶しました|保存しました|記録しました|覚えておきました)/gu },
]

const NON_MEMORY_RESPONSE_WORK_PATTERNS = [
  /(?:설명|정리|요약|분석|비교|계산|번역)(?:해|하|해서|하고|한\s*(?:뒤|다음)|하여)/u,
  /(?:알려|답변|작성|고쳐|수정|찾아|검색)(?:줘|주세요|주십시오|해서|하고|한\s*(?:뒤|다음))/u,
  /\b(?:explain|summarize|analyse|analyze|compare|calculate|translate|answer|write|draft|fix|revise|find|search|list)\b/iu,
  /(?:解释|說明|说明|总结|總結|分析|比较|比較|计算|計算|翻译|翻譯|回答|查找|搜索)/u,
  /(?:説明|要約|整理|分析|比較|計算|翻訳|回答|作成|検索)(?:して|し、|した)/u,
]

/**
 * A failed pure memory command has no trustworthy prose result to preserve.
 * Keep a draft only when the user separately requested user-facing work such
 * as an explanation or summary; this prevents a hallucinated save promise
 * from surviving merely because it used indirect wording.
 */
function requestsNonMemoryResponseWork(input: string): boolean {
  return NON_MEMORY_RESPONSE_WORK_PATTERNS.some((pattern) => pattern.test(input.normalize('NFKC')))
}

function stripUnconfirmedMemorySuccessClaims(candidate: string): string {
  let sanitized = candidate
    .trim()
    .replace(/^(?:ANSWER|FINAL(?:\s+ANSWER)?)\s*:\s*/gimu, '')
    .replace(/^.*(?:저장|기억)(?:된|해\s*둔|한)\s*(?:내용|정보|설명|사실|선호)[^\n]*(?:바탕|기반|활용|참고)[^\n]*(?:\n|$)/gimu, '')
    .replace(/^.*\bbased\s+on\s+(?:the\s+)?(?:saved|remembered|stored|recorded)\s+(?:content|information|explanation|fact|preference)[^\n]*(?:\n|$)/gimu, '')
  for (const { pattern, replacement = '' } of MEMORY_WRITE_SUCCESS_CLAIM_PATTERNS) {
    sanitized = sanitized.replace(pattern, replacement)
  }
  return sanitized
    .replace(/\s*(?:그리고|또한|또|and|also)\s*[,;:]?\s*(?=[.!?。！？]|$)/giu, '')
    .replace(/(^|[\n.!?。！？]\s*)(?:네|예|알겠습니다|좋아요|yes|okay|ok)\s*[,;:]?\s*(?=$|[\n.!?。！？])/giu, '$1')
    .replace(/([.!?。！？])\s*[.!?。！？]+/g, '$1')
    .replace(/(^|\n)\s*[,;:.!?。！？]+\s*(?=\n|$)/g, '$1')
    .replace(/[ \t]+([,.;:!?。！？])/g, '$1')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

function messageText(message: Message): string {
  if (typeof message.content === 'string') return message.content
  return message.content
    .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

function isMemoryWriteAcknowledgementOnly(candidate: string): boolean {
  const residual = stripUnconfirmedMemorySuccessClaims(candidate)
    .replace(/[\s,.;:!?。！？]+/g, ' ')
    .trim()
  if (!residual) return true
  return /^(?:네|예|알겠습니다|좋아요|완료|처리 완료|yes|okay|ok|sure|done|completed|好的|好|完成|了解|はい|わかりました)$/iu
    .test(residual)
}

/**
 * Text emitted alongside the tool call is buffered while the write is still
 * unverified. If the post-tool response is only a save acknowledgement, put
 * the substantive buffered answer back so compound requests (for example,
 * "explain this and remember it") do not lose their explanation.
 */
export function restoreSuppressedMemoryWriteDraft(
  messages: readonly Message[],
  candidate: string,
): string {
  if (!isMemoryWriteAcknowledgementOnly(candidate)) return candidate
  const drafts = messagesAfterLatestUser(messages)
    .filter((message) => (
      message.role === 'assistant'
      && message.toolCalls?.some((toolCall) => toolCall.name === MEMORY_REMEMBER_TOOL_NAME)
    ))
    .map((message) => stripUnconfirmedMemorySuccessClaims(messageText(message)))
    .filter(Boolean)
    .sort((left, right) => right.length - left.length)
  const draft = drafts[0]
  if (!draft || candidate.includes(draft)) return candidate
  return `${draft}\n\n${candidate.trim()}`.trim()
}

type FailureLanguage = 'ko' | 'en' | 'zh' | 'ja'

function memoryWriteFailureLanguage(input: string): FailureLanguage {
  if (/[가-힣]/u.test(input)) return 'ko'
  if (/[ぁ-ゟ゠-ヿ]/u.test(input)) return 'ja'
  if (/\p{Script=Han}/u.test(input)) return 'zh'
  return 'en'
}

export function buildMemoryWriteFailureOutput(
  outcome: Exclude<MemoryWriteOutcome, 'success'>,
  options: MemoryWriteFailureOutputOptions = {},
): string {
  const userInput = options.userInput ?? ''
  const candidate = requestsNonMemoryResponseWork(userInput)
    ? stripUnconfirmedMemorySuccessClaims(options.candidate ?? '')
    : ''
  const language = memoryWriteFailureLanguage(userInput)
  const notice = (() => {
    switch (language) {
      case 'ko':
        return outcome === 'failed'
          ? 'INCOMPLETE: 기억 저장 중 하나 이상이 실패하여 요청한 기억 전체가 저장되었다고 확인할 수 없습니다.'
          : 'INCOMPLETE: 기억 저장 도구를 사용할 수 없었거나 성공적으로 호출하지 못해 요청한 기억이 저장되었다고 확인할 수 없습니다.'
      case 'zh':
        return outcome === 'failed'
          ? 'INCOMPLETE: 至少有一次记忆写入失败，因此无法确认请求的全部内容都已保存。'
          : 'INCOMPLETE: 无法使用或成功调用记忆工具，因此无法确认请求的内容已保存。'
      case 'ja':
        return outcome === 'failed'
          ? 'INCOMPLETE: 1件以上の記憶保存に失敗したため、依頼された内容がすべて保存されたとは確認できません。'
          : 'INCOMPLETE: 記憶ツールを利用または正常に呼び出せなかったため、依頼された内容が保存されたとは確認できません。'
      default:
        return outcome === 'failed'
          ? 'INCOMPLETE: At least one memory write failed, so the full requested memory was not confirmed saved.'
          : 'INCOMPLETE: The memory tool was unavailable or was not called successfully, so the requested memory was not confirmed saved.'
    }
  })()
  return candidate ? `${notice}\n\n${candidate}` : notice
}
