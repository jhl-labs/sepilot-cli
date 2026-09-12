import type { ILLMProvider, ISessionStore, SessionMeta } from '@sepilotd/core'
import { generateSessionTitle } from './title-generator.js'

const DEFAULT_SESSION_TITLES = new Set([
  '새 대화',
  'New Chat',
  'New Conversation',
  'Untitled session',
  '제목 없는 세션',
])

export function fallbackSessionTitle(message: string): string {
  const normalized = message.trim().replace(/\s+/g, ' ')
  const firstSentence = normalized.match(/^[^.!?\n]+[.!?]?/)?.[0] ?? normalized
  if (firstSentence.length <= 50) return firstSentence || '새 대화'
  return `${firstSentence.slice(0, 47)}...`
}

export function shouldAutoTitleSession(session: SessionMeta, firstMessage: string): boolean {
  if (session.messageCount > 0) return false
  const title = session.title.trim()
  if (!title || DEFAULT_SESSION_TITLES.has(title)) return true
  return title === fallbackSessionTitle(firstMessage)
}

export async function updateSessionTitleFromFirstTurn(options: {
  sessions: ISessionStore
  session: SessionMeta
  provider: ILLMProvider
  model: string
  firstMessage: string
  assistantReply: string
}): Promise<SessionMeta> {
  const { sessions, session, provider, model, firstMessage, assistantReply } = options
  if (!sessions.updateMeta || !shouldAutoTitleSession(session, firstMessage)) {
    return session
  }

  const title = await generateSessionTitle(provider, model, firstMessage, assistantReply)
  return (await sessions.updateMeta(session.id, { title })) ?? session
}
