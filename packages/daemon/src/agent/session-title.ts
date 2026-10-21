import type { ILLMProvider, ISessionStore, SessionMeta } from '@sepilotd/core'
import { generateSessionTitle } from './title-generator.js'
import { AuxiliaryLlmTurnBudget } from './auxiliary-llm.js'
import { createLogger } from '../logger.js'

const logger = createLogger('session-title')
const MAX_PENDING_TITLES = 4
const TITLE_BUDGET_MS = 25_000

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

type SessionTitleOptions = {
  sessions: ISessionStore
  session: SessionMeta
  provider: ILLMProvider
  model: string
  firstMessage: string
  assistantReply: string
  signal?: AbortSignal
  budget?: AuxiliaryLlmTurnBudget
}

export async function updateSessionTitleFromFirstTurn(options: SessionTitleOptions): Promise<SessionMeta> {
  const { sessions, session, provider, model, firstMessage, assistantReply } = options
  if (options.signal?.aborted || !sessions.updateTitleIfUnchanged || !shouldAutoTitleSession(session, firstMessage)) {
    return session
  }

  const expected = { title: session.title, createdAt: session.createdAt }
  const title = await generateSessionTitle(provider, model, firstMessage, assistantReply, options)
  if (options.signal?.aborted) return session
  return (await sessions.updateTitleIfUnchanged(session.id, title, expected)) ?? session
}

export class SessionTitleUpdates {
  private closed = false
  private readonly pending = new Set<AbortController>()
  private readonly sessions = new WeakMap<ISessionStore, Set<string>>()

  schedule(options: Omit<SessionTitleOptions, 'signal' | 'budget'>): void {
    if (this.closed || !options.sessions.updateTitleIfUnchanged
      || !shouldAutoTitleSession(options.session, options.firstMessage)
      || this.pending.size >= MAX_PENDING_TITLES) return
    const ids = this.sessions.get(options.sessions) ?? new Set<string>()
    if (ids.has(options.session.id)) return
    this.sessions.set(options.sessions, ids)
    ids.add(options.session.id)
    const controller = new AbortController()
    this.pending.add(controller)
    void updateSessionTitleFromFirstTurn({
      ...options,
      signal: controller.signal,
      budget: new AuxiliaryLlmTurnBudget(TITLE_BUDGET_MS),
    }).catch(() => {
      logger.warn('Optional title update failed', { sessionId: options.session.id })
    }).finally(() => {
      ids.delete(options.session.id)
      this.pending.delete(controller)
    })
  }

  close(): void {
    this.closed = true
    for (const controller of this.pending) controller.abort()
  }
}
