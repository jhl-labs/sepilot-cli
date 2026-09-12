import type { ISessionStore, SessionEvent, SessionMeta } from '@sepilotd/core'

export interface SessionWatchQuestionDto {
  id: string
  prompt: string
  choices?: string[]
}

export type SessionWatchChange =
  | {
      type: 'created'
      session: SessionMeta
    }
  | {
      type: 'event'
      sessionId: string
      event: SessionEvent
    }
  | {
      type: 'deleted'
      sessionId: string
    }
  | {
      type: 'events_replaced'
      sessionId: string
      events: SessionEvent[]
    }
  | {
      type: 'metadata_updated'
      sessionId: string
      session: SessionMeta
    }
  | {
      type: 'question_requested'
      sessionId: string
      question: SessionWatchQuestionDto
    }
  | {
      type: 'question_answered'
      sessionId: string
      questionId: string
    }
  | {
      type: 'primary_agent_updated'
      sessionId: string
      agentId: string
    }
  | {
      type: 'steering_ack'
      sessionId: string
      noteId: string
      kind: 'instruction' | 'question'
      message: string
    }
  | {
      type: 'steering_consumed'
      sessionId: string
      noteId: string
    }
  | {
      type: 'steering_cancelled'
      sessionId: string
      noteId: string
    }
  | {
      type: 'run_status_changed'
      sessionId: string
      running: boolean
    }

type SessionWatchListener = (change: SessionWatchChange) => void

export class SessionWatchBroker {
  private readonly listeners = new Map<string, Set<SessionWatchListener>>()
  private readonly globalListeners = new Set<SessionWatchListener>()

  subscribe(
    sessionId: string,
    listener: SessionWatchListener,
  ): () => void {
    const current = this.listeners.get(sessionId) ?? new Set<SessionWatchListener>()
    current.add(listener)
    this.listeners.set(sessionId, current)

    return () => {
      const active = this.listeners.get(sessionId)
      if (!active) return
      active.delete(listener)
      if (active.size === 0) {
        this.listeners.delete(sessionId)
      }
    }
  }

  subscribeAll(listener: SessionWatchListener): () => void {
    this.globalListeners.add(listener)
    return () => {
      this.globalListeners.delete(listener)
    }
  }

  emit(change: SessionWatchChange): void {
    const sessionId =
      change.type === 'created'
        ? change.session.id
        : change.sessionId
    const listeners = this.listeners.get(sessionId)
    const allListeners = [
      ...this.globalListeners,
      ...(listeners ? Array.from(listeners) : []),
    ]
    if (allListeners.length === 0) {
      return
    }

    for (const listener of allListeners) {
      try {
        listener(change)
      } catch {
        // Watch listeners must not affect session persistence.
      }
    }
  }
}

export class WatchedSessionStore implements ISessionStore {
  replaceEvents?: (sessionId: string, events: SessionEvent[]) => Promise<void>
  updateMeta?: (
    sessionId: string,
    patch: {
      title?: string
      status?: SessionMeta['status']
      cwd?: string | null
      workspaceIsolation?: 'policy' | 'strict'
      provider?: string
      model?: string
      personaIds?: string[]
      preferPromptReact?: boolean
      starred?: boolean
      tags?: string[]
    },
  ) => Promise<SessionMeta | null>

  constructor(
    private readonly inner: ISessionStore,
    private readonly broker: SessionWatchBroker,
  ) {
    if (inner.replaceEvents) {
      this.replaceEvents = async (sessionId, events) => {
        await inner.replaceEvents!(sessionId, events)
        this.broker.emit({
          type: 'events_replaced',
          sessionId,
          events,
        })
      }
    }

    if (inner.updateMeta) {
      this.updateMeta = async (sessionId, patch) => {
        const session = await inner.updateMeta!(sessionId, patch)
        if (session) {
          this.broker.emit({
            type: 'metadata_updated',
            sessionId,
            session,
          })
        }
        return session
      }
    }
  }

  init?(): Promise<void> {
    return this.inner.init?.() ?? Promise.resolve()
  }

  create(
    meta: Omit<SessionMeta, 'messageCount' | 'totalTokens' | 'totalCost'>,
  ): Promise<SessionMeta> {
    return this.inner.create(meta).then((session) => {
      this.broker.emit({
        type: 'created',
        session,
      })
      return session
    })
  }

  get(id: string): Promise<SessionMeta | null> {
    return this.inner.get(id)
  }

  list(params?: { page?: number; perPage?: number; query?: string; workspaceRoot?: string }) {
    return this.inner.list(params)
  }

  delete(id: string): Promise<void> {
    return this.inner.delete(id).then(() => {
      this.broker.emit({
        type: 'deleted',
        sessionId: id,
      })
    })
  }

  appendEvent(sessionId: string, event: SessionEvent): Promise<void> {
    return this.inner.appendEvent(sessionId, event).then(() => {
      this.broker.emit({
        type: 'event',
        sessionId,
        event,
      })
    })
  }

  getEvents(sessionId: string): Promise<SessionEvent[]> {
    return this.inner.getEvents(sessionId)
  }
}
