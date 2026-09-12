import { randomUUID } from 'node:crypto'

export interface AgentMessage {
  id?: string
  from: string
  to?: string
  type: 'task' | 'result' | 'status_update' | 'question' | 'feedback'
  payload: Record<string, unknown>
  timestamp?: string
}

type MessageHandler = (message: AgentMessage) => void

export class AgentMessageBus {
  private readonly subscribers = new Map<string, MessageHandler>()
  private readonly history = new Map<string, AgentMessage[]>()

  subscribe(agentId: string, handler: MessageHandler): () => void {
    this.subscribers.set(agentId, handler)
    if (!this.history.has(agentId)) {
      this.history.set(agentId, [])
    }
    return () => {
      this.subscribers.delete(agentId)
    }
  }

  async send(message: AgentMessage): Promise<void> {
    const nextMessage: AgentMessage = {
      ...message,
      id: message.id ?? randomUUID(),
      timestamp: message.timestamp ?? new Date().toISOString(),
    }

    if (!nextMessage.to) return

    const history = this.history.get(nextMessage.to)
    if (history) {
      history.push(nextMessage)
    }

    const handler = this.subscribers.get(nextMessage.to)
    if (handler) {
      handler(nextMessage)
    }
  }

  async broadcast(message: Omit<AgentMessage, 'to'>): Promise<void> {
    const baseMessage: AgentMessage = {
      ...message,
      id: message.id ?? randomUUID(),
      timestamp: message.timestamp ?? new Date().toISOString(),
    }

    for (const [agentId, handler] of this.subscribers) {
      if (agentId === baseMessage.from) continue

      const nextMessage: AgentMessage = {
        ...baseMessage,
        to: agentId,
      }

      this.history.get(agentId)?.push(nextMessage)
      handler(nextMessage)
    }
  }

  getHistory(agentId: string): AgentMessage[] {
    return this.history.get(agentId) ?? []
  }

  clear(): void {
    this.history.clear()
  }
}
