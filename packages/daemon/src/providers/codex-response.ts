import type { JsonRpcMessage } from '../acp/rpc.js'

export class CodexResponseObservation {
  private finalText = ''
  private fallbackText = ''
  private readonly sealedItems = new Map<string, string>()
  private readonly sealedSnapshot = new Map<string, string>()
  private readonly counts = {
    notificationCount: 0,
    completedItems: 0,
    completedMessages: 0,
    completedMessageChars: 0,
    snapshotItems: 0,
    snapshotMessages: 0,
    deltaCount: 0,
    deltaChars: 0,
    stderrChunks: 0,
    stderrBytes: 0,
    processExited: false,
  }

  observe(message: JsonRpcMessage): void {
    this.counts.notificationCount++
    const params = message.params && typeof message.params === 'object'
      ? message.params as Record<string, unknown>
      : null
    if (message.method === 'item/completed') this.collect(params?.item, false)
    if (message.method === 'item/agentMessage/delta' && typeof params?.delta === 'string') {
      this.counts.deltaCount++
      this.counts.deltaChars += params.delta.length
    }
    if (message.method === 'turn/completed') {
      const turn = params?.turn
      if (turn && typeof turn === 'object' && Array.isArray((turn as { items?: unknown }).items)) {
        for (const item of (turn as { items: unknown[] }).items) this.collect(item, true)
      }
    }
  }

  observeStderr(chunk: Buffer): void {
    this.counts.stderrChunks++
    this.counts.stderrBytes += chunk.length
  }

  observeExit(): void {
    this.counts.processExited = true
  }

  get finalAnswer(): string {
    return (this.finalText || this.fallbackText).trim()
  }

  get transcript(): string {
    const messages = [...this.sealedSnapshot.values()]
    for (const [key, text] of this.sealedItems) {
      if (!this.sealedSnapshot.has(key)) messages.push(text)
    }
    return messages.map(text => text.trim()).join('\n\n').trim()
  }

  get sealedMessageCount(): number {
    return new Set([...this.sealedSnapshot.keys(), ...this.sealedItems.keys()]).size
  }

  diagnostics(): Readonly<typeof this.counts> {
    return { ...this.counts }
  }

  private collect(item: unknown, snapshot: boolean): void {
    if (snapshot) this.counts.snapshotItems++
    else this.counts.completedItems++
    if (!item || typeof item !== 'object') return
    const record = item as Record<string, unknown>
    if (record.type !== 'agentMessage' || typeof record.text !== 'string') return
    if (snapshot) this.counts.snapshotMessages++
    else this.counts.completedMessages++
    this.counts.completedMessageChars += record.text.length
    if (!record.text.trim()) return
    this.fallbackText = record.text
    if (record.phase === 'final_answer') this.finalText = record.text
    const messages = snapshot ? this.sealedSnapshot : this.sealedItems
    const key = typeof record.id === 'string' && record.id ? `id:${record.id}` : `text:${record.phase ?? ''}:${record.text}`
    messages.set(key, record.text)
  }
}
