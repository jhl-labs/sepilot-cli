import { createHash } from 'node:crypto'
import type { ChatRequest, ChatResponse } from '@sepilotd/core'

interface CacheEntry {
  response: ChatResponse
  timestamp: number
  hits: number
}

export class LLMCache {
  private cache = new Map<string, CacheEntry>()
  private maxSize: number
  private ttlMs: number

  constructor(maxSize = 1000, ttlMs = 300000) { // 5min default TTL
    this.maxSize = maxSize
    this.ttlMs = ttlMs
  }

  get(request: ChatRequest): ChatResponse | null {
    const key = this.hashRequest(request)
    const entry = this.cache.get(key)
    if (!entry) return null
    if (Date.now() - entry.timestamp > this.ttlMs) {
      this.cache.delete(key)
      return null
    }
    entry.hits++
    return entry.response
  }

  set(request: ChatRequest, response: ChatResponse): void {
    // Don't cache tool_use responses (they should always be fresh)
    if (response.finishReason === 'tool_use') return

    const key = this.hashRequest(request)

    // Evict oldest entries if at capacity
    if (this.cache.size >= this.maxSize) {
      const oldest = [...this.cache.entries()].sort((a, b) => a[1].timestamp - b[1].timestamp)[0]
      if (oldest) this.cache.delete(oldest[0])
    }

    this.cache.set(key, { response, timestamp: Date.now(), hits: 0 })
  }

  getStats(): { size: number; maxSize: number; hitRate: string } {
    let totalHits = 0
    for (const entry of this.cache.values()) totalHits += entry.hits
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      hitRate: this.cache.size > 0 ? `${totalHits} hits` : '0 hits',
    }
  }

  clear(): void {
    this.cache.clear()
  }

  private hashRequest(request: ChatRequest): string {
    // Hash model + messages content (ignore timestamps, request IDs)
    const content = JSON.stringify({
      model: request.model,
      messages: request.messages.map(m => ({ role: m.role, content: m.content })),
      tools: request.tools?.map(t => t.name),
      toolChoice: request.toolChoice,
      temperature: request.temperature,
      thinkingLevel: request.thinkingLevel,
    })
    return createHash('sha256').update(content).digest('hex').slice(0, 32)
  }
}
