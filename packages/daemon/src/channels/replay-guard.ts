import type { ChannelReplayState } from '../server/runtime/channel-replays.js'
import type { ChannelPipelineCapabilities } from '../server/runtime/capabilities.js'
import type { NormalizedIncomingChannelMessage } from './normalizer.js'

interface ReplayEntry {
  expiresAt: number
  state: ChannelReplayState
}

export interface ChannelReplayClaim {
  key?: string
  duplicate: boolean
  state?: ChannelReplayState
}

export class ChannelReplayGuard {
  private static readonly MESSAGE_DEDUPE_TTL_MS = 10 * 60 * 1000
  private static readonly MESSAGE_DEDUPE_MAX_ENTRIES = 10_000
  private readonly inboundMessageDedup = new Map<string, ReplayEntry>()

  constructor(private readonly runtime: ChannelPipelineCapabilities) {}

  async claim(
    normalized: NormalizedIncomingChannelMessage,
  ): Promise<ChannelReplayClaim> {
    const { message, replayKey: key } = normalized
    if (!key) {
      return { duplicate: false }
    }

    if (this.runtime.channelReplayStore) {
      return this.runtime.channelReplayStore.claim({
        key,
        channelType: message.channelType,
        channelId: message.channelId,
        messageId: message.messageId,
        ttlMs: ChannelReplayGuard.MESSAGE_DEDUPE_TTL_MS,
      })
    }

    const now = Date.now()
    this.prune(now)

    const existing = this.inboundMessageDedup.get(key)
    if (existing && existing.expiresAt > now) {
      return {
        key,
        duplicate: true,
        state: existing.state,
      }
    }

    this.inboundMessageDedup.set(key, {
      expiresAt: now + ChannelReplayGuard.MESSAGE_DEDUPE_TTL_MS,
      state: 'processing',
    })
    this.prune(now)
    return { key, duplicate: false }
  }

  async markProcessed(key?: string): Promise<void> {
    if (!key) return

    if (this.runtime.channelReplayStore) {
      await this.runtime.channelReplayStore.markProcessed(
        key,
        ChannelReplayGuard.MESSAGE_DEDUPE_TTL_MS,
      )
      return
    }

    const entry = this.inboundMessageDedup.get(key)
    if (!entry) return

    this.inboundMessageDedup.set(key, {
      expiresAt: Date.now() + ChannelReplayGuard.MESSAGE_DEDUPE_TTL_MS,
      state: 'processed',
    })
  }

  async release(key?: string): Promise<void> {
    if (!key) return

    if (this.runtime.channelReplayStore) {
      await this.runtime.channelReplayStore.release(key)
      return
    }

    this.inboundMessageDedup.delete(key)
  }

  private prune(now = Date.now()): void {
    for (const [key, entry] of this.inboundMessageDedup) {
      if (entry.expiresAt <= now) {
        this.inboundMessageDedup.delete(key)
      }
    }

    while (
      this.inboundMessageDedup.size > ChannelReplayGuard.MESSAGE_DEDUPE_MAX_ENTRIES
    ) {
      const oldestKey = this.inboundMessageDedup.keys().next().value
      if (typeof oldestKey !== 'string') break
      this.inboundMessageDedup.delete(oldestKey)
    }
  }
}
