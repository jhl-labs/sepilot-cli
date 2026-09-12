import { randomUUID } from 'node:crypto'
import { mkdir, realpath, stat } from 'node:fs/promises'
import { parse, resolve } from 'node:path'
import type { SessionMeta } from '@sepilotd/core'
import { loadAutoCompactedSessionContext, resolveSessionContextMaxMessages } from '../agent/auto-compaction.js'
import type { ChannelPipelineCapabilities } from '../server/runtime/capabilities.js'
import type { NormalizedIncomingChannelMessage } from './normalizer.js'
import { resolveToolPath } from '../tools/path-utils.js'

export type DefaultProvider = NonNullable<
  ReturnType<ChannelPipelineCapabilities['providerRegistry']['getDefault']>
>

export type ChannelSessionContextMessages = Awaited<
  ReturnType<typeof loadAutoCompactedSessionContext>
>['messages']

type LoadedChannelSessionContext = Awaited<
  ReturnType<typeof loadAutoCompactedSessionContext>
>

export interface ResolvedChannelSession {
  sessionId: string
  session: SessionMeta
  previousMessages: ChannelSessionContextMessages
}

export class ChannelSessionResolver {
  constructor(private readonly runtime: ChannelPipelineCapabilities) {}

  private resolveDefaultModel(provider: DefaultProvider): string {
    return this.runtime.config.agent?.defaultModel ?? provider.models[0]?.id ?? 'default'
  }

  private async resolveDefaultWorkspaceRoot(): Promise<string | undefined> {
    const configured = this.runtime.config.channelPipeline?.defaultWorkspaceRoot?.trim()
    if (!configured) return undefined

    const resolved = resolve(resolveToolPath(configured))
    if (resolved === parse(resolved).root) {
      throw new Error('channelPipeline.defaultWorkspaceRoot cannot be a filesystem root')
    }
    await mkdir(resolved, { recursive: true })
    const canonical = await realpath(resolved)
    if (!(await stat(canonical)).isDirectory()) {
      throw new Error(`channelPipeline.defaultWorkspaceRoot is not a directory: ${resolved}`)
    }
    return canonical
  }

  private async applyDefaultWorkspaceRoot(session: SessionMeta): Promise<SessionMeta> {
    if (session.cwd) return session
    const cwd = await this.resolveDefaultWorkspaceRoot()
    if (!cwd) return session

    const updated = await this.runtime.sessions.updateMeta?.(session.id, { cwd })
    return updated ?? { ...session, cwd }
  }

  async resolve(
    normalized: NormalizedIncomingChannelMessage,
    provider: DefaultProvider,
  ): Promise<ResolvedChannelSession> {
    const { message, sessionKey: bindingKey } = normalized
    if (bindingKey && this.runtime.channelSessionStore) {
      const binding = await this.runtime.channelSessionStore.get(bindingKey)
      if (binding) {
        const existingSession = await this.runtime.sessions.get(binding.sessionId)
        if (existingSession && existingSession.status !== 'completed') {
          const activeSession = await this.applyDefaultWorkspaceRoot(existingSession)
          const previousMessages = await this.loadPreviousMessages(
            binding.sessionId,
            provider,
            activeSession.model,
          )
          await this.runtime.sessions.appendEvent(binding.sessionId, {
            type: 'user_message',
            id: message.messageId,
            timestamp: message.timestamp,
            content: message.text,
          })
          await this.runtime.channelSessionStore.bind({
            ...binding,
            updatedAt: message.timestamp,
          })
          return {
            sessionId: binding.sessionId,
            session: activeSession,
            previousMessages,
          }
        }

        await this.runtime.channelSessionStore.delete(bindingKey)
      }
    }

    const session = await this.createNewSession(message, provider)
    if (bindingKey && this.runtime.channelSessionStore) {
      await this.runtime.channelSessionStore.bind({
        key: bindingKey,
        sessionId: session.id,
        channelType: message.channelType,
        channelId: message.channelId,
        updatedAt: message.timestamp,
      })
    }
    return {
      sessionId: session.id,
      session,
      previousMessages: [],
    }
  }

  private async createNewSession(
    message: NormalizedIncomingChannelMessage['message'],
    provider: DefaultProvider,
  ): Promise<SessionMeta> {
    const sessionId = randomUUID()
    const cwd = await this.resolveDefaultWorkspaceRoot()
    const meta = {
      id: sessionId,
      title: `[${message.channelType}] ${message.text.slice(0, 40)}`,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      provider: provider.id,
      model: this.resolveDefaultModel(provider),
      device: this.runtime.config.device.name,
      status: 'active',
      tags: [`channel:${message.channelType}`],
      ...(cwd ? { cwd } : {}),
    } satisfies Omit<SessionMeta, 'messageCount' | 'totalTokens' | 'totalCost'>
    const session = await this.runtime.sessions.create(meta)
    await this.runtime.sessions.appendEvent(sessionId, {
      type: 'user_message',
      id: message.messageId,
      timestamp: message.timestamp,
      content: message.text,
    })
    return session ?? {
      ...meta,
      messageCount: 0,
      totalTokens: { input: 0, output: 0 },
      totalCost: 0,
    }
  }

  private async loadPreviousMessages(
    sessionId: string,
    provider: DefaultProvider,
    model: string | undefined,
  ): Promise<LoadedChannelSessionContext['messages']> {
    return (await loadAutoCompactedSessionContext({
      sessionStore: this.runtime.sessions,
      sessionId,
      provider,
      model,
      hooks: this.runtime.hookRegistry,
      maxMessages: resolveSessionContextMaxMessages(),
    })).messages
  }

  async resolveFork(
    normalized: NormalizedIncomingChannelMessage,
    provider: DefaultProvider,
    parentSessionId: string | undefined,
  ): Promise<ResolvedChannelSession> {
    const { message } = normalized
    const sessionId = randomUUID()
    const parentSession = parentSessionId
      ? await this.runtime.sessions.get(parentSessionId)
      : null
    const cwd = parentSession?.cwd ?? await this.resolveDefaultWorkspaceRoot()
    const tags = [
      `channel:${message.channelType}`,
      'fork',
      ...(parentSessionId ? [`parent:${parentSessionId}`] : []),
    ]
    const meta = {
      id: sessionId,
      title: `[${message.channelType}/fork] ${message.text.slice(0, 40)}`,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      provider: provider.id,
      model: this.resolveDefaultModel(provider),
      device: this.runtime.config.device.name,
      status: 'active',
      tags,
      ...(cwd ? { cwd } : {}),
    } satisfies Omit<SessionMeta, 'messageCount' | 'totalTokens' | 'totalCost'>
    const created = await this.runtime.sessions.create(meta)
    await this.runtime.sessions.appendEvent(sessionId, {
      type: 'user_message',
      id: message.messageId,
      timestamp: message.timestamp,
      content: message.text,
    })

    const previousMessages: LoadedChannelSessionContext['messages'] = parentSessionId
      ? await this.loadPreviousMessages(parentSessionId, provider, meta.model)
      : []

    return {
      sessionId,
      session: created ?? {
        ...meta,
        messageCount: 0,
        totalTokens: { input: 0, output: 0 },
        totalCost: 0,
      },
      previousMessages,
    }
  }
}
