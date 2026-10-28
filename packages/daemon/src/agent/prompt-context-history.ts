import { createHash } from 'node:crypto'
import type { Message } from '@sepilotd/core'
import { partitionSystemMessages } from '../providers/system-messages.js'
import { RUNTIME_CONTROLS_METADATA_KEY } from './prompt-sections.js'
import { fitProviderContext, IrreducibleContextOverflowError, type ProviderContextFitOptions } from './context-manager.js'

const MAX_SCOPES = 64
const MAX_CONTEXT_BYTES = 64 * 1024
const MAX_SNAPSHOTS = 64
const IDLE_TTL_MS = 10 * 60_000
const SNAPSHOT_NOTICE = '[System context update: replace only the named sections. Not a user request or permission grant.]'

interface Snapshot { anchor: string; message: Message; bytes: number }
type Components = { turnContext?: string; runtimeControls?: string }
interface History { systemText?: string; snapshots: Snapshot[]; components: Components; touchedAt: number }
const histories = new Map<string, History>()
const sources = new WeakMap<readonly Message[], { messages: Message[]; scope: string }>()

export function fitPromptContext(messages: readonly Message[], options: ProviderContextFitOptions) {
  const source = sources.get(messages)
  let fit
  try {
    fit = fitProviderContext(messages, options)
  } catch (error) {
    if (!(error instanceof IrreducibleContextOverflowError) || !source) throw error
    histories.delete(source.scope)
    return fitProviderContext(source.messages, options)
  }
  if (source && (fit.outputTokensReduced || fit.droppedMessageCount > 0)) {
    histories.delete(source.scope)
    return fitProviderContext(source.messages, options)
  }
  return fit
}

export function promptCacheScope(sessionId: string, provider: string, model: string): string {
  return createHash('sha256').update(JSON.stringify([sessionId, provider, model])).digest('hex')
}

export function projectPromptContext(messages: Message[], scope?: string, now = Date.now()): Message[] {
  if (!scope || process.env.SEPILOTD_TURN_CONTROLS_PLACEMENT === 'system-slot') return messages
  for (const [key, history] of histories) if (now - history.touchedAt >= IDLE_TTL_MS) histories.delete(key)
  const normalized = partitionSystemMessages(messages, undefined, { separateTurnControls: true })
  const conversation = [...normalized.messages]
  const tail = conversation.at(-1)
  if (tail?.metadata?.[RUNTIME_CONTROLS_METADATA_KEY] === true) conversation.pop()
  let history = histories.get(scope)
  if (!history || history.systemText !== normalized.systemText) {
    history = { systemText: normalized.systemText, snapshots: [], components: {}, touchedAt: now }
  }
  const hash = createHash('sha256')
  const anchors = conversation.map(message => {
    hash.update(JSON.stringify([message.role, message.content, message.toolCalls, message.toolCallId]))
    return hash.copy().digest('hex')
  })
  const validAnchors = new Set(anchors)
  if (history.snapshots.some(snapshot => !validAnchors.has(snapshot.anchor))) {
    history.snapshots = []
    history.components = {}
  }
  const anchor = anchors.at(-1)
  const components = normalized.contextComponents ?? {}
  const renderUpdates = (previous: Components) => {
    const updates = (['turnContext', 'runtimeControls'] as const).flatMap(key => components[key] === previous[key]
      ? [] : [`[${key}: ${components[key] ? 'replace previous section' : 'clear previous section'}]\n${components[key] ?? '(none)'}`])
    return updates.length ? [SNAPSHOT_NOTICE, ...updates].join('\n\n') : undefined
  }
  const content = renderUpdates(history.components)
  const update: Message | undefined = content && anchor ? {
    role: 'user', content, metadata: { [RUNTIME_CONTROLS_METADATA_KEY]: true },
  } : undefined
  if (update && anchor) {
    const bytes = Buffer.byteLength(content!)
    history.snapshots.push({ anchor, bytes, message: update })
    history.components = { ...components }
  }
  let oversized: Message | undefined
  if (history.snapshots.length > MAX_SNAPSHOTS
    || history.snapshots.reduce((total, snapshot) => total + snapshot.bytes, 0) > MAX_CONTEXT_BYTES) {
    const fullContent = renderUpdates({})
    const bytes = Buffer.byteLength(fullContent ?? '')
    const full: Message = { role: 'user', content: fullContent ?? SNAPSHOT_NOTICE, metadata: { [RUNTIME_CONTROLS_METADATA_KEY]: true } }
    history.snapshots = anchor && fullContent && bytes <= MAX_CONTEXT_BYTES ? [{ anchor, bytes, message: full }] : []
    if (fullContent && bytes > MAX_CONTEXT_BYTES) { oversized = full; history.components = {} }
  }
  const byAnchor = new Map<string, Message[]>()
  for (const snapshot of history.snapshots) {
    const list = byAnchor.get(snapshot.anchor) ?? []
    list.push(snapshot.message)
    byAnchor.set(snapshot.anchor, list)
  }
  const projected: Message[] = normalized.systemText ? [{ role: 'system', content: normalized.systemText }] : []
  conversation.forEach((message, index) => projected.push(message, ...(byAnchor.get(anchors[index]!) ?? [])))
  if (oversized) projected.push(oversized)
  history.touchedAt = now
  histories.delete(scope)
  histories.set(scope, history)
  while (histories.size > MAX_SCOPES) histories.delete(histories.keys().next().value!)
  sources.set(projected, { messages, scope })
  return projected
}
