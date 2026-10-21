import { z } from 'zod'

export type RemoteTransportKind = 'extension' | 'cdp'

export interface RemoteTransportCapabilities {
  refs: boolean
  console: boolean
  network: boolean
}

export interface RemoteElement {
  ref: string
  role: string
  name: string
  value?: string
  href?: string
  x: number
  y: number
}

export interface RemoteSnapshot {
  url: string
  title: string
  text: string
  elements: RemoteElement[]
  image?: string
  snapshotId: string
}

export interface RemoteFrame {
  url: string
  title: string
  image: string
  capturedAt: number
  width?: number
  height?: number
}

export type StillFormat = 'jpeg' | 'png'

export const remoteFrameSchema = z.object({
  url: z.string().max(8192),
  title: z.string().max(500),
  image: z
    .string()
    .max(2_000_000)
    .regex(/^data:image\/jpeg;base64,[A-Za-z0-9+/=]+$/),
  capturedAt: z.number().finite(),
  width: z.number().int().positive().max(16384).optional(),
  height: z.number().int().positive().max(16384).optional(),
})

const httpUrl = z
  .string()
  .url()
  .max(8192)
  .refine(
    (url) =>
      ['https:', 'http:'].includes(new URL(url).protocol) &&
      !new URL(url).username &&
      !new URL(url).password,
  )

const ref = z.string().regex(/^e\d+$/)
const expectedUrl = z.string().url()
const snapshotId = z.string().max(100).optional()

export const remoteActionSchema = z.discriminatedUnion('action', [
  z.object({ action: z.literal('snapshot') }),
  z.object({
    action: z.literal('handover'),
    reason: z.string().trim().min(1).max(1000),
    expectedUrl,
  }),
  z.object({ action: z.literal('navigate'), url: httpUrl }),
  z.object({
    action: z.literal('click'),
    ref: ref.optional(),
    x: z.number().min(0).max(20000).optional(),
    y: z.number().min(0).max(20000).optional(),
    expectedUrl,
    snapshotId,
  }),
  z.object({ action: z.literal('fill'), ref, text: z.string().max(10000), expectedUrl, snapshotId }),
  z.object({ action: z.literal('type'), text: z.string().max(10000), expectedUrl }),
  z.object({ action: z.literal('press'), key: z.string().min(1).max(40), expectedUrl }),
  z.object({
    action: z.literal('scroll'),
    deltaY: z.number().min(-10000).max(10000).optional(),
    ref: ref.optional(),
    expectedUrl,
    snapshotId,
  }),
  z.object({
    action: z.literal('key'),
    key: z.enum(['Enter', 'Tab', 'Escape', 'Backspace', 'ArrowDown', 'ArrowUp']),
    expectedUrl,
  }),
  z.object({
    action: z.literal('select'),
    ref,
    values: z.array(z.string().max(500)).min(1).max(50),
    expectedUrl,
    snapshotId,
  }),
  z.object({ action: z.literal('hover'), ref, expectedUrl, snapshotId }),
  z.object({
    action: z.literal('wait_for'),
    text: z.string().min(1).max(500).optional(),
    ref: ref.optional(),
    timeoutMs: z.number().int().min(100).max(10000).default(5000),
    expectedUrl,
    snapshotId,
  }),
  z.object({
    action: z.literal('console'),
    level: z.enum(['error', 'warning', 'info', 'log']).optional(),
  }),
  z.object({ action: z.literal('network'), filter: z.string().max(200).optional() }),
])
export type RemoteAction = z.infer<typeof remoteActionSchema>

export interface BrowserTransport {
  readonly kind: RemoteTransportKind
  readonly capabilities: RemoteTransportCapabilities
  currentUrl(): string | undefined
  snapshot(signal?: AbortSignal): Promise<RemoteSnapshot>
  act(action: RemoteAction, signal?: AbortSignal): Promise<string>
  onFrame(listener: (frame: RemoteFrame) => void): () => void
  still?(maxWidth: number, format?: StillFormat): Promise<RemoteFrame | undefined>
  onUserInput(listener: () => void): () => void
  onClosed(listener: (reason: string) => void): () => void
  close(): Promise<void>
}

export function formatSnapshot(
  snapshot: RemoteSnapshot,
  kind: RemoteTransportKind,
  capabilities: RemoteTransportCapabilities,
): string {
  const yesNo = (value: boolean) => (value ? 'yes' : 'no')
  const lines = [
    `URL: ${snapshot.url}`,
    `Title: ${snapshot.title}`,
    `Transport: ${kind} (refs: ${yesNo(capabilities.refs)}, console: ${yesNo(capabilities.console)}, network: ${yesNo(capabilities.network)})`,
  ]
  if (snapshot.snapshotId) lines.push(`Snapshot: ${snapshot.snapshotId}`)
  if (snapshot.elements.length > 0) {
    lines.push(`Elements (${snapshot.elements.length}):`)
    for (const element of snapshot.elements) {
      const value = element.value !== undefined ? ` value=${JSON.stringify(element.value)}` : ''
      const href = element.href !== undefined ? ` href=${JSON.stringify(element.href)}` : ''
      lines.push(
        `${element.ref} ${element.role} ${JSON.stringify(element.name)}${value}${href} (${Math.round(element.x)},${Math.round(element.y)})`,
      )
    }
  }
  lines.push('Text:', snapshot.text)
  return lines.join('\n')
}
