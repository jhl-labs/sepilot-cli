import { PassThrough } from 'node:stream'
import { StringDecoder } from 'node:string_decoder'

export interface MouseEvent {
  kind: 'wheel-up' | 'wheel-down' | 'press' | 'release' | 'move'
  x: number
  y: number
  modifiers: { shift: boolean; alt: boolean; ctrl: boolean }
}

export interface ParseResult {
  event: MouseEvent
  consumed: number
}

export interface TuiInputSource extends NodeJS.ReadableStream {
  isTTY?: boolean
  setRawMode?: (enabled: boolean) => unknown
  ref?: () => unknown
  unref?: () => unknown
  pause(): this
  resume(): this
  on(event: 'data', listener: (chunk: Buffer | string) => void): this
  off(event: 'data', listener: (chunk: Buffer | string) => void): this
  once(event: 'end' | 'close', listener: () => void): this
  once(event: 'error', listener: (error: Error) => void): this
  off(event: 'end' | 'close', listener: () => void): this
  off(event: 'error', listener: (error: Error) => void): this
}

export interface MouseAwareStdin {
  isTTY: boolean
  setRawMode(enabled: boolean): this
  ref(): this
  unref(): this
  pause(): this
  resume(): this
  once(event: 'end' | 'close', listener: () => void): this
  once(event: 'error', listener: (error: Error) => void): this
  on(event: 'data', listener: (chunk: Buffer | string) => void): this
  on(event: 'error', listener: (error: Error) => void): this
  on(event: 'mouse', listener: (event: MouseEvent) => void): this
  off(event: 'end' | 'close', listener: () => void): this
  off(event: 'error', listener: (error: Error) => void): this
  off(event: 'data', listener: (chunk: Buffer | string) => void): this
  off(event: 'mouse', listener: (event: MouseEvent) => void): this
  dispose(): void
}

const SGR_PREFIX = '\x1b[<'
const X10_PREFIX = '\x1b[M'
const SGR_RE = /^\x1b\[<(\d+);(\d+);(\d+)([Mm])/
const X10_RE = /^\x1b\[M([\s\S])([\s\S])([\s\S])/
const PARTIAL_MOUSE_PREFIX_RE = /^\x1b(?:\[(?:<[\d;]*)?|M[\s\S]{0,2})?$/
const FUNCTION_KEY_RE = /^\x1b\[2[34](?:;\d+)?~/
const PARTIAL_FUNCTION_KEY_PREFIX_RE = /^(?:\x1b|\x1b\[|\x1b\[2|\x1b\[2[34](?:;\d*)?)$/
const FUNCTION_KEY_PREFIXES = ['\x1b[23', '\x1b[24'] as const
const ESCAPE_SEQUENCE_TIMEOUT_MS = 30
const MOUSE_AWARE_STDIN_BRAND = Symbol('mouse-aware-stdin')

function buildMouseEvent(raw: number, x: number, y: number, isRelease: boolean): MouseEvent {
  const isMotion = (raw & 32) !== 0
  const isWheel = (raw & 64) !== 0
  const button = raw & 0b11
  const modifiers = {
    shift: (raw & 4) !== 0,
    alt: (raw & 8) !== 0,
    ctrl: (raw & 16) !== 0,
  }

  let kind: MouseEvent['kind']
  if (isWheel) {
    kind = button === 0 ? 'wheel-up' : 'wheel-down'
  } else if (isMotion) {
    kind = 'move'
  } else if (isRelease || button === 3) {
    kind = 'release'
  } else {
    kind = 'press'
  }

  return { kind, x, y, modifiers }
}

export function parseMouseSgr(buffer: string): ParseResult | null {
  const match = SGR_RE.exec(buffer)
  if (!match) return null

  const raw = Number(match[1])
  const x = Number(match[2])
  const y = Number(match[3])
  const trailer = match[4]

  return {
    event: buildMouseEvent(raw, x, y, trailer === 'm'),
    consumed: match[0].length,
  }
}

export function parseMouseX10(buffer: string): ParseResult | null {
  const match = X10_RE.exec(buffer)
  if (!match) return null

  const raw = match[1].charCodeAt(0) - 32
  const x = match[2].charCodeAt(0) - 32
  const y = match[3].charCodeAt(0) - 32

  return {
    event: buildMouseEvent(raw, x, y, false),
    consumed: match[0].length,
  }
}

export function parseMouseSequence(buffer: string): ParseResult | null {
  return parseMouseSgr(buffer) ?? parseMouseX10(buffer)
}

export function findNextMousePrefix(buffer: string): number {
  const sgrIndex = buffer.indexOf(SGR_PREFIX)
  const x10Index = buffer.indexOf(X10_PREFIX)
  if (sgrIndex < 0) return x10Index
  if (x10Index < 0) return sgrIndex
  return Math.min(sgrIndex, x10Index)
}

export function findPartialMousePrefixStart(buffer: string): number {
  for (let index = buffer.length - 1; index >= 0; index -= 1) {
    if (buffer[index] !== '\x1b') continue
    if (PARTIAL_MOUSE_PREFIX_RE.test(buffer.slice(index))) {
      return index
    }
  }

  return -1
}

function findNextFunctionKeyPrefix(buffer: string): number {
  let earliest = -1
  for (const prefix of FUNCTION_KEY_PREFIXES) {
    const index = buffer.indexOf(prefix)
    if (index >= 0 && (earliest < 0 || index < earliest)) earliest = index
  }
  return earliest
}

function findPartialFunctionKeyPrefixStart(buffer: string): number {
  for (let index = buffer.length - 1; index >= 0; index -= 1) {
    if (buffer[index] !== '\x1b') continue
    if (PARTIAL_FUNCTION_KEY_PREFIX_RE.test(buffer.slice(index))) return index
  }
  return -1
}

class MouseAwareStdinProxy extends PassThrough implements MouseAwareStdin {
  readonly isTTY: boolean
  readonly [MOUSE_AWARE_STDIN_BRAND] = true
  private readonly decoder = new StringDecoder('utf8')
  private buffer = ''
  private escapeFlushTimer: ReturnType<typeof setTimeout> | null = null
  private readonly handleData = (chunk: Buffer | string): void => {
    this.clearEscapeFlushTimer()
    const text = typeof chunk === 'string' ? chunk : this.decoder.write(chunk)
    this.buffer += text

    while (this.buffer.length > 0) {
      const functionKeyIndex = findNextFunctionKeyPrefix(this.buffer)
      const mouseIndex = this.captureMouse ? findNextMousePrefix(this.buffer) : -1
      const idx = functionKeyIndex < 0
        ? mouseIndex
        : mouseIndex < 0
          ? functionKeyIndex
          : Math.min(functionKeyIndex, mouseIndex)
      if (idx < 0) {
        const partialFunctionKey = findPartialFunctionKeyPrefixStart(this.buffer)
        const partialMouse = this.captureMouse ? findPartialMousePrefixStart(this.buffer) : -1
        const keep = partialFunctionKey < 0
          ? partialMouse
          : partialMouse < 0
            ? partialFunctionKey
            : Math.min(partialFunctionKey, partialMouse)
        if (keep >= 0) {
          this.forward(this.buffer.slice(0, keep))
          this.buffer = this.buffer.slice(keep)
          this.scheduleEscapeFlush()
        } else {
          this.forward(this.buffer)
          this.buffer = ''
        }
        break
      }

      if (idx > 0) {
        this.forward(this.buffer.slice(0, idx))
        this.buffer = this.buffer.slice(idx)
      }

      const functionKey = FUNCTION_KEY_RE.exec(this.buffer)
      if (functionKey) {
        this.forward(functionKey[0])
        this.buffer = this.buffer.slice(functionKey[0].length)
        continue
      }

      if (PARTIAL_FUNCTION_KEY_PREFIX_RE.test(this.buffer)) {
        this.scheduleEscapeFlush()
        break
      }

      const parsed = this.captureMouse ? parseMouseSequence(this.buffer) : null
      if (parsed) {
        this.emit('mouse', parsed.event)
        this.buffer = this.buffer.slice(parsed.consumed)
        continue
      }

      if (PARTIAL_MOUSE_PREFIX_RE.test(this.buffer)) {
        this.scheduleEscapeFlush()
        break
      }

      this.forward(this.buffer[0] ?? '')
      this.buffer = this.buffer.slice(1)
    }
  }

  private readonly handleEnd = (): void => {
    this.clearEscapeFlushTimer()
    this.buffer += this.decoder.end()
    if (this.buffer.length > 0) {
      this.forward(this.buffer)
      this.buffer = ''
    }
    this.end()
  }

  private readonly handleClose = (): void => {
    this.handleEnd()
  }

  private readonly handleError = (error: Error): void => {
    this.destroy(error)
  }

  constructor(
    private readonly source: TuiInputSource,
    private readonly captureMouse: boolean,
  ) {
    super()
    this.isTTY = Boolean(source.isTTY)
    source.on('data', this.handleData)
    source.once('end', this.handleEnd)
    source.once('close', this.handleClose)
    source.once('error', this.handleError)
  }

  dispose(): void {
    this.clearEscapeFlushTimer()
    this.source.off('data', this.handleData)
    this.source.off('end', this.handleEnd)
    this.source.off('close', this.handleClose)
    this.source.off('error', this.handleError)
  }

  setRawMode(enabled: boolean): this {
    this.source.setRawMode?.(enabled)
    return this
  }

  ref(): this {
    this.source.ref?.()
    return this
  }

  unref(): this {
    this.source.unref?.()
    return this
  }

  override pause(): this {
    this.source.pause()
    return super.pause()
  }

  override resume(): this {
    this.source.resume()
    return super.resume()
  }

  private forward(text: string): void {
    if (!text) return
    this.write(text)
  }

  private scheduleEscapeFlush(): void {
    if (this.escapeFlushTimer) return
    this.escapeFlushTimer = setTimeout(() => {
      this.escapeFlushTimer = null
      if (this.buffer.length > 0) {
        this.forward(this.buffer)
        this.buffer = ''
      }
    }, ESCAPE_SEQUENCE_TIMEOUT_MS)
    this.escapeFlushTimer.unref?.()
  }

  private clearEscapeFlushTimer(): void {
    if (!this.escapeFlushTimer) return
    clearTimeout(this.escapeFlushTimer)
    this.escapeFlushTimer = null
  }
}

export function createMouseAwareStdin(source: TuiInputSource): MouseAwareStdin {
  return new MouseAwareStdinProxy(source, true)
}

export function createFunctionKeyAwareStdin(source: TuiInputSource): MouseAwareStdin {
  return new MouseAwareStdinProxy(source, false)
}

export function isMouseEventSource(stream: unknown): stream is MouseAwareStdin {
  return Boolean(
    (stream as { [MOUSE_AWARE_STDIN_BRAND]?: boolean } | null)?.[MOUSE_AWARE_STDIN_BRAND],
  )
}
