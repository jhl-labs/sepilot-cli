import { useEffect, useRef } from 'react'
import { useStdin } from 'ink'

export type InspectionFunctionKey = 'f11' | 'f12'

/** Decode the xterm-compatible sequences Ink intentionally drops from useInput(). */
export function inspectionFunctionKey(input: Buffer | string): InspectionFunctionKey | null {
  const value = Buffer.isBuffer(input) ? input.toString('utf8') : input
  if (/^\u001b\[23(?:;\d+)?~$/.test(value)) return 'f11'
  if (/^\u001b\[24(?:;\d+)?~$/.test(value)) return 'f12'
  return null
}

const FUNCTION_KEY_SEQUENCE_RE = /\u001b\[2[34](?:;\d+)?~/g
const MAX_PENDING_SEQUENCE_LENGTH = 32

function isInspectionFunctionKeyPrefix(value: string): boolean {
  return value === '\u001b'
    || value === '\u001b['
    || value === '\u001b[2'
    || value === '\u001b[23'
    || value === '\u001b[24'
    || /^\u001b\[2[34];\d*$/.test(value)
}

/**
 * Decode F11/F12 from a terminal byte stream rather than assuming each stdin
 * event contains exactly one complete escape sequence. PTYs are free to split
 * a sequence across events or coalesce it with neighbouring input.
 */
export class InspectionFunctionKeyDecoder {
  private pending = ''

  push(input: Buffer | string): InspectionFunctionKey[] {
    const value = Buffer.isBuffer(input) ? input.toString('utf8') : input
    const combined = `${this.pending}${value}`
    const keys: InspectionFunctionKey[] = []
    let match: RegExpExecArray | null
    let consumedThrough = 0

    FUNCTION_KEY_SEQUENCE_RE.lastIndex = 0
    while ((match = FUNCTION_KEY_SEQUENCE_RE.exec(combined)) !== null) {
      const key = inspectionFunctionKey(match[0])
      if (key) keys.push(key)
      consumedThrough = FUNCTION_KEY_SEQUENCE_RE.lastIndex
    }

    const tail = combined.slice(consumedThrough)
    this.pending = ''
    const searchStart = Math.max(0, tail.length - MAX_PENDING_SEQUENCE_LENGTH)
    for (let index = searchStart; index < tail.length; index += 1) {
      const suffix = tail.slice(index)
      if (isInspectionFunctionKeyPrefix(suffix)) {
        this.pending = suffix
        break
      }
    }

    return keys
  }

  reset(): void {
    this.pending = ''
  }
}

export function useFunctionKeys(
  handlers: { onF11(): void; onF12(): void },
  active = true,
): void {
  const { internal_eventEmitter: eventEmitter } = useStdin()
  const handlersRef = useRef(handlers)
  handlersRef.current = handlers
  const decoderRef = useRef<InspectionFunctionKeyDecoder | null>(null)
  if (!decoderRef.current) decoderRef.current = new InspectionFunctionKeyDecoder()

  useEffect(() => {
    if (!active) return

    const handleInput = (chunk: Buffer | string): void => {
      for (const key of decoderRef.current!.push(chunk)) {
        if (key === 'f11') handlersRef.current.onF11()
        else handlersRef.current.onF12()
      }
    }

    eventEmitter.on('input', handleInput)
    return () => {
      eventEmitter.off('input', handleInput)
      decoderRef.current?.reset()
    }
  }, [active, eventEmitter])
}
