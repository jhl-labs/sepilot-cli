import { useEffect, useRef } from 'react'
import { useStdin } from 'ink'
import {
  findNextMousePrefix,
  findPartialMousePrefixStart,
  isMouseEventSource,
  parseMouseSequence,
  type MouseEvent,
} from '../utils/mouse-input.js'

export type { MouseEvent } from '../utils/mouse-input.js'
export { parseMouseSgr } from '../utils/mouse-input.js'

export function useMouseEvents(handler: (event: MouseEvent) => void): void {
  const { stdin, setRawMode, isRawModeSupported } = useStdin()
  const handlerRef = useRef(handler)

  useEffect(() => {
    handlerRef.current = handler
  }, [handler])

  useEffect(() => {
    if (!isRawModeSupported) return
    setRawMode(true)

    const emitEvent = (event: MouseEvent): void => {
      try {
        handlerRef.current(event)
      } catch (err) {
        process.stderr.write(`useMouseEvents handler error: ${String(err)}\n`)
      }
    }

    if (isMouseEventSource(stdin)) {
      stdin.on('mouse', emitEvent)
      return () => {
        stdin.off('mouse', emitEvent)
        stdin.pause()
        setRawMode(false)
      }
    }

    let buffer = ''

    const onData = (chunk: Buffer | string): void => {
      const text = typeof chunk === 'string' ? chunk : chunk.toString('utf8')
      buffer += text

      while (buffer.length > 0) {
        const idx = findNextMousePrefix(buffer)
        if (idx < 0) {
          const keep = findPartialMousePrefixStart(buffer)
          buffer = keep >= 0 ? buffer.slice(keep) : ''
          break
        }
        if (idx > 0) buffer = buffer.slice(idx)

        const parsed = parseMouseSequence(buffer)
        if (!parsed) break

        emitEvent(parsed.event)
        buffer = buffer.slice(parsed.consumed)
      }
    }

    stdin.on('data', onData)
    return () => {
      stdin.off('data', onData)
      stdin.pause()
      setRawMode(false)
    }
  }, [isRawModeSupported, setRawMode, stdin])
}
