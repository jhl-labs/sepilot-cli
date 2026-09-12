import { useEffect } from 'react'
import { useStdin } from 'ink'

export function useRawCtrlC(onCtrlC: () => void, enabled = true) {
  const { stdin } = useStdin()

  useEffect(() => {
    if (!enabled) return

    const handleData = (chunk: Buffer | string): void => {
      const text = Buffer.isBuffer(chunk) ? chunk.toString('utf8') : chunk
      if (text.includes('\u0003')) {
        onCtrlC()
      }
    }

    stdin.on('data', handleData)
    return () => {
      stdin.off('data', handleData)
    }
  }, [enabled, onCtrlC, stdin])
}
