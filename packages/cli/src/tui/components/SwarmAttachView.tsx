import { useEffect, useMemo, useRef, useState } from 'react'
import { Box, Text, useStdin } from 'ink'
import type { SwarmAgentName, SwarmClient } from '@sepilotd/api-client'
import {
  createSwarmAttachInputState,
  parseSwarmAttachInputChunk,
} from '../../commands/swarm.js'
import { colors } from '../theme.js'

export type SwarmAttachExitReason = 'detached' | 'cancelled' | 'ended' | 'error'

export interface SwarmAttachTarget {
  runId: string
  handle: string
  agent: SwarmAgentName
  role?: string
  tmuxSessionName?: string
}

interface SwarmAttachViewProps {
  client: SwarmClient
  target: SwarmAttachTarget
  width: number
  height: number
  onExit: (reason: SwarmAttachExitReason, message?: string) => void
}

function toInputBuffer(chunk: Buffer | string): Buffer {
  return Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk, 'utf8')
}

function lastLines(text: string, limit: number): string[] {
  if (!text) return ['']
  const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
  return normalized.split('\n').slice(-Math.max(1, limit))
}

export function SwarmAttachView({
  client,
  target,
  width,
  height,
  onExit,
}: SwarmAttachViewProps) {
  const { stdin } = useStdin()
  const [snapshot, setSnapshot] = useState('')
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const onExitRef = useRef(onExit)
  const clientRef = useRef(client)
  const targetRef = useRef(target)

  useEffect(() => {
    onExitRef.current = onExit
  }, [onExit])

  useEffect(() => {
    clientRef.current = client
  }, [client])

  useEffect(() => {
    targetRef.current = target
  }, [target])

  const paneHeight = Math.max(1, height - 5)
  const paneLines = useMemo(() => lastLines(snapshot, paneHeight), [paneHeight, snapshot])

  useEffect(() => {
    let stopped = false
    let polling = false
    const ac = new AbortController()
    const captureLines = (): number => Math.max(10, Math.min(5000, paneHeight))
    const refresh = async (): Promise<void> => {
      if (stopped || polling) return
      polling = true
      try {
        const pane = await client.captureAgent(target.runId, target.handle, {
          lines: captureLines(),
        })
        if (!stopped) {
          setSnapshot(pane.text)
          setError(null)
        }
      } catch (err) {
        if (!stopped) {
          const message = err instanceof Error ? err.message : String(err)
          setError(message)
        }
      } finally {
        polling = false
      }
    }

    void refresh()
    const pollTimer = setInterval(() => {
      void refresh()
    }, 500)
    pollTimer.unref?.()

    void (async () => {
      try {
        for await (const ev of client.streamEvents(target.runId, { signal: ac.signal })) {
          if (stopped) return
          if (ev.type === 'pane.snapshot' && ev.handle === target.handle) {
            setSnapshot(ev.text)
            setError(null)
          }
          if (ev.type === 'run.ended') {
            stopped = true
            onExitRef.current('ended', `${target.runId} ended with status ${ev.status}`)
            return
          }
        }
      } catch {
        // The normal detach path aborts the SSE reader.
      }
    })()

    return () => {
      stopped = true
      ac.abort()
      clearInterval(pollTimer)
    }
  }, [client, paneHeight, target.handle, target.runId])

  useEffect(() => {
    void client
      .sendKeys(target.runId, target.handle, { resize: { cols: width, rows: paneHeight } })
      .catch(() => undefined)
  }, [client, paneHeight, target.handle, target.runId, width])

  useEffect(() => {
    const source = stdin as unknown as {
      on?: (event: 'data', listener: (chunk: Buffer | string) => void) => void
      off?: (event: 'data', listener: (chunk: Buffer | string) => void) => void
      resume?: () => void
    } | null
    if (!source?.on || !source.off) return

    const inputState = createSwarmAttachInputState()
    let closed = false
    const handleData = (chunk: Buffer | string): void => {
      if (closed) return
      const actions = parseSwarmAttachInputChunk(toInputBuffer(chunk), inputState)
      for (const action of actions) {
        if (action.type === 'detach') {
          closed = true
          onExitRef.current('detached', `${targetRef.current.runId}/${targetRef.current.handle}`)
          return
        }
        if (action.type === 'cancel') {
          closed = true
          void clientRef.current
            .cancel(targetRef.current.runId)
            .catch(() => undefined)
            .finally(() => {
              onExitRef.current('cancelled', targetRef.current.runId)
            })
          return
        }
        if (action.type === 'list-agents') {
          void clientRef.current
            .get(targetRef.current.runId)
            .then((run) => {
              setNotice(
                run.agents.map((a) => `${a.handle} ${a.agent} ${a.status}`).join('  |  '),
              )
            })
            .catch((err) => {
              setNotice(err instanceof Error ? err.message : String(err))
            })
          continue
        }
        void clientRef.current
          .sendKeys(targetRef.current.runId, targetRef.current.handle, {
            keys: action.keys,
            enter: action.enter,
          })
          .catch((err) => {
            setError(err instanceof Error ? err.message : String(err))
          })
      }
    }

    source.resume?.()
    source.on('data', handleData)
    return () => {
      closed = true
      source.off?.('data', handleData)
    }
  }, [stdin])

  return (
    <Box flexDirection="column" height={Math.max(1, height)} width={width} paddingX={1}>
      <Box height={1}>
        <Text color={colors.primary} bold wrap="truncate-end">
          Swarm Attach {target.runId}/{target.handle} ({target.agent})
        </Text>
      </Box>
      <Box height={1}>
        <Text color={colors.dimText} wrap="truncate-end">
          Ctrl-B D detach · Ctrl-B K cancel · Ctrl-B L agents
          {target.tmuxSessionName ? ` · ${target.tmuxSessionName}` : ''}
        </Text>
      </Box>
      <Box
        borderStyle="single"
        borderColor={error ? colors.error : colors.border}
        flexDirection="column"
        height={Math.max(3, paneHeight + 2)}
        overflow="hidden"
      >
        {paneLines.map((line, index) => (
          <Text key={`${index}-${line}`} wrap="truncate-end">
            {line || ' '}
          </Text>
        ))}
      </Box>
      <Box height={1}>
        <Text color={error ? colors.error : colors.dimText} wrap="truncate-end">
          {error ?? notice ?? 'Input is sent directly to the selected tmux agent pane.'}
        </Text>
      </Box>
    </Box>
  )
}
