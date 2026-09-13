import { Box, Text, useInput, useStdout } from 'ink'
import { useEffect, useRef, useState } from 'react'
import type { WorkItem } from '@sepilotd/api-client'
import { cancelTask, inspectTask, type TasksClient } from '../../../commands/tasks.js'
import { colors } from '../../theme.js'

export function TasksDialog({ client, width, onClose }: { client: TasksClient; width: number; onClose(): void }) {
  const { stdout } = useStdout()
  const visibleRows = Math.max(3, Math.min(10, (stdout.rows || 24) - 12))
  const detailRows = Math.max(2, Math.min(12, (stdout.rows || 24) - visibleRows - 8))
  const [items, setItems] = useState<WorkItem[]>([])
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const selected = Math.max(0, items.findIndex(item => item.key === selectedKey))
  const mounted = useRef(true)
  const [detail, setDetail] = useState('Loading tasks…')
  const [refreshError, setRefreshError] = useState('')
  const [confirm, setConfirm] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    let stopped = false
    mounted.current = true
    let timer: ReturnType<typeof setTimeout>
    const refresh = async () => {
      try {
        const result = await client.listWork()
        if (!stopped) {
          setItems(result.items)
          setSelectedKey(value => result.items.some(item => item.key === value) ? value : result.items[0]?.key ?? null)
          setRefreshError(result.unavailable.length ? `Unavailable sources: ${result.unavailable.join(', ')}` : '')
          setDetail(value => value === 'Loading tasks…' ? 'Select a task to inspect retained evidence.' : value)
        }
      } catch (error) { if (!stopped) setRefreshError(String(error)) }
      if (!stopped) timer = setTimeout(() => { void refresh() }, 3000)
    }
    void refresh()
    return () => { stopped = true; mounted.current = false; clearTimeout(timer) }
  }, [client])
  useInput((input, key) => {
    if (key.escape || input === 'q') { if (confirm) setConfirm(null); else onClose(); return }
    if (busy) return
    if (confirm) {
      if (input === 'y') {
        const target = confirm
        setConfirm(null); setBusy(true)
        void cancelTask(client, target).then(result => { if (mounted.current) setDetail(JSON.stringify(result, null, 2)) }).catch(error => { if (mounted.current) setDetail(String(error)) }).finally(() => { if (mounted.current) setBusy(false) })
      } else setConfirm(null)
      return
    }
    if (key.upArrow) setSelectedKey(items[Math.max(0, selected - 1)]?.key ?? null)
    if (key.downArrow) setSelectedKey(items[Math.min(items.length - 1, selected + 1)]?.key ?? null)
    const item = items[selected]
    if (!item) return
    if (input === 'c') setConfirm(item.key)
    if (key.return) {
      setBusy(true)
      void inspectTask(client, item.key).then(result => { if (mounted.current) setDetail(JSON.stringify(result, null, 2)) }).catch(error => { if (mounted.current) setDetail(String(error)) }).finally(() => { if (mounted.current) setBusy(false) })
    }
  })
  const start = Math.max(0, selected - Math.floor(visibleRows / 2))
  return <Box flexDirection="column" width={width} borderStyle="round" borderColor={colors.primary} paddingX={1}>
    <Text bold color={colors.primary}>Tasks · {items.length} · refresh 3s</Text>
    {items.slice(start, start + visibleRows).map((item, index) => <Text key={item.key} wrap="truncate-end" color={start + index === selected ? colors.primary : colors.text}>{start + index === selected ? '›' : ' '} {item.status} · {item.kind} · {item.title} · {item.id}</Text>)}
    {!items.length && <Text>No tasks or sources still loading.</Text>}
    <Text wrap="truncate-end">{(refreshError || detail).slice(0, 3000).split('\n').slice(0, detailRows).join('\n')}</Text>
    <Text color={confirm ? colors.warning : colors.dimText}>{confirm ? `Stop/pause ${confirm}? y confirms; any other key cancels` : busy ? 'Loading…' : '↑↓ select · enter inspect/logs · c stop/pause · esc return (work continues)'}</Text>
  </Box>
}
