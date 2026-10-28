import { Box, Static, Text } from 'ink'
import { useMemo, useRef } from 'react'
import wrapAnsi from 'wrap-ansi'
import type { ActivityItem } from '@sepilotd/api-client'
import type { Message, ToolCallState } from '../../types.js'
import { colors } from '../../theme.js'
import { MessageBubble } from '../../components/MessageBubble.js'
import { BrandMark } from '../../components/BrandMark.js'
import { ToolCallView } from '../../components/ToolCall.js'
import { createCommitLog, selectCommits } from '../runtime/commit-log.js'
import { conversationPresentation } from '../runtime/conversation-presentation.js'
import { toolDisplayName } from '../../utils/tooling.js'

export const LIVE_REGION_MAX_ROWS = 12
const MAX_VISIBLE_TOOLS = 3
const MAX_VISIBLE_DELEGATES = 3

export function selectLiveDelegates(activities: readonly ActivityItem[] | undefined): Array<{ id: string; label: string; detail: string }> {
  if (!activities?.length) return []
  const latest = new Map<string, ActivityItem>()
  for (const activity of activities) {
    if (activity.kind !== 'subagent' || !activity.meta) continue
    latest.set(activity.meta, activity)
  }
  return [...latest.values()]
    .filter((activity) => activity.status === 'running')
    .slice(-MAX_VISIBLE_DELEGATES)
    .map((activity) => ({
      id: activity.meta!,
      label: activity.label.replace(/^subagent(?: · )?/u, '') || 'subagent',
      detail: activity.detail,
    }))
}

export interface ConversationProps {
  messages: Message[]
  isStreaming: boolean
  width: number
  runningTools: ToolCallState[]
  compactTools?: boolean
  showBrandMark?: boolean
  activityFrame?: string | null
  activities?: readonly ActivityItem[]
  suspended?: boolean
  maxLiveRows?: number
}

export function Conversation({
  messages,
  isStreaming,
  width,
  runningTools,
  activities,
  compactTools = true,
  showBrandMark = true,
  activityFrame = null,
  suspended = false,
  maxLiveRows = LIVE_REGION_MAX_ROWS,
}: ConversationProps) {
  const logRef = useRef(createCommitLog())
  const selection = useMemo(() => suspended
    ? {
        commits: conversationPresentation(messages, isStreaming, logRef.current.committedRepeatCounts).filter(({ id }) => logRef.current.committed.has(id)),
        live: [],
      }
    : selectCommits(logRef.current, messages, isStreaming), [messages, isStreaming, suspended])
  const { commits, live } = selection
  const rowLimit = Math.max(0, Math.min(LIVE_REGION_MAX_ROWS, maxLiveRows))
  const representedToolIds = new Set(live.slice(-LIVE_REGION_MAX_ROWS).flatMap(({ toolCall }) => toolCall ? [toolCall.id] : []))
  const unrepresentedRunningTools = runningTools.filter(({ id, status }) => status === 'running' && !representedToolIds.has(id))
  const visibleTools = suspended
    ? []
    : unrepresentedRunningTools.slice(0, rowLimit > 0 ? MAX_VISIBLE_TOOLS : 0)
  const hiddenToolCount = Math.max(
    0,
    unrepresentedRunningTools.length - MAX_VISIBLE_TOOLS,
  )
  const delegates = suspended || !isStreaming ? [] : selectLiveDelegates(activities).slice(0, Math.max(0, rowLimit - (visibleTools.length > 0 ? 1 : 0)))
  const liveRowBudget = rowLimit - (visibleTools.length > 0 ? 1 : 0) - delegates.length
  const visibleLive = liveRowBudget > 0 ? live.slice(-liveRowBudget) : []
  const messageRowBudget = Math.max(1, Math.floor(liveRowBudget / Math.max(1, visibleLive.length)))
  const toolSpinner = activityFrame ?? '[-]'

  return (
    <Box flexDirection="column" width={width}>
      {!suspended && showBrandMark && rowLimit > 0 && messages.length === 0 && !isStreaming && visibleTools.length === 0 ? (
        <Box height={rowLimit} overflowY="hidden"><BrandMark width={width} /></Box>
      ) : null}
      <Static items={commits}>
        {(item) => <MessageBlock key={item.id} message={item} width={width} compactTools={compactTools} />}
      </Static>
      {live.length > 0 || visibleTools.length > 0 || delegates.length > 0 ? (
        <Box flexDirection="column">
          {visibleLive.map((item) => (
            <LiveMessageBlock
              key={item.id}
              message={item}
              width={width}
              rowBudget={messageRowBudget}
              isStreaming={isStreaming}
              compactTools={compactTools}
            />
          ))}
          {visibleTools.length > 0 ? (
            <Text color={colors.dimText}>
              {`  ${toolSpinner} executing · ${visibleTools.map(({ name }) => toolDisplayName(name)).join(' · ')}${
                hiddenToolCount > 0 ? ` · +${hiddenToolCount}` : ''
              }`}
            </Text>
          ) : null}
          {delegates.map((delegate) => (
            <Text key={delegate.id} color={colors.dimText} wrap="truncate-end">
              {`    ↳ ${delegate.label} · ${delegate.detail}`}
            </Text>
          ))}
        </Box>
      ) : null}
    </Box>
  )
}

function LiveMessageBlock({
  message,
  width,
  rowBudget,
  isStreaming,
  compactTools,
}: {
  message: Message
  width: number
  rowBudget: number
  isStreaming: boolean
  compactTools: boolean
}) {
  if (message.role === 'tool' && message.toolCall) {
    const needsAttention = message.toolCall.status === 'error' || message.toolCall.status === 'pending'
    const compact = compactTools && !needsAttention
    return (
      <ToolCallView
        tool={message.toolCall}
        indent={0}
        compact={compact}
        forceExpanded={!compactTools}
        maxRows={compact ? 1 : isStreaming ? rowBudget : undefined}
      />
    )
  }

  if (message.variant === 'progress') {
    return <Text color={colors.dimText} wrap="truncate-end">{`  ${message.content.split('\n')[0]}`}</Text>
  }

  if (!isStreaming && message.variant !== 'thinking') {
    return <MessageBlock message={message} width={width} compactTools />
  }

  const thinking = message.variant === 'thinking'
  const marker = thinking ? '' : `${roleMarker(message.role)} `
  const wrapped = wrapAnsi(`${marker}${message.content}`, Math.max(1, width), {
    hard: true,
    trim: false,
  }).split('\n')
  const budget = thinking ? Math.min(4, rowBudget) : rowBudget
  const visible = wrapped.slice(-Math.max(1, budget - (thinking ? 1 : 0)))
  if (thinking && budget > 1) visible.unshift('◐ Thinking · Ctrl+O expand')

  return (
    <Text wrap="truncate-end" color={message.variant === 'thinking' ? colors.dimText : message.role === 'user' ? colors.primary : colors.text}>
      {visible.join('\n')}
    </Text>
  )
}

function MessageBlock({ message, width, compactTools }: { message: Message; width: number; compactTools: boolean }) {
  const displayMessage = !compactTools && message.toolCall
    ? { ...message, toolCall: { ...message.toolCall, collapsed: false } }
    : message
  return <MessageBubble message={displayMessage} compactTools={compactTools} contentWidth={Math.max(12, width - 4)} />
}

function roleMarker(role: Message['role']): string {
  if (role === 'user') return '›'
  if (role === 'assistant') return '│'
  return '·'
}
