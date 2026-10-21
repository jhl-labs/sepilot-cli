import { Box, Static, Text } from 'ink'
import { useRef } from 'react'
import wrapAnsi from 'wrap-ansi'
import type { ActivityItem } from '@sepilotd/api-client'
import type { Message, ToolCallState } from '../../types.js'
import { colors } from '../../theme.js'
import { MessageBubble } from '../../components/MessageBubble.js'
import { BrandMark } from '../../components/BrandMark.js'
import { ToolCallView } from '../../components/ToolCall.js'
import { createCommitLog, selectCommits } from '../runtime/commit-log.js'

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
}: ConversationProps) {
  const logRef = useRef(createCommitLog())
  const selection = suspended
    ? {
        commits: messages.filter(({ id }) => logRef.current.committed.has(id)),
        live: [],
      }
    : selectCommits(logRef.current, messages, isStreaming)
  const { commits, live } = selection
  const visibleTools = suspended
    ? []
    : runningTools.filter(({ status }) => status === 'running').slice(0, 3)
  const hiddenToolCount = Math.max(
    0,
    runningTools.filter(({ status }) => status === 'running').length - MAX_VISIBLE_TOOLS,
  )
  const delegates = suspended || !isStreaming ? [] : selectLiveDelegates(activities)
  const liveRowBudget = LIVE_REGION_MAX_ROWS - (visibleTools.length > 0 ? 1 : 0) - delegates.length
  const visibleLive = live.slice(-liveRowBudget)
  const messageRowBudget = Math.max(1, Math.floor(liveRowBudget / Math.max(1, visibleLive.length)))
  const toolSpinner = activityFrame ?? '[-]'
  const latestLiveToolId = [...live]
    .reverse()
    .find(({ role, toolCall }) => role === 'tool' && toolCall)?.id

  return (
    <Box flexDirection="column" width={width}>
      {!suspended && showBrandMark && messages.length === 0 && !isStreaming && visibleTools.length === 0 ? <BrandMark width={width} /> : null}
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
              forceExpandedTool={item.id === latestLiveToolId}
            />
          ))}
          {visibleTools.length > 0 ? (
            <Text color={colors.dimText}>
              {`  ${toolSpinner} executing · ${visibleTools.map(({ name }) => name).join(' · ')}${
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
  forceExpandedTool,
}: {
  message: Message
  width: number
  rowBudget: number
  isStreaming: boolean
  forceExpandedTool: boolean
}) {
  if (message.role === 'tool' && message.toolCall) {
    return (
        <ToolCallView
          tool={message.toolCall}
          indent={0}
          compact={!forceExpandedTool}
          forceExpanded={forceExpandedTool}
          maxRows={isStreaming ? rowBudget : undefined}
        />
    )
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
  return <MessageBubble message={message} compactTools={compactTools} contentWidth={Math.max(12, width - 4)} />
}

function roleMarker(role: Message['role']): string {
  if (role === 'user') return '›'
  if (role === 'assistant') return '│'
  return '·'
}
