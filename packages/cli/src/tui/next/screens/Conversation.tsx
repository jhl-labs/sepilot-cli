import { Box, Static, Text } from 'ink'
import { useRef } from 'react'
import wrapAnsi from 'wrap-ansi'
import type { Message, ToolCallState } from '../../types.js'
import { colors } from '../../theme.js'
import { MessageBubble } from '../../components/MessageBubble.js'
import { BrandMark } from '../../components/BrandMark.js'
import { ToolCallView } from '../../components/ToolCall.js'
import { createCommitLog, selectCommits } from '../runtime/commit-log.js'

export const LIVE_REGION_MAX_ROWS = 12
const MAX_VISIBLE_TOOLS = 3

export interface ConversationProps {
  messages: Message[]
  isStreaming: boolean
  width: number
  runningTools: ToolCallState[]
  compactTools?: boolean
  showBrandMark?: boolean
  /** Shared run activity frame; avoids a second tool-only animation timer. */
  activityFrame?: string | null
  /**
   * Keep already committed scrollback mounted, but freeze the mutable tail.
   * Full-height inline dialogs use this so Ink never renders a frame taller
   * than the terminal and promotes the dialog itself into host scrollback.
   */
  suspended?: boolean
}

export function Conversation({
  messages,
  isStreaming,
  width,
  runningTools,
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
  const liveRowBudget = LIVE_REGION_MAX_ROWS - (visibleTools.length > 0 ? 1 : 0)
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
      {live.length > 0 || visibleTools.length > 0 ? (
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

  const marker = message.variant === 'thinking' ? '◐ Thinking · ' : `${roleMarker(message.role)} `
  const wrapped = wrapAnsi(`${marker}${message.content}`, Math.max(1, width), {
    hard: true,
    trim: false,
  }).split('\n')
  const visible = wrapped.slice(-Math.max(1, rowBudget))

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
