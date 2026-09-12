import React from 'react'
import { Box, Text } from 'ink'
import { colors, symbols } from '../theme.js'
import { renderMarkdown } from '../renderer/markdown.js'
import { ToolCallView } from './ToolCall.js'
import type { Message } from '../types.js'
import { tailWrappedText } from '../utils/transcript.js'

interface MessageBubbleProps {
  message: Message
  compactTools?: boolean
  /**
   * When provided, assistant markdown is pre-wrapped to this column budget
   * (CJK-width aware, hanging indents) and rendered line-by-line so Ink does
   * not re-wrap and break list/quote indentation.
   */
  contentWidth?: number
}

export type SystemLineTone =
  | 'heading'
  | 'success'
  | 'warning'
  | 'error'
  | 'command'
  | 'metadata'
  | 'muted'
  | 'body'

export function shouldUseReadableSystemBody(content: string): boolean {
  return (
    content.includes('\n') ||
    /^(Installed skills \(|Skill catalog results for |Installed \d+ skill|No installed skills found\.)/.test(
      content,
    )
  )
}

export function classifySystemLine(line: string, index: number): SystemLineTone {
  const trimmed = line.trim()
  if (!trimmed) return 'muted'
  if (index === 0 && /:($|\s)|\(\d+\):$/.test(trimmed)) return 'heading'
  if (
    /^(Usage:|Installed skills \(|Skill catalog results for |Built-in active \(|Built-in opt-in \(|Installed from sources \(|Local \/ custom \(|Development \(|Operations \(|Connectors & automation \(|Research & data \(|Personal productivity \(|Work collaboration \(|Writing & docs \(|General \(|Paired |Available |Configured |Health |Status\b)/.test(
      trimmed,
    )
  ) {
    return 'heading'
  }
  if (/^(✓|✔|Success\b|OK\b|- ✓|enabled\b|active\b)/i.test(trimmed)) return 'success'
  if (/^(✗|✘|Error\b|Failed\b|Cannot\b|Issue\b|- ✗|disabled\b)/i.test(trimmed)) return 'error'
  if (/^note:/i.test(trimmed)) return 'metadata'
  if (/^(Warning\b|Next\b|Hint\b|Note\b|No\b|Run\b|Use\b|Toggle\b|…|\.\.\.)/i.test(trimmed)) {
    return 'warning'
  }
  if (/^(\/[a-z]|\S+\s+\/[a-z]|[`']?sepilot\b|install:|source:)/i.test(trimmed)) {
    return 'command'
  }
  if (
    /^(-\s*)?(id|labels|category|note|source catalog|source|path|url|status|provider|model|session|workspace|created|updated|last|next|retry|attempt|header|events|ips):/i.test(
      trimmed,
    )
  ) {
    return 'metadata'
  }
  if (/^\[[^\]]+\]$|^[A-Z_]+=/i.test(trimmed)) return 'metadata'
  return 'body'
}

function systemLineColor(tone: SystemLineTone): string {
  switch (tone) {
    case 'heading':
    case 'command':
      return colors.info
    case 'success':
      return colors.success
    case 'warning':
      return colors.warning
    case 'error':
      return colors.error
    case 'metadata':
      return colors.muted
    case 'muted':
      return colors.dimText
    case 'body':
      return colors.text
  }
}

function SystemMessageBody({ content }: { content: string }) {
  if (!shouldUseReadableSystemBody(content)) {
    return <Text color={colors.dimText}>{content}</Text>
  }

  return (
    <Text>
      {content.split('\n').map((line, index) => {
        const tone = classifySystemLine(line, index)
        return (
          <React.Fragment key={`system-line-${index}`}>
            {index > 0 ? '\n' : null}
            <Text color={systemLineColor(tone)} bold={tone === 'heading'}>
              {line || ' '}
            </Text>
          </React.Fragment>
        )
      })}
    </Text>
  )
}

function AssistantMarkdownBody({
  content,
  contentWidth,
}: {
  content: string
  contentWidth?: number
}) {
  if (contentWidth === undefined) {
    return <Text>{renderMarkdown(content)}</Text>
  }

  const lines = renderMarkdown(content, { width: contentWidth }).split('\n')
  return (
    <>
      {lines.map((line, index) => (
        <Text key={`md-${index}`} wrap="truncate-end">
          {line || ' '}
        </Text>
      ))}
    </>
  )
}

export const MessageBubble = React.memo(function MessageBubble({
  message,
  compactTools = false,
  contentWidth,
}: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const isSystem = message.role === 'system'
  const isTool = message.role === 'tool'
  const isThinking = message.variant === 'thinking'
  const normalizedContent = (message.content ?? '').normalize('NFC')
  const isContext = isSystem && /^Relevant context\b/.test(normalizedContent)
  const accentColor = isContext
    ? colors.info
    : isSystem
      ? colors.warning
      : isTool
        ? colors.info
        : isUser
          ? colors.primary
          : colors.success
  const label = isContext
    ? 'Context'
    : isThinking
      ? 'Thinking'
    : isSystem
      ? 'System'
      : isTool
        ? 'Tool'
        : isUser
          ? 'You'
          : 'Assistant'

  if (isTool && message.toolCall) {
    return (
      <Box flexDirection="column" marginBottom={1}>
        <ToolCallView tool={message.toolCall} indent={0} compact={compactTools} />
      </Box>
    )
  }

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Box gap={1}>
        <Text color={accentColor} bold>
          {isSystem
            ? symbols.thinking
            : isTool
              ? symbols.pending
              : isUser
                ? symbols.user
                : symbols.assistant}
        </Text>
        <Text color={accentColor} bold>
          {label}
        </Text>
      </Box>
      <Box marginLeft={2} flexDirection="column">
        {isSystem ? (
          <SystemMessageBody content={normalizedContent} />
        ) : isUser ? (
          <>
            {normalizedContent ? <Text>{normalizedContent}</Text> : null}
            {(message.attachments?.length ?? 0) > 0 && (
              <Box flexDirection="column" marginTop={normalizedContent ? 1 : 0}>
                <Text color={colors.dimText}>Attached files</Text>
                {message.attachments?.map((attachment) => (
                  <Text key={`${attachment.path}:${attachment.id ?? 'local'}`} color={colors.info}>
                    - {attachment.path}
                  </Text>
                ))}
              </Box>
            )}
          </>
        ) : (
          <>
            <AssistantMarkdownBody content={normalizedContent} contentWidth={contentWidth} />
            {(message.citations?.length ?? 0) > 0 && (
              <Box flexDirection="column" marginTop={1}>
                <Text color={colors.info}>Sources used</Text>
                {message.citations?.map((citation) => (
                  <Box key={`${message.id}:${citation.id}`} flexDirection="column" marginTop={1}>
                    <Text color={colors.info}>- {citation.citationLabel}</Text>
                    <Text color={colors.dimText}>{citation.snippet}</Text>
                  </Box>
                ))}
              </Box>
            )}
          </>
        )}
      </Box>
    </Box>
  )
})

interface StreamingBubbleProps {
  content: string
  contentWidth: number
  maxBodyLines?: number
  activeBadgeLabel?: string | null
  activeStatusLabel?: string | null
}

export const StreamingBubble = React.memo(function StreamingBubble({
  content,
  contentWidth,
  maxBodyLines,
  activeBadgeLabel = null,
  activeStatusLabel = null,
}: StreamingBubbleProps) {
  if (!content && !activeStatusLabel) return null

  const rendered = renderMarkdown(content, { width: contentWidth })
  const body = tailWrappedText(rendered, contentWidth, maxBodyLines ?? Number.MAX_SAFE_INTEGER)

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Box justifyContent={activeBadgeLabel ? 'space-between' : 'flex-start'}>
        <Box gap={1} flexShrink={0}>
          <Text color={colors.success} bold>
            {symbols.assistant}
          </Text>
          <Text color={colors.success} bold>
            Assistant
          </Text>
        </Box>
        {activeBadgeLabel && (
          <Box flexGrow={1} justifyContent="flex-end" minWidth={0} overflow="hidden">
            <Text color={colors.warning} bold wrap="truncate-start">
              {activeBadgeLabel}
            </Text>
          </Box>
        )}
      </Box>
      <Box marginLeft={2}>
        <Box flexDirection="column">
          {content ? (
            <>
              {body.omitted > 0 && (
                <Text color={colors.dimText}>
                  ... {body.omitted} earlier line{body.omitted === 1 ? '' : 's'}
                </Text>
              )}
              {body.visible.map((line, index) => (
                <Text key={`stream-${index}`}>{line || ' '}</Text>
              ))}
            </>
          ) : null}
          {!content && activeStatusLabel && <Text color={colors.dimText}>{activeStatusLabel}</Text>}
        </Box>
      </Box>
    </Box>
  )
})
