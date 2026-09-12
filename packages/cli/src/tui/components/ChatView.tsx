import React, { useImperativeHandle, useMemo } from 'react'
import { Box, Text } from 'ink'
import { colors } from '../theme.js'
import { MessageBubble, StreamingBubble } from './MessageBubble.js'
import { ThinkingIndicator } from './ThinkingIndicator.js'
import { BrandMark } from './BrandMark.js'
import type { ChatState, Message } from '../types.js'
import { useScrollable } from '../hooks/useScrollable.js'
import {
  buildTranscriptReaderRows,
  buildTranscriptReaderViewport,
  buildTranscriptViewport,
  estimateMessageLines,
  filterTranscriptMessages,
} from '../utils/transcript.js'

interface ChatViewProps {
  state: ChatState
  height: number
  width: number
  transcriptClearedAt?: number | null
  activeRunLabel?: string | null
}

export interface ChatViewHandle {
  scrollUp: (lines?: number) => void
  scrollDown: (lines?: number) => void
  pageUp: () => void
  pageDown: () => void
  scrollToBottom: () => void
}

export const ChatView = React.forwardRef<ChatViewHandle, ChatViewProps>(function ChatView(
  {
    state,
    height,
    width,
    transcriptClearedAt = null,
    activeRunLabel = null,
  }: ChatViewProps,
  ref,
) {
  const {
    messages,
    currentMessage,
    isStreaming,
    isThinking,
    thinkingText,
    isHydratingSession,
  } = state

  const transcriptMessages = useMemo(
    () => filterTranscriptMessages(messages, transcriptClearedAt),
    [messages, transcriptClearedAt],
  )
  const hasConversationMessages = useMemo(
    () => transcriptMessages.some((message) => message.role !== 'system'),
    [transcriptMessages],
  )
  const showHydrationPlaceholder = (
    isHydratingSession
    && !hasConversationMessages
    && !isStreaming
    && !currentMessage
  )
  const showHydrationBanner = isHydratingSession && !showHydrationPlaceholder
  const transcriptReaderRows = useMemo(
    () => buildTranscriptReaderRows(transcriptMessages, width),
    [transcriptMessages, width],
  )

  const totalLines = useMemo(
    () => transcriptReaderRows.length,
    [transcriptReaderRows],
  )

  const viewportLines = Math.max(1, height - (isStreaming ? 4 : 0))
  const { state: scrollState, scrollUp, scrollDown, pageUp, pageDown, scrollToBottom } = useScrollable(
    totalLines,
    viewportLines,
  )

  useImperativeHandle(
    ref,
    () => ({ scrollUp, scrollDown, pageUp, pageDown, scrollToBottom }),
    [scrollUp, scrollDown, pageUp, pageDown, scrollToBottom],
  )

  const viewport = useMemo(
    () => buildTranscriptViewport({
      messages: transcriptMessages,
      height,
      width,
      isStreaming,
      currentMessage,
      isThinking,
      error: state.error,
      scrollOffset: scrollState.offset,
    }),
    [
      currentMessage,
      height,
      isStreaming,
      isThinking,
      scrollState.offset,
      state.error,
      transcriptMessages,
      width,
    ],
  )
  const readerViewport = useMemo(
    () => buildTranscriptReaderViewport({
      rows: transcriptReaderRows,
      height,
      width,
      isStreaming,
      currentMessage,
      isThinking,
      error: state.error,
      scrollOffset: scrollState.offset,
    }),
    [
      currentMessage,
      height,
      isStreaming,
      isThinking,
      scrollState.offset,
      state.error,
      transcriptReaderRows,
      width,
    ],
  )
  const useReaderMode = (
    scrollState.offset > 0
    || viewport.messages.some((message) => (
      estimateMessageLines(message, width)
      > Math.max(1, height - readerViewport.reservedFooterLines)
    ))
  )

  const hiddenAboveCount = useReaderMode
    ? readerViewport.hiddenAboveCount
    : viewport.hiddenAboveCount
  const hiddenBelowCount = useReaderMode
    ? readerViewport.hiddenBelowCount
    : viewport.hiddenBelowCount
  const visibleReaderRows = readerViewport.rows

  const streamingMaxBodyLines = Math.max(
    2,
    height
    - (useReaderMode ? readerViewport.usedLines : viewport.usedLines)
    - (hiddenAboveCount > 0 ? 1 : 0)
    - (hiddenBelowCount > 0 ? 1 : 0)
    - (isThinking ? 1 : 0)
    - (state.error ? 1 : 0)
    - 2,
  )

  return (
    <Box flexDirection="column" height={height} overflow="hidden">
      {showHydrationPlaceholder && (
        <Box flexDirection="column" justifyContent="center" alignItems="center" flexGrow={1}>
          <Text color={colors.dimText}>Loading session history...</Text>
        </Box>
      )}

      {showHydrationBanner && (
        <Text color={colors.dimText}>Loading session history...</Text>
      )}

      {!hasConversationMessages && !isStreaming && !isHydratingSession && (
        <Box flexDirection="column" justifyContent="center" alignItems="center" flexGrow={1}>
          <BrandMark width={width} />
          <Box height={1} />
          {/* Grouped, scannable quick-reference instead of one wall-of-text
              paragraph (CLI_BACKLOG.md A1). */}
          <Text color={colors.text}>Start a conversation by typing a message below.</Text>
          <Box height={1} />
          <Text color={colors.dimText}>
            <Text color={colors.muted}>Input  </Text>
            / commands · ! shell · !! rerun · # save memory · #? search memory · Tab complete · @path attach
          </Text>
          <Text color={colors.dimText}>
            <Text color={colors.muted}>Edit   </Text>
            {' '}Ctrl+←/→ word move · Ctrl+W delete word · Ctrl+U/K kill line · Ctrl+J newline
          </Text>
          <Text color={colors.dimText}>
            <Text color={colors.muted}>Pickers</Text>
            {'  '}Ctrl+T model · Ctrl+G mode · Ctrl+A autonomy · Ctrl+B activity
          </Text>
          <Text color={colors.dimText}>
            <Text color={colors.muted}>Panels </Text>
            {' '}/usage cost · /memory context · /theme palette · Ctrl+F files
          </Text>
          <Text color={colors.dimText}>
            <Text color={colors.muted}>Copy   </Text>
            {' '}Ctrl+Y latest code block · Ctrl+O full transcript to scrollback
          </Text>
        </Box>
      )}

      {hiddenAboveCount > 0 && (
        <Text color={colors.dimText}>
          ... {hiddenAboveCount} earlier {useReaderMode ? 'line' : 'message'}{hiddenAboveCount === 1 ? '' : 's'} • Ctrl+P/N or wheel to scroll
        </Text>
      )}

      {useReaderMode ? (
        visibleReaderRows.map((row) => (
          <Text
            key={row.key}
            color={colors[row.tone]}
            wrap="truncate-end"
          >
            {row.text || ' '}
          </Text>
        ))
      ) : (
        viewport.messages.map((msg: Message) => (
          <MessageBubble key={msg.id} message={msg} contentWidth={Math.max(20, width - 4)} />
        ))
      )}

      {isStreaming && (
        <>
          <StreamingBubble
            content={currentMessage}
            contentWidth={Math.max(20, width - 2)}
            maxBodyLines={streamingMaxBodyLines}
            activeBadgeLabel={
              state.pendingApproval
                ? 'WAITING FOR APPROVAL'
                : state.pendingQuestions.length > 0
                  ? 'WAITING FOR ANSWER'
                  : 'RUNNING NOW'
            }
            activeStatusLabel={currentMessage ? null : activeRunLabel}
          />
          {isThinking && <ThinkingIndicator text={thinkingText} />}
        </>
      )}

      {state.error && (
        <Box marginLeft={2}>
          <Text color={colors.error}>Error: {state.error}</Text>
        </Box>
      )}

      {hiddenBelowCount > 0 && (
        <Text color={colors.dimText}>
          ... {hiddenBelowCount} newer {useReaderMode ? 'line' : 'message'}{hiddenBelowCount === 1 ? '' : 's'} • Ctrl+P/N or wheel to scroll
        </Text>
      )}
    </Box>
  )
})

ChatView.displayName = 'ChatView'
