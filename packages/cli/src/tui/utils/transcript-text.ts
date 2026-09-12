/**
 * Render the conversation as a single plain-text block for Ctrl+O terminal
 * scrollback export and for clipboard export.
 * Structural message shape so this stays decoupled from the exact `Message`
 * type the caller holds.
 */
export interface CopyableMessage {
  role: string
  content?: string
  toolName?: string
  toolResult?: string
}

function roleHeader(message: CopyableMessage): string {
  switch (message.role) {
    case 'user':
      return '## You'
    case 'assistant':
      return '## Assistant'
    case 'tool':
      return `## Tool · ${message.toolName ?? 'tool'}`
    case 'context':
      return '## Context'
    default:
      return `## ${message.role}`
  }
}

// Roles that are part of the user-visible conversation. Excludes:
//   - `system`  : CLI-local notices ("Compacted session", "Theme → …",
//                 /help output, error banners, planner self-critique notes,
//                 [Compressed history] from semantic compression). The dump
//                 must not lead with the multi-KB system prompt either.
//   - `context` : auto-injected memory/RAG citations (already shown as
//                 footnotes in the surface; including them again would
//                 bloat the dump with content the user didn't write).
//   - `todo`    : surface-only artifact for the planner todo panel.
const COPYABLE_ROLES = new Set(['user', 'assistant', 'tool'])

// Collapse runs of 3+ newlines to a paragraph break. The system prompt and
// long assistant outputs accumulate stretches of blank lines when paragraphs
// are joined; left as-is they pad the scrollable viewport with empty rows.
function normaliseBlankRuns(value: string): string {
  return value.replace(/\n{3,}/g, '\n\n')
}

export function formatConversationForCopy(
  messages: readonly CopyableMessage[],
  sessionId?: string | null,
): string {
  const blocks: string[] = [
    sessionId ? `# Conversation — session ${sessionId.slice(0, 8)}` : '# Conversation',
  ]

  for (const message of messages) {
    if (!COPYABLE_ROLES.has(message.role)) {
      continue
    }
    const content = normaliseBlankRuns(message.content?.trim() ?? '')
    const result = message.role === 'tool'
      ? normaliseBlankRuns(message.toolResult?.trim() ?? '')
      : ''
    if (!content && !result) {
      continue
    }

    const parts: string[] = [roleHeader(message)]
    if (content) {
      parts.push(content)
    }
    if (result) {
      parts.push('— result —', result)
    }
    blocks.push(parts.join('\n'))
  }

  return `${blocks.join('\n\n')}\n`
}

export function formatTerminalTranscriptDump(
  transcript: string,
  messageCount: number,
): string {
  const body = transcript.endsWith('\n') ? transcript : `${transcript}\n`
  return [
    '',
    body.trimEnd(),
    '',
    `-- transcript · ${messageCount} messages --`,
    'Use terminal scrollback/selection to copy. Press Esc or Ctrl+O to return to sepilot.',
    '',
  ].join('\n')
}
