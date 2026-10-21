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

const COPYABLE_ROLES = new Set(['user', 'assistant', 'tool'])

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
