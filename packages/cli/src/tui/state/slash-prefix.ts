import type { Dispatch } from 'react'
import type { ChatAction } from './chat-reducer.js'

export interface SlashPrefixHttpClient {
  undoSession(sessionId: string): Promise<unknown>
  redoSession(sessionId: string): Promise<unknown>
  resolveUserCommand(
    commandId: string,
    args: string,
  ): Promise<{ prompt: string; agent?: string }>
}

const SLASH_COMMAND_REGEX = /^\/([a-z0-9][a-z0-9-_]*)\b\s*([\s\S]*)$/i

function parseSlashCommand(
  content: string,
): { commandId: string; args: string } | null {
  const match = content.match(SLASH_COMMAND_REGEX)
  if (!match) return null
  return { commandId: match[1], args: match[2].trim() }
}

export type SlashPrefixOutcome =
  | { applied: false }
  | { applied: true; outcome: 'short-circuit' }
  | {
      applied: true
      outcome: 'rewritten'
      workingContent: string
      resolvedAgent?: string
    }

export async function applySlashCommandPrefix(opts: {
  content: string
  sessionId: string | null
  httpClient: SlashPrefixHttpClient
  dispatch: Dispatch<ChatAction>
}): Promise<SlashPrefixOutcome> {
  const slash = parseSlashCommand(opts.content)
  if (!slash) return { applied: false }

  const reserved = slash.commandId.toLowerCase()
  if (reserved === 'undo' || reserved === 'redo') {
    if (!opts.sessionId) {
      opts.dispatch({
        type: 'ERROR',
        message: `/${reserved} requires an active session`,
      })
      return { applied: true, outcome: 'short-circuit' }
    }
    try {
      if (reserved === 'undo') {
        await opts.httpClient.undoSession(opts.sessionId)
      } else {
        await opts.httpClient.redoSession(opts.sessionId)
      }
      opts.dispatch({
        type: 'SYSTEM_MESSAGE',
        content: `[/${reserved} applied; reload session view to see effect]`,
      })
    } catch (err) {
      opts.dispatch({
        type: 'ERROR',
        message: `Failed to /${reserved}: ${(err as Error).message}`,
      })
    }
    return { applied: true, outcome: 'short-circuit' }
  }

  try {
    const resolved = await opts.httpClient.resolveUserCommand(
      slash.commandId,
      slash.args,
    )
    return {
      applied: true,
      outcome: 'rewritten',
      workingContent: resolved.prompt,
      resolvedAgent: resolved.agent,
    }
  } catch (err) {
    opts.dispatch({
      type: 'ERROR',
      message: `Failed to resolve /${slash.commandId}: ${(err as Error).message}`,
    })
    return { applied: true, outcome: 'short-circuit' }
  }
}
