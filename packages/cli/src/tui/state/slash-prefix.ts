// startStream's first 50 lines were dedicated to handling the
// "command starts with /" prefix: the reserved /undo and /redo
// shortcuts that hit the daemon directly, plus the generic
// resolveUserCommand() round-trip that turns a custom slash command
// into a real prompt + agent override. Pull that out so startStream
// itself is just "send the prompt and stream the response."
//
// Returns either:
//   { applied: false } — the input wasn't a slash command, use it as-is
//   { applied: true; outcome: 'short-circuit' } — a reserved command
//       was handled (or attempted) and the caller should not start a
//       chat run for this input
//   { applied: true; outcome: 'rewritten'; workingContent; resolvedAgent }
//       — the slash was resolved into a fresh prompt that startStream
//         should send instead

import type { Dispatch } from 'react'
import type { ChatAction } from './chat-reducer.js'

// The httpClient surface this helper depends on. Mirrors the subset
// useChat actually uses for its prefix handling so the helper isn't
// pulled into the full DaemonClient type.
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
