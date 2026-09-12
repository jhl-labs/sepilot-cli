import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { formatQueuedSteerNote, submitSteer } from '../steer-shared.js'

export interface SteerCommandOptions {
  url?: string
  kind?: string
}

function parseKind(raw: string | undefined): 'instruction' | 'question' | undefined {
  if (!raw) return undefined
  const trimmed = raw.trim().toLowerCase()
  return trimmed === 'question' ? 'question' : 'instruction'
}

export async function steerCommand(
  sessionId: string,
  messageParts: string[],
  options: SteerCommandOptions = {},
): Promise<void> {
  const message = messageParts.join(' ').trim()
  if (!message) {
    console.error(chalk.red('Usage: sepilot steer <sessionId> <message...>'))
    process.exit(1)
  }

  const client = new DaemonClient(options.url)
  const result = await submitSteer(client, sessionId, message, parseKind(options.kind))

  if (result.ok) {
    output(
      {
        noteId: result.noteId,
        pendingSteeringNoteCount: result.pendingSteeringNoteCount,
      },
      () =>
        chalk.green(
          `Steering note ${formatQueuedSteerNote(result.noteId, result.pendingSteeringNoteCount)}`,
        ),
    )
    return
  }

  if (result.noActiveRun) {
    console.error(chalk.red(result.guidance))
    process.exit(1)
  }

  console.error(chalk.red(`Failed to steer ${sessionId}: ${result.message}`))
  process.exit(1)
}
