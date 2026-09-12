import chalk from 'chalk'
import { formatStateBoardText } from '@sepilotd/presentation'
import { DaemonClient } from '../client/http.js'
import { ensureDaemon } from '../client/ensure-daemon.js'
import { getOutputFormat } from '../output/formatter.js'

export interface StateCommandOptions {
  url?: string
  json?: boolean
}

/**
 * `sepilot state [sessionId]` — render the agent state board for a run.
 * Without a sessionId, the most recent session is used. `--json` (or the
 * global `--json`/`--output-format json`) prints the raw board response.
 */
export async function stateCommand(
  sessionId: string | undefined,
  options: StateCommandOptions = {},
): Promise<void> {
  const client = new DaemonClient(options.url)

  try {
    await ensureDaemon(client, { url: options.url, quiet: true })
  } catch {
    if (options.json || getOutputFormat() === 'json') {
      console.log(JSON.stringify({ ok: false, error: 'daemon-unreachable' }, null, 2))
    } else {
      console.error(chalk.red('Cannot connect to sepilotd.'))
      console.error(chalk.gray('Is the daemon running? Start with: sepilot start'))
    }
    process.exit(1)
  }

  let targetSessionId = sessionId
  if (!targetSessionId) {
    const list = await client.sessions()
    targetSessionId = list.items[0]?.id
    if (!targetSessionId) {
      if (options.json || getOutputFormat() === 'json') {
        console.log(JSON.stringify({ board: null, source: 'none', updatedAt: null }, null, 2))
      } else {
        console.log(chalk.gray('No sessions found.'))
      }
      return
    }
  }

  const response = await client.stateBoard(targetSessionId)

  if (options.json || getOutputFormat() === 'json') {
    console.log(JSON.stringify(response, null, 2))
    return
  }

  if (!response.board || response.source === 'none') {
    console.log(chalk.gray(`No state board for session ${targetSessionId}.`))
    return
  }

  const text = formatStateBoardText(response.board)
  console.log(text ?? chalk.gray('State board is empty.'))
}
