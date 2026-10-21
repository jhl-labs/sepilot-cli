import chalk from 'chalk'
import type { DaemonRoleBot } from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export interface BotsListClient {
  bots(): Promise<DaemonRoleBot[]>
}

function stageLabel(bot: DaemonRoleBot): string {
  const on = Object.entries(bot.stages).filter(([, enabled]) => enabled).map(([stage]) => stage)
  return on.length === 0 ? 'no control stages' : on.join(', ')
}

export async function botsCommandImpl(client: BotsListClient): Promise<void> {
  const bots = await client.bots()
  output(bots, (list) => {
    if (list.length === 0) return 'No role bots registered.'
    const byCategory = new Map<string, DaemonRoleBot[]>()
    for (const bot of list) {
      const group = byCategory.get(bot.category) ?? []
      group.push(bot)
      byCategory.set(bot.category, group)
    }
    const lines: string[] = []
    for (const [category, group] of byCategory) {
      lines.push(chalk.bold(category))
      for (const bot of group) {
        const flags = [
          bot.engine,
          bot.routable ? '' : chalk.gray('explicit only'),
          bot.source === 'user' ? chalk.cyan('user') : '',
        ].filter(Boolean).join(' · ')
        lines.push(`  ${chalk.green(bot.id.padEnd(22))} ${bot.description}`)
        lines.push(`  ${''.padEnd(22)} ${chalk.gray(`${flags} · ${stageLabel(bot)}${bot.skills.length ? ` · skills: ${bot.skills.join(', ')}` : ''}`)}`)
      }
      lines.push('')
    }
    lines.push(chalk.gray('Use: sepilot ask --bot <id> "..."   or   sepilot --bot <id>'))
    return lines.join('\n')
  })
}

export async function botsCommand(options: { url?: string } = {}): Promise<void> {
  await botsCommandImpl(new DaemonClient(options.url))
}
