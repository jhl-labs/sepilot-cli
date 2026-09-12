import type { McpLoggingLevel } from '@sepilotd/api-client'
import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

const levels = new Set<McpLoggingLevel>([
  'debug',
  'info',
  'notice',
  'warning',
  'error',
  'critical',
  'alert',
  'emergency',
])

export async function mcpLoggingSetLevelCommand(
  server: string,
  level: string,
  options: { url?: string },
) {
  if (!levels.has(level as McpLoggingLevel)) {
    throw new Error(`Invalid logging level: ${level}`)
  }
  const client = new DaemonClient(options.url)
  const result = await client.mcpSetLoggingLevel(server, level as McpLoggingLevel)
  output(result, (data) => `Set ${chalk.bold(server)} logging level to ${chalk.bold(data.level)}`)
}

export async function mcpLoggingLogsCommand(
  server: string,
  options: { url?: string },
) {
  const client = new DaemonClient(options.url)
  const state = await client.mcpLogs(server)
  output(state, (data) => {
    const header = `level=${data.level ?? 'default'}`
    if (!data.messages.length) return `${header}\nNo MCP log messages.`
    const lines = data.messages.map((message) => {
      const logger = message.logger ? ` ${message.logger}` : ''
      const payload = typeof message.data === 'string' ? message.data : JSON.stringify(message.data)
      return `${message.timestamp} ${message.level}${logger}  ${payload}`
    })
    return [header, ...lines].join('\n')
  })
}
