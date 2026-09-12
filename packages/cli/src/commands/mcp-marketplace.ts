import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export interface McpMarketplaceOptions {
  url?: string
}

export async function mcpMarketplaceListCommand(options: McpMarketplaceOptions) {
  const client = new DaemonClient(options.url)
  const data = await client.mcpMarketplaceList()
  output(data, (list) => {
    if (!list.length) return 'No MCP marketplaces registered.'
    return list
      .map((m) => {
        const synced = m.lastSync
          ? chalk.green(`synced ${m.lastSync}`)
          : chalk.gray('never synced')
        return `  ${chalk.bold(m.name)}  ${m.url}  ${synced}`
      })
      .join('\n')
  })
}

export async function mcpMarketplaceAddCommand(
  name: string,
  url: string,
  options: McpMarketplaceOptions,
) {
  if (!/^[a-z0-9][a-z0-9-_.]*$/i.test(name)) {
    throw new Error(
      `Invalid MCP marketplace name: ${name} (use letters, digits, '-', '_', '.' only; must start with a letter or digit)`,
    )
  }
  const client = new DaemonClient(options.url)
  const result = await client.mcpMarketplaceAdd(name, url)
  output({ ok: true, ...result }, (r) => `Added MCP marketplace: ${chalk.bold(r.name)} — ${r.url}`)
}

export async function mcpMarketplaceRemoveCommand(
  name: string,
  options: McpMarketplaceOptions,
) {
  const client = new DaemonClient(options.url)
  const result = await client.mcpMarketplaceRemove(name)
  output({ ok: result.removed, name, ...result }, (r) =>
    r.removed
      ? chalk.green(`Removed MCP marketplace: ${chalk.bold(name)}`)
      : chalk.yellow(`MCP marketplace "${name}" not found.`),
  )
  if (!result.removed) process.exit(1)
}
