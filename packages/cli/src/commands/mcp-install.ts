import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export interface McpInstallOptions {
  url?: string
  marketplace?: string
  var?: string[]
}

function parseVariables(entries: string[] | undefined): Record<string, string> | undefined {
  if (!entries?.length) return undefined
  const variables: Record<string, string> = {}
  for (const entry of entries) {
    const index = entry.indexOf('=')
    if (index <= 0) {
      throw new Error(`Invalid --var entry: ${entry} (expected KEY=VALUE)`)
    }
    const key = entry.slice(0, index).trim()
    if (!key) {
      throw new Error(`Invalid --var entry: ${entry} (expected KEY=VALUE)`)
    }
    variables[key] = entry.slice(index + 1)
  }
  return variables
}

export async function mcpInstallCommand(name: string, options: McpInstallOptions) {
  const client = new DaemonClient(options.url)
  const result = await client.mcpInstall(
    name,
    options.marketplace,
    parseVariables(options.var),
  )
  // Wrap the daemon record in an {ok} envelope so the --json output
  // matches every other mutation command. The text formatter still
  // sees the inner record via `data.result`.
  const envelope = {
    ok: result.installed,
    requested: name,
    ...result,
  }
  output(envelope, (r) => {
    if (r.installed) {
      const renamed = r.serverName !== name
      const head = `${chalk.green('Installed')} MCP server: ${chalk.bold(r.serverName)}`
      const renameHint = renamed
        ? chalk.gray(`  (registered as "${r.serverName}" — use this name with subsequent mcp commands)`)
        : null
      return [
        head,
        renameHint,
        chalk.gray('The daemon will auto-reload the config.'),
      ].filter((line): line is string => line !== null).join('\n')
    }
    return chalk.red(`Failed to install MCP server: ${name}`)
  })
}
