import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { tildify } from '../utils/path-display.js'
import {
  buildPlaywrightMcpServerConfig,
  type PlaywrightMcpConfigResult,
  type PlaywrightMcpEnvironment,
  type PlaywrightMcpOptions,
} from '../utils/playwright-mcp.js'

export interface McpPlaywrightOptions extends PlaywrightMcpOptions {
  url?: string
  sandbox?: boolean
}

interface McpPlaywrightCommandResult extends PlaywrightMcpConfigResult {
  ok: boolean
  status?: Awaited<ReturnType<DaemonClient['mcpServers']>>[number]
}

function formatPlaywrightResult(result: McpPlaywrightCommandResult): string {
  const requested = result.requestedMode === result.resolvedMode
    ? result.resolvedMode
    : `${result.resolvedMode} (${result.requestedMode})`
  const profile = result.isolated
    ? 'isolated'
    : result.userDataDir
      ? tildify(result.userDataDir)
      : 'default'
  const status = result.status
    ? `${result.status.status}${result.status.error ? `: ${result.status.error}` : ''}`
    : 'pending daemon reload'

  return [
    `${chalk.green('Configured')} Playwright MCP server: ${chalk.bold(result.server.name)}`,
    `  mode: ${requested}`,
    `  browser: ${result.browser}`,
    `  profile: ${profile}`,
    `  status: ${status}`,
  ].join('\n')
}

export async function mcpPlaywrightCommand(
  options: McpPlaywrightOptions,
  environment?: PlaywrightMcpEnvironment,
) {
  const config = buildPlaywrightMcpServerConfig(
    {
      ...options,
      noSandbox: options.noSandbox ?? options.sandbox === false,
    },
    environment,
  )
  const client = new DaemonClient(options.url)
  await client.upsertMcpServer(config.server)
  const status = (await client.mcpServers()).find((server) => server.name === config.server.name)

  output(
    {
      ok: true,
      ...config,
      status,
    },
    formatPlaywrightResult,
  )
}
