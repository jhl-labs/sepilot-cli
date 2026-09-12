import chalk from 'chalk'
import type { DaemonMcpServerStatus } from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export interface McpMutationOptions {
  url?: string
}

export interface McpAddOptions extends McpMutationOptions {
  transport?: string
  command?: string
  arg?: string[]
  env?: string[]
  serverUrl?: string
  header?: string[]
  disabled?: boolean
}

type McpTransport = 'stdio' | 'sse' | 'http'

function parseKeyValueEntries(
  entries: string[] | undefined,
  label: string,
): Record<string, string> {
  const record: Record<string, string> = {}
  for (const entry of entries ?? []) {
    const separator = entry.indexOf('=')
    if (separator <= 0) {
      throw new Error(`Invalid ${label} entry: ${entry}. Expected KEY=VALUE.`)
    }

    const key = entry.slice(0, separator)
    const value = entry.slice(separator + 1)
    record[key] = value
  }

  return record
}

function resolveMcpTransport(options: McpAddOptions): McpTransport {
  const raw = options.transport?.trim().toLowerCase()
  if (!raw) return options.serverUrl ? 'http' : 'stdio'
  if (raw === 'stdio' || raw === 'sse' || raw === 'http') return raw
  throw new Error(`Invalid --transport value: ${options.transport}. Expected stdio, http, or sse.`)
}

function formatServerStatus(server: DaemonMcpServerStatus): string {
  const status = server.status === 'connected'
    ? chalk.green(server.status)
    : server.status === 'error'
      ? chalk.red(server.status)
      : chalk.gray(server.status)

  const command = server.command
    ? [server.command, ...(server.args ?? [])].join(' ').trim()
    : (server.url ?? server.transport)
  const tools = server.toolCount > 0 ? ` tools=${server.toolCount}` : ''
  const detail = server.error ? ` ${chalk.red(server.error)}` : ''
  const security = server.toolManifestStatus === 'changed'
    ? chalk.red(' manifest=changed tools=quarantined')
    : server.securityAlerts?.length
      ? chalk.yellow(` security-alerts=${server.securityAlerts.length}`)
      : ''

  return `  ${server.name.padEnd(18)} ${status.padEnd(18)} ${command}${tools}${security}${detail}`
}

export async function mcpListCommand(options: McpMutationOptions) {
  const client = new DaemonClient(options.url)
  const data = await client.mcpServers()
  output(data, (servers) => {
    if (!servers.length) {
      return 'No MCP servers configured.\nUse `sepilot mcp add <name> --command <cmd>` to add one.'
    }
    return servers.map(formatServerStatus).join('\n')
  })
}

export async function mcpAddCommand(name: string, options: McpAddOptions) {
  if (!/^[a-z0-9][a-z0-9-_.]*$/i.test(name)) {
    throw new Error(
      `Invalid MCP server name: ${name} (use letters, digits, '-', '_', '.' only; must start with a letter or digit)`,
    )
  }
  const transport = resolveMcpTransport(options)
  const remoteUrl = options.serverUrl?.trim()
  const client = new DaemonClient(options.url)
  if (transport === 'stdio') {
    if (!options.command || options.command.trim().length === 0) {
      throw new Error('--command cannot be empty (e.g. --command python --arg server.py)')
    }
    if (remoteUrl) {
      throw new Error('--server-url is only valid with --transport http or --transport sse')
    }
    if ((options.header ?? []).length > 0) {
      throw new Error('--header is only valid with --transport http or --transport sse')
    }
    await client.upsertMcpServer({
      name,
      enabled: options.disabled ? false : true,
      transport,
      command: options.command.trim(),
      args: options.arg ?? [],
      env: parseKeyValueEntries(options.env, '--env'),
    })
  } else {
    if (!remoteUrl) {
      throw new Error('--server-url is required with --transport http or --transport sse')
    }
    if (options.command || (options.arg ?? []).length > 0 || (options.env ?? []).length > 0) {
      throw new Error('--command, --arg, and --env are only valid with --transport stdio')
    }
    await client.upsertMcpServer({
      name,
      enabled: options.disabled ? false : true,
      transport,
      url: remoteUrl,
      headers: parseKeyValueEntries(options.header, '--header'),
    })
  }

  const status = (await client.mcpServers()).find((server) => server.name === name)
  output(
    {
      ok: true,
      name,
      action: 'upserted',
      status,
    },
    (result) => {
      if (!result.status) {
        return `MCP server updated: ${result.name}`
      }
      return `MCP server updated: ${result.name}\n${formatServerStatus(result.status)}`
    },
  )
}

export async function mcpRemoveCommand(name: string, options: McpMutationOptions) {
  const client = new DaemonClient(options.url)
  await client.deleteMcpServer(name)
  output(
    {
      name,
      action: 'removed',
    },
    (result) => `MCP server removed: ${result.name}`,
  )
}

export async function mcpEnableCommand(name: string, options: McpMutationOptions) {
  const client = new DaemonClient(options.url)
  await client.setMcpServerEnabled(name, true)
  const status = (await client.mcpServers()).find((server) => server.name === name)
  output(
    {
      name,
      action: 'enabled',
      status,
    },
    (result) => {
      if (!result.status) {
        return `MCP server enabled: ${result.name}`
      }
      return `MCP server enabled: ${result.name}\n${formatServerStatus(result.status)}`
    },
  )
}

export async function mcpDisableCommand(name: string, options: McpMutationOptions) {
  const client = new DaemonClient(options.url)
  await client.setMcpServerEnabled(name, false)
  const status = (await client.mcpServers()).find((server) => server.name === name)
  output(
    {
      name,
      action: 'disabled',
      status,
    },
    (result) => {
      if (!result.status) {
        return `MCP server disabled: ${result.name}`
      }
      return `MCP server disabled: ${result.name}\n${formatServerStatus(result.status)}`
    },
  )
}

export async function mcpTrustManifestCommand(name: string, options: McpMutationOptions) {
  const client = new DaemonClient(options.url)
  const result = await client.trustMcpServerManifest(name)
  output(
    result,
    (trusted) => `Trusted current MCP tool manifest: ${trusted.name}\n  digest: ${trusted.digest}`,
  )
}
