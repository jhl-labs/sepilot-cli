import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export interface McpDoctorOptions {
  url?: string
}

type McpServerStatus = Awaited<ReturnType<DaemonClient['mcpServers']>>[number]

interface McpDoctorEntry {
  name: string
  status: McpServerStatus['status']
  enabled: boolean
  connected: boolean
  toolCount: number
  toolManifestStatus?: McpServerStatus['toolManifestStatus']
  securityAlerts: NonNullable<McpServerStatus['securityAlerts']>
  error?: string
  hint?: string
}

function connectorHint(server: McpServerStatus): string | undefined {
  if (server.toolManifestStatus === 'changed') {
    return `Review the advertised tools, then run: sepilot mcp trust-manifest ${server.name}`
  }
  if (server.status === 'connected') {
    return `Inspect tools with: sepilot mcp tools ${server.name}`
  }
  if (server.status === 'disabled') {
    return `Enable with: sepilot mcp enable ${server.name}`
  }

  switch (server.name) {
    case 'notion':
    case 'linear':
    case 'atlassian':
    case 'slack-mcp':
      return 'Complete the browser OAuth/admin approval flow started by mcp-remote, then refresh MCP.'
    case 'github':
      return 'Check that GITHUB_TOKEN exists in the daemon secret vault and has the required repo scopes.'
    case 'google-workspace-remote':
    case 'email-remote':
    case 'zapier-remote':
      return 'Verify the remote MCP endpoint URL and the OAuth/app credentials managed by that endpoint.'
    default:
      return server.error ? 'Inspect daemon logs and run the server command manually if it is stdio-based.' : undefined
  }
}

function summarize(server: McpServerStatus): McpDoctorEntry {
  return {
    name: server.name,
    status: server.status,
    enabled: server.enabled,
    connected: server.connected,
    toolCount: server.toolCount,
    toolManifestStatus: server.toolManifestStatus,
    securityAlerts: server.securityAlerts ?? [],
    error: server.error,
    hint: connectorHint(server),
  }
}

function doctorOk(entry: McpDoctorEntry): boolean {
  return entry.status === 'connected' && entry.connected && entry.toolManifestStatus !== 'changed'
}

function formatEntry(entry: McpDoctorEntry): string {
  const marker = doctorOk(entry)
    ? chalk.green('PASS')
    : entry.status === 'disabled'
      ? chalk.yellow('WARN')
      : chalk.red('FAIL')
  const lines = [
    `${marker} ${chalk.bold(entry.name)} · ${entry.status} · ${entry.toolCount} tools`,
  ]
  if (entry.error) lines.push(`  error: ${entry.error}`)
  if (entry.toolManifestStatus) lines.push(`  manifest: ${entry.toolManifestStatus}`)
  for (const alert of entry.securityAlerts) {
    lines.push(`  ${alert.severity}: ${alert.code} — ${alert.message}`)
  }
  if (entry.hint) lines.push(`  hint: ${entry.hint}`)
  return lines.join('\n')
}

export async function mcpDoctorCommand(
  server: string | undefined,
  options: McpDoctorOptions,
) {
  const client = new DaemonClient(options.url)
  const servers = await client.mcpServers()
  const selected = server
    ? servers.filter((entry) => entry.name === server)
    : servers

  if (server && selected.length === 0) {
    throw new Error(`MCP server not found: ${server}`)
  }

  const entries = selected.map(summarize)
  const result = {
    ok: entries.length > 0 && entries.every(doctorOk),
    servers: entries,
  }

  output(result, (data) => {
    if (data.servers.length === 0) {
      return 'No MCP servers configured. Install one with `sepilot mcp install notion --marketplace builtin`.'
    }
    return data.servers.map(formatEntry).join('\n')
  })
}
