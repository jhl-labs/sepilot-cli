import type React from 'react'
import { Box, Text } from 'ink'
import {
  useMcpStatus,
  type McpMarketplaceInfo,
  type McpMetricsSnapshot,
  type McpServerInfo,
} from '../hooks/useMcpStatus.js'
import { colors, symbols } from '../theme.js'

export interface McpStatusPanelClient {
  mcpServers: () => Promise<McpServerInfo[]>
  mcpMetrics: () => Promise<McpMetricsSnapshot>
  mcpMarketplaceList?: () => Promise<McpMarketplaceInfo[]>
}

interface McpStatusPanelProps {
  client: {
    mcpServers: () => Promise<McpServerInfo[]>
    mcpMetrics: () => Promise<McpMetricsSnapshot>
    mcpMarketplaceList?: () => Promise<McpMarketplaceInfo[]>
  } | null
}

function statusIcon(status: string): string {
  if (status === 'connected') return '●'
  if (status === 'error') return '✗'
  return '○'
}

function statusColor(status: string): string {
  if (status === 'connected') return 'green'
  if (status === 'error') return 'red'
  return 'gray'
}

function avgMs(total: number, calls: number): string {
  if (calls === 0) return '-'
  return `${Math.round(total / calls)}ms`
}

function formatTransport(server: McpServerInfo): string {
  if (server.transport !== 'stdio') return server.transport
  const command = server.command
    ? [server.command, ...(server.args ?? [])].join(' ').trim()
    : 'stdio'
  return command
}

export function McpStatusPanel({ client }: McpStatusPanelProps): React.ReactElement {
  const { servers, metrics, marketplaces, loading, error } = useMcpStatus(client)

  if (loading) {
    return <Box width="100%" minWidth={0}><Text color="gray" wrap="truncate-end">Loading MCP status...</Text></Box>
  }
  if (error) {
    return <Box width="100%" minWidth={0}><Text color="red" wrap="truncate-end">MCP error: {error}</Text></Box>
  }
  if (!servers.length) {
    return <Box width="100%" minWidth={0}><Text color="gray" wrap="truncate-end">No MCP servers configured.</Text></Box>
  }

  return (
    <Box
      flexDirection="column"
      borderStyle="single"
      borderColor="cyan"
      paddingX={1}
      width="100%"
      minWidth={0}
    >
      <Text bold color="cyan">MCP Management</Text>
      <Text color={colors.dimText} wrap="truncate-end">
        {servers.length} servers {symbols.separator} {marketplaces.length} marketplaces
      </Text>
      {marketplaces.length > 0 ? (
        <Text color={colors.dimText} wrap="truncate-end">
          marketplaces {symbols.separator} {marketplaces.map((m) => m.name).join(', ')}
        </Text>
      ) : null}
      {servers.map((s) => {
        const m = metrics.servers[s.name]
        const calls = m?.totalCalls ?? 0
        const errs = m?.errors ?? 0
        const avg = m ? avgMs(m.totalDurationMs, m.totalCalls) : '-'
        const disabledTools = s.disabledTools?.length ?? 0
        return (
          <Box key={s.name} flexDirection="column" width="100%" minWidth={0}>
            <Box gap={1} width="100%" minWidth={0}>
            <Text color={statusColor(s.status)}>{statusIcon(s.status)}</Text>
            <Box flexGrow={1} flexShrink={1} minWidth={0}>
              <Text wrap="truncate-end">{s.name}</Text>
            </Box>
            <Text color="gray">{s.transport}</Text>
            <Text color="gray">tools:{s.toolCount}</Text>
            {disabledTools > 0 ? <Text color="yellow">off:{disabledTools}</Text> : null}
            <Text color="gray">calls:{calls}</Text>
            {errs > 0 && <Text color="red">err:{errs}</Text>}
            <Text color="gray">avg:{avg}</Text>
            </Box>
            <Text color="gray" wrap="truncate-end">  {formatTransport(s)}</Text>
            {s.error ? <Text color="red" wrap="truncate-end">  {s.error}</Text> : null}
          </Box>
        )
      })}
      <Text color={colors.dimText} wrap="truncate-end">
        /mcp add, search, install, enable, disable, remove
      </Text>
      <Text color={colors.dimText} wrap="truncate-end">
        /mcp tools enable|disable, marketplace list|add|remove
      </Text>
    </Box>
  )
}
