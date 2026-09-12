import type { DaemonMcpServerStatus } from '@sepilotd/api-client'
import { Box, Text, useInput } from 'ink'
import { useMemo, useState } from 'react'
import { ControlSafeTextInput } from '../../components/ControlSafeTextInput.js'
import { colors } from '../../theme.js'

export interface McpDialogProps {
  servers: DaemonMcpServerStatus[]
  initialQuery: string
  loading: boolean
  error: string | null
  width: number
  onClose(): void
}

function statusIcon(status: DaemonMcpServerStatus['status']): string {
  if (status === 'connected') return '●'
  if (status === 'error') return '✗'
  return '○'
}

export function McpDialog({ servers, initialQuery, loading, error, width, onClose }: McpDialogProps) {
  const [query, setQuery] = useState(initialQuery)
  const normalized = query.trim().toLowerCase()
  const visible = useMemo(() => servers.map((server) => ({
    server,
    tools: (server.allTools ?? server.tools).filter((tool) => !normalized || `${server.name} ${tool}`.toLowerCase().includes(normalized)),
  })).filter(({ server, tools }) => !normalized || server.name.toLowerCase().includes(normalized) || tools.length > 0), [normalized, servers])

  useInput((_, key) => {
    if (key.escape) onClose()
  })

  return (
    <Box flexDirection="column" width={width} borderStyle="round" borderColor={colors.info} paddingX={1}>
      <Text color={colors.info} bold>{`MCP · ${servers.length} servers · ${servers.reduce((sum, server) => sum + server.toolCount, 0)} tools`}</Text>
      <Box>
        <Text color={colors.dimText}>filter </Text>
        <ControlSafeTextInput value={query} onChange={setQuery} placeholder="server or tool…" focus={!loading} />
      </Box>
      {loading ? <Text color={colors.dimText}>Loading MCP status…</Text> : null}
      {error ? <Text color={colors.error}>{error}</Text> : null}
      {!loading && !error && servers.length === 0 ? <Text color={colors.dimText}>No MCP servers configured. Use `sepilot mcp add`.</Text> : null}
      {!loading && !error ? visible.slice(0, 10).map(({ server, tools }) => (
        <Box key={server.name} flexDirection="column">
          <Text color={server.status === 'connected' ? colors.success : server.status === 'error' ? colors.error : colors.dimText} wrap="truncate-end">
            {`${statusIcon(server.status)} ${server.name} · ${server.transport} · ${server.status} · ${server.toolCount} tools`}
          </Text>
          {server.error ? <Text color={colors.error}>{`  ${server.error}`}</Text> : null}
          {tools.length > 0 ? <Text color={colors.dimText} wrap="truncate-end">{`  ${tools.slice(0, 8).join(' · ')}${tools.length > 8 ? ` · +${tools.length - 8}` : ''}`}</Text> : null}
        </Box>
      )) : null}
      {!loading && !error && normalized && visible.length === 0 ? <Text color={colors.dimText}>No matching MCP server or tool.</Text> : null}
      <Text color={colors.dimText}>type to filter · esc close · mutations: `sepilot mcp …`</Text>
    </Box>
  )
}
