import { useState, useEffect, useCallback } from 'react'

export interface McpServerInfo {
  name: string
  status: 'connected' | 'error' | 'disabled'
  connected: boolean
  transport: string
  command?: string
  args?: string[]
  toolCount: number
  tools: string[]
  allTools?: string[]
  disabledTools?: string[]
  error?: string
}

export interface McpMarketplaceInfo {
  name: string
  url: string
  addedAt: string
  lastSync: string | null
}

interface McpMetricsEntry {
  totalCalls: number
  errors: number
  totalDurationMs: number
  lastCallAt: string | null
}

export interface McpMetricsSnapshot {
  servers: Record<string, McpMetricsEntry>
}

interface McpStatusState {
  servers: McpServerInfo[]
  metrics: McpMetricsSnapshot
  marketplaces: McpMarketplaceInfo[]
  loading: boolean
  error: string | null
}

export function useMcpStatus(
  client: {
    mcpServers: () => Promise<McpServerInfo[]>
    mcpMetrics: () => Promise<McpMetricsSnapshot>
    mcpMarketplaceList?: () => Promise<McpMarketplaceInfo[]>
  } | null,
  intervalMs = 10_000,
): McpStatusState {
  const [state, setState] = useState<McpStatusState>({
    servers: [],
    metrics: { servers: {} },
    marketplaces: [],
    loading: true,
    error: null,
  })

  const refresh = useCallback(async () => {
    if (!client) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: 'MCP client unavailable',
      }))
      return
    }
    try {
      const [servers, metrics, marketplaces] = await Promise.all([
        client.mcpServers(),
        client.mcpMetrics(),
        client.mcpMarketplaceList?.().catch(() => []) ?? Promise.resolve([]),
      ])
      setState({ servers, metrics, marketplaces, loading: false, error: null })
    } catch (err) {
      setState(prev => ({ ...prev, loading: false, error: err instanceof Error ? err.message : String(err) }))
    }
  }, [client])

  useEffect(() => {
    void refresh()
    const timer = setInterval(() => void refresh(), intervalMs)
    return () => clearInterval(timer)
  }, [refresh, intervalMs])

  return state
}
