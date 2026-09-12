import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js'
import type { SecretVault } from '../security/secret-vault.js'
import { resolveSecretPlaceholders } from './secret-resolver.js'
import type { McpServerProvenance, McpToolManifest } from './security.js'

type McpServerSecurityConfig = {
  provenance?: McpServerProvenance
  toolManifest?: McpToolManifest
}

export type StdioMcpServerConfig = McpServerSecurityConfig & {
  name: string
  enabled: boolean
  transport: 'stdio'
  command: string
  args: string[]
  env: Record<string, string>
  disabledTools: string[]
  timeoutMs?: number
}

export type SseMcpServerConfig = McpServerSecurityConfig & {
  name: string
  enabled: boolean
  transport: 'sse'
  url: string
  headers: Record<string, string>
  disabledTools: string[]
  timeoutMs?: number
}

export type HttpMcpServerConfig = McpServerSecurityConfig & {
  name: string
  enabled: boolean
  transport: 'http'
  url: string
  headers: Record<string, string>
  disabledTools: string[]
  timeoutMs?: number
}

export type McpServerConfig =
  | StdioMcpServerConfig
  | SseMcpServerConfig
  | HttpMcpServerConfig

// Variables the spawned MCP child needs to function (PATH lookup, locale,
// terminal sizing, package-manager caches). Everything else from the daemon
// env is *excluded* by default, so provider API keys, signing tokens, and
// SMTP creds do not silently leak to every MCP server. Operators include
// explicit env entries via the server config when a server needs more.
const MCP_ENV_ALLOWLIST = new Set([
  'PATH',
  'HOME',
  'USER',
  'LOGNAME',
  'SHELL',
  'LANG',
  'LC_ALL',
  'LC_CTYPE',
  'TZ',
  'TMPDIR',
  'TEMP',
  'TMP',
  'TERM',
  'COLORTERM',
  'COLUMNS',
  'LINES',
  'NODE_PATH',
  'PNPM_HOME',
  'NPM_CONFIG_USERCONFIG',
  'NPM_CONFIG_CACHE',
  'PYTHONPATH',
  'SystemRoot',     // Windows
  'WINDIR',         // Windows
  'APPDATA',        // Windows
  'LOCALAPPDATA',   // Windows
  'PROGRAMFILES',   // Windows
  'USERPROFILE',    // Windows
  // Corporate proxy + CA bundle (HTTP transport-level, not secret-bearing).
  // Without these, MCP children behind a corporate proxy / private CA cannot
  // reach the network they were already reaching when run outside the daemon.
  'HTTP_PROXY',
  'HTTPS_PROXY',
  'NO_PROXY',
  'ALL_PROXY',
  'http_proxy',
  'https_proxy',
  'no_proxy',
  'all_proxy',
  'SSL_CERT_FILE',
  'SSL_CERT_DIR',
  'NODE_EXTRA_CA_CERTS',
  'REQUESTS_CA_BUNDLE',
  'CURL_CA_BUNDLE',
])

function pickAllowedDaemonEnv(): Record<string, string> {
  const out: Record<string, string> = {}
  for (const key of MCP_ENV_ALLOWLIST) {
    const value = process.env[key]
    if (typeof value === 'string') out[key] = value
  }
  // Pass through SEPILOTD_MCP_* explicitly — these are MCP-side knobs the
  // operator put there on purpose. Never pass other SEPILOTD_* (they may
  // include tokens, secret paths, audit destinations).
  for (const [k, v] of Object.entries(process.env)) {
    if (k.startsWith('SEPILOTD_MCP_') && typeof v === 'string') out[k] = v
  }
  return out
}

export function createMcpTransport(config: McpServerConfig, vault?: SecretVault | null): Transport {
  switch (config.transport) {
    case 'stdio': {
      // Scrub the daemon's env (allowlist) before merging the
      // server-specific entries; without this, every MCP child inherits
      // provider keys, gateway tokens, encryption keys, etc.
      const baseEnv = pickAllowedDaemonEnv()
      const resolvedEnv = Object.fromEntries(
        Object.entries({ ...baseEnv, ...config.env }).map(
          ([k, v]) => [k, v ? resolveSecretPlaceholders(v, vault ?? null) : v]
        )
      )
      return new StdioClientTransport({ command: config.command, args: config.args, env: resolvedEnv as Record<string, string> })
    }
    case 'sse': {
      const resolvedHeaders = Object.fromEntries(
        Object.entries(config.headers).map(
          ([k, v]) => [k, resolveSecretPlaceholders(v, vault ?? null)]
        )
      )
      return new SSEClientTransport(new URL(config.url), { requestInit: { headers: resolvedHeaders } })
    }
    case 'http': {
      const resolvedHeaders = Object.fromEntries(
        Object.entries(config.headers).map(
          ([k, v]) => [k, resolveSecretPlaceholders(v, vault ?? null)]
        )
      )
      return new StreamableHTTPClientTransport(new URL(config.url), { requestInit: { headers: resolvedHeaders } })
    }
  }
}
