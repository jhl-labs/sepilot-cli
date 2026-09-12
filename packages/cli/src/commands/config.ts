import type { DaemonChannelConfig, DaemonConfigProvider } from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { formatProviderModelBadges } from '../utils/provider-display.js'

export async function configCommand(options: { url?: string }) {
  const client = new DaemonClient(options.url)
  const data = await client.config()
  output(data, (d) => [
    `Device: ${d.device.name} (${d.device.role})`,
    `Daemon: ${d.daemon.host}:${d.daemon.port}`,
    `Agent: autonomy=${d.agent.autonomy}, thinking=${d.agent.thinkingLevel}`,
    `Providers: ${d.providers.map((p: DaemonConfigProvider) => p.id).join(', ') || '(none)'}`,
    `Channels: ${d.channels.filter((c: DaemonChannelConfig) => c.enabled).map((c: DaemonChannelConfig) => c.type).join(', ') || '(none)'}`,
    `MCP Servers: ${d.mcp.servers.filter((server) => server.enabled !== false).map((server) => server.name).join(', ') || '(none)'}`,
    `Outbound Webhooks: ${d.hooks.outboundWebhooks.filter((webhook) => webhook.enabled !== false).length}`,
  ].join('\n'))
}

export async function providersCommand(options: { url?: string }) {
  const client = new DaemonClient(options.url)
  const data = await client.providers()
  output(data, (d) => {
    if (!d?.length) return 'No providers configured.'
    const lines: string[] = []
    for (const p of d) {
      lines.push(`  ${p.name} (${p.id})`)
      const embedIds = new Set(p.embeddingModelIds)
      for (const m of p.models) {
        // A model is embed-only when daemon flags it as such *and* the
        // capabilities don't include tool use. Some providers (Ollama)
        // tag every chat model as embedding-capable, which previously
        // made the whole list render as [embed]. Trust capabilities
        // first, fall back to the embeddingModelIds set.
        const cap = (m as { capabilities?: { toolUse?: boolean; embedding?: boolean } }).capabilities
        const embedOnly = cap?.embedding === true && cap?.toolUse !== true
        const role = embedOnly ? 'embed' : embedIds.has(m.id) && !cap?.toolUse ? 'embed' : 'chat '
        lines.push(`    [${role}] ${m.id} — ctx:${m.contextWindow} out:${m.maxOutputTokens}${formatProviderModelBadges(m)}`)
      }
      for (const modelId of p.unavailableConfiguredModelIds ?? []) {
        lines.push(`    [down ] ${modelId} — configured, absent from current endpoint catalog`)
      }
    }
    return lines.join('\n')
  })
}
