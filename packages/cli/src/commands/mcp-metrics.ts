import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

function avgMs(totalDurationMs: number, calls: number): string {
  if (calls === 0) return '-'
  return `${Math.round(totalDurationMs / calls)}ms`
}

export async function mcpMetricsCommand(server: string | undefined, options: { url?: string }) {
  const client = new DaemonClient(options.url)
  if (server) {
    const entry = await client.mcpMetricsForServer(server)
    output(entry, (e) => {
      const lines = [
        chalk.cyan(server),
        `  calls=${e.totalCalls}  errors=${e.errors}  avg=${avgMs(e.totalDurationMs, e.totalCalls)}  last=${e.lastCallAt ?? '-'}`,
      ]
      for (const [tool, per] of Object.entries(e.perTool)) {
        lines.push(`  ${tool.padEnd(30)} calls=${per.calls}  errors=${per.errors}  avg=${avgMs(per.totalDurationMs, per.calls)}`)
      }
      return lines.join('\n')
    })
    return
  }
  const snap = await client.mcpMetrics()
  output(snap, (s) => {
    const entries = Object.entries(s.servers)
    if (!entries.length) return 'No MCP tool calls recorded yet.'
    return entries.map(([name, e]) =>
      `  ${name.padEnd(18)} calls=${e.totalCalls}  errors=${e.errors}  avg=${avgMs(e.totalDurationMs, e.totalCalls)}  last=${e.lastCallAt ?? '-'}`,
    ).join('\n')
  })
}
