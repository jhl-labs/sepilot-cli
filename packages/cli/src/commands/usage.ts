import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export async function usageCommand(options: { url?: string }) {
  const client = new DaemonClient(options.url)
  const data = await client.usage()
  output(data, (d) => [
    `Total Usage:`,
    `  Input tokens:  ${d.inputTokens.toLocaleString()}`,
    `  Output tokens: ${d.outputTokens.toLocaleString()}`,
    `  Cost:          $${d.costUsd.toFixed(4)}`,
    `  Requests:      ${d.requestCount}`,
  ].join('\n'))
}

export async function usageDailyCommand(options: { url?: string; days?: string }) {
  const client = new DaemonClient(options.url)
  const data = await client.usageDaily(options.days ? parseInt(options.days) : undefined)
  output(data, (d) => {
    if (!d?.length) return 'No usage data.'
    const headers = ['Date', 'Provider', 'Model', 'Input', 'Output', 'Cost']
    // Header widths must match data column widths so the table aligns.
    const widths = [10, 10, 15, 8, 8, 10] as const
    const formatCell = (text: string, width: number, align: 'left' | 'right' = 'left') =>
      align === 'right' ? text.padStart(width) : text.padEnd(width)
    const lines = [
      [
        formatCell(headers[0], widths[0]),
        formatCell(headers[1], widths[1]),
        formatCell(headers[2], widths[2]),
        formatCell(headers[3], widths[3], 'right'),
        formatCell(headers[4], widths[4], 'right'),
        formatCell(headers[5], widths[5], 'right'),
      ].join('  '),
    ]
    for (const row of d) {
      lines.push([
        formatCell(row.date, widths[0]),
        formatCell(row.provider, widths[1]),
        formatCell(row.model, widths[2]),
        formatCell(String(row.totalInputTokens), widths[3], 'right'),
        formatCell(String(row.totalOutputTokens), widths[4], 'right'),
        formatCell(`$${row.totalCostUsd.toFixed(4)}`, widths[5], 'right'),
      ].join('  '))
    }
    return lines.join('\n')
  })
}
