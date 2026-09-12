import type { DaemonDevice } from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export async function devicesCommand(options: { url?: string }) {
  const client = new DaemonClient(options.url)
  const data = await client.devices()
  output(data, (d) => d.map((dev: DaemonDevice) =>
    `  ${dev.name.padEnd(20)} ${dev.role.padEnd(10)} ${dev.status.padEnd(10)} ${dev.platform}/${dev.arch}  caps: ${dev.capabilities.join(', ')}`
  ).join('\n'))
}
