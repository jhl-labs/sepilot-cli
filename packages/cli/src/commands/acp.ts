import { DaemonClient } from '../client/http.js'
import { runAcpProxy } from '../client/acp-proxy.js'

export interface AcpCommandOptions {
  url?: string
}

export async function acpCommand(options: AcpCommandOptions = {}): Promise<void> {
  const client = new DaemonClient(options.url)
  const handle = runAcpProxy(
    { client },
    { stdin: process.stdin, stdout: process.stdout },
  )
  await handle.closed
}
