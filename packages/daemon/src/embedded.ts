import { startDaemonRuntime, type DaemonRuntimeHandle } from './bootstrap.js'

export interface StartEmbeddedDaemonOptions {
  dataDir?: string
  port?: number
  host?: string
  autoApproveCliFlag?: boolean
}

export interface EmbeddedDaemonHandle {
  address: string
  dataDir: string
  host: string
  port: number
  close(reason?: string): Promise<void>
  shutdown(reason?: string): Promise<void>
}

export async function startEmbeddedDaemon(
  options: StartEmbeddedDaemonOptions = {},
): Promise<EmbeddedDaemonHandle> {
  const handle: DaemonRuntimeHandle = await startDaemonRuntime({
    dataDir: options.dataDir,
    port: options.port,
    host: options.host,
    autoApproveCliFlag: options.autoApproveCliFlag ?? process.argv.includes('--yes-to-everything'),
    setupSignalHandlers: false,
    exitOnShutdown: false,
  })
  handle.app.server.unref?.()
  const serverAddress = handle.app.server.address()
  const address = typeof serverAddress === 'string'
    ? serverAddress
    : `http://${serverAddress?.address ?? handle.host}:${serverAddress?.port ?? handle.port}`

  const close = (reason = 'desktop embedded daemon shutdown') => handle.shutdown(reason)

  return {
    address,
    dataDir: handle.dataDir,
    host: handle.host,
    port: handle.port,
    close,
    shutdown: close,
  }
}
