import { access } from 'node:fs/promises'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'

interface EmbeddedDaemonOptions {
  dataDir: string
  host: string
  port: number
  autoApproveCliFlag?: boolean
}

type EmbeddedDaemonModule = {
  startEmbeddedDaemon(options: EmbeddedDaemonOptions): Promise<unknown>
}

let embeddedStartPromise: Promise<unknown> | null = null

function loopbackListenOptions(baseUrl: string): Pick<EmbeddedDaemonOptions, 'host' | 'port'> | null {
  let parsed: URL
  try {
    parsed = new URL(baseUrl)
  } catch {
    return null
  }
  if (parsed.protocol !== 'http:') return null
  if (!['127.0.0.1', 'localhost', '::1', '[::1]'].includes(parsed.hostname)) {
    return null
  }
  const port = Number(parsed.port || '80')
  if (!Number.isInteger(port) || port <= 0) return null
  const host = parsed.hostname === '[::1]' ? '::1' : parsed.hostname
  return {
    host: host === 'localhost' ? '127.0.0.1' : host,
    port,
  }
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await access(path)
    return true
  } catch {
    return false
  }
}

async function resolveEmbeddedDaemonSpecifiers(): Promise<string[]> {
  const explicit = process.env.SEPILOTD_EMBEDDED_DAEMON_MODULE?.trim()
  const specifiers = explicit ? [explicit] : ['@sepilotd/daemon/embedded']
  const root = import.meta.dirname
  const candidates = [
    join(root, 'node_modules', '@sepilotd', 'daemon', 'dist', 'embedded.js'),
    join(root, '..', 'node_modules', '@sepilotd', 'daemon', 'dist', 'embedded.js'),
    join(root, '..', 'daemon', 'dist', 'embedded.js'),
    join(root, '..', '..', 'daemon', 'dist', 'embedded.js'),
    join(root, '..', '..', '..', 'daemon', 'dist', 'embedded.js'),
  ]

  for (const candidate of candidates) {
    if (await fileExists(candidate)) {
      specifiers.push(pathToFileURL(candidate).href)
    }
  }
  return [...new Set(specifiers)]
}

export async function startEmbeddedDaemonIfAvailable(options: {
  baseUrl: string
  dataDir: string
}): Promise<boolean> {
  const listen = loopbackListenOptions(options.baseUrl)
  if (!listen) return false

  if (!embeddedStartPromise) {
    embeddedStartPromise = (async () => {
      let lastError: unknown = null
      for (const specifier of await resolveEmbeddedDaemonSpecifiers()) {
        let mod: EmbeddedDaemonModule
        try {
          mod = await import(specifier) as EmbeddedDaemonModule
        } catch (error) {
          lastError = error
          continue
        }
        return await mod.startEmbeddedDaemon({
          dataDir: options.dataDir,
          host: listen.host,
          port: listen.port,
          autoApproveCliFlag: process.argv.includes('--yes-to-everything'),
        })
      }
      throw lastError ?? new Error('embedded daemon module not found')
    })()
  }

  try {
    await embeddedStartPromise
    return true
  } catch (error) {
    embeddedStartPromise = null
    if (process.env.SEPILOTD_DEBUG === '1') {
      process.stderr.write(`Embedded sepilotd startup failed: ${error instanceof Error ? error.message : String(error)}\n`)
    }
    return false
  }
}
