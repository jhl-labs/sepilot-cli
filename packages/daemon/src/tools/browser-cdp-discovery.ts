import { readFile as fsReadFile } from 'node:fs/promises'
import { join } from 'node:path'

export type CdpChannel = 'stable' | 'beta' | 'dev' | 'canary'

export type CdpDiscoveryCode =
  | 'CDP_NOT_ENABLED'
  | 'CDP_INVALID_ENDPOINT'
  | 'CDP_UNSUPPORTED_PLATFORM'

export const CDP_ENABLE_INSTRUCTION =
  'Open chrome://inspect/#remote-debugging in Chrome 144 or newer, enable remote debugging, then connect again.'

export class CdpDiscoveryError extends Error {
  override name = 'CdpDiscoveryError'
  constructor(
    readonly code: CdpDiscoveryCode,
    detail: string,
  ) {
    super(`${code}: ${detail}`)
  }
}

const LINUX_SUFFIX: Record<CdpChannel, string | null> = {
  stable: 'google-chrome',
  beta: 'google-chrome-beta',
  dev: 'google-chrome-unstable',
  canary: null,
}
const DARWIN_SUFFIX: Record<CdpChannel, string> = {
  stable: 'Chrome',
  beta: 'Chrome Beta',
  dev: 'Chrome Dev',
  canary: 'Chrome Canary',
}
const WIN32_SUFFIX: Record<CdpChannel, string> = {
  stable: 'Chrome',
  beta: 'Chrome Beta',
  dev: 'Chrome Dev',
  canary: 'Chrome SxS',
}

export function cdpUserDataDir(
  channel: CdpChannel,
  platform: NodeJS.Platform,
  env: Record<string, string | undefined>,
): string | null {
  if (platform === 'darwin') {
    if (!env.HOME) return null
    return join(env.HOME, 'Library', 'Application Support', 'Google', DARWIN_SUFFIX[channel])
  }
  if (platform === 'win32') {
    if (!env.LOCALAPPDATA) return null
    return [env.LOCALAPPDATA, 'Google', WIN32_SUFFIX[channel], 'User Data'].join('\\')
  }
  const suffix = LINUX_SUFFIX[channel]
  if (!suffix || !env.HOME) return null
  return join(env.HOME, '.config', suffix)
}

const BROWSER_PATH = /^\/devtools\/browser\/[A-Za-z0-9-]+$/

export function parseDevToolsActivePort(content: string): {
  port: number
  path: string
  wsEndpoint: string
} {
  const [rawPort, rawPath] = content
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
  if (!rawPort || !rawPath) {
    throw new CdpDiscoveryError('CDP_INVALID_ENDPOINT', 'DevToolsActivePort is empty or truncated')
  }
  const port = Number.parseInt(rawPort, 10)
  if (!Number.isInteger(port) || port <= 0 || port > 65535 || String(port) !== rawPort) {
    throw new CdpDiscoveryError('CDP_INVALID_ENDPOINT', `invalid port '${rawPort}'`)
  }
  if (!BROWSER_PATH.test(rawPath)) {
    throw new CdpDiscoveryError('CDP_INVALID_ENDPOINT', 'endpoint path is not a browser target')
  }
  return { port, path: rawPath, wsEndpoint: `ws://127.0.0.1:${port}${rawPath}` }
}

export interface DiscoverCdpOptions {
  channel: CdpChannel
  userDataDir?: string
  platform?: NodeJS.Platform
  env?: Record<string, string | undefined>
  readFile?: (path: string) => Promise<string>
}

export async function discoverCdpEndpoint(
  options: DiscoverCdpOptions,
): Promise<{ wsEndpoint: string; userDataDir: string }> {
  const userDataDir =
    options.userDataDir ??
    cdpUserDataDir(options.channel, options.platform ?? process.platform, options.env ?? process.env)
  if (!userDataDir) {
    throw new CdpDiscoveryError(
      'CDP_UNSUPPORTED_PLATFORM',
      `no Chrome ${options.channel} user data directory on this platform`,
    )
  }
  const readFile = options.readFile ?? ((path: string) => fsReadFile(path, 'utf8'))
  let content: string
  try {
    content = await readFile(join(userDataDir, 'DevToolsActivePort'))
  } catch {
    throw new CdpDiscoveryError('CDP_NOT_ENABLED', CDP_ENABLE_INSTRUCTION)
  }
  const { wsEndpoint } = parseDevToolsActivePort(content)
  return { wsEndpoint, userDataDir }
}
