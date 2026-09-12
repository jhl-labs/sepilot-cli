import { homedir, platform } from 'node:os'
import { isAbsolute, join, resolve } from 'node:path'
import type { DaemonMcpServerConfig } from '@sepilotd/api-client'

type PlaywrightMcpMode = 'auto' | 'visible' | 'headless'
type ResolvedPlaywrightMcpMode = 'visible' | 'headless'

export interface PlaywrightMcpOptions {
  name?: string
  mode?: string
  browser?: string
  userDataDir?: string
  caps?: string
  mcpPackage?: string
  isolated?: boolean
  storageState?: string
  outputDir?: string
  noSandbox?: boolean
  viewportSize?: string
  userAgent?: string
  device?: string
  proxyServer?: string
  proxyBypass?: string
  timeoutAction?: string
  timeoutNavigation?: string
  imageResponses?: string
  allowedOrigins?: string
  blockedOrigins?: string
  ignoreHttpsErrors?: boolean
  blockServiceWorkers?: boolean
  saveSession?: boolean
  disabled?: boolean
}

export interface PlaywrightMcpEnvironment {
  platform?: NodeJS.Platform
  env?: NodeJS.ProcessEnv
  homeDir?: string
  cwd?: string
}

export interface PlaywrightMcpConfigResult {
  requestedMode: PlaywrightMcpMode
  resolvedMode: ResolvedPlaywrightMcpMode
  browser: string
  userDataDir?: string
  isolated: boolean
  server: DaemonMcpServerConfig
}

const MODE_VALUES = new Set<PlaywrightMcpMode>(['auto', 'visible', 'headless'])
const CAPABILITY_VALUES = new Set([
  'vision',
  'pdf',
  'devtools',
])
const IMAGE_RESPONSE_VALUES = new Set(['allow', 'omit'])

function safePathSegment(value: string): string {
  return value.replace(/[^a-z0-9._-]+/gi, '-').replace(/^-+|-+$/g, '') || 'playwright'
}

function normalizeMode(raw: string | undefined): PlaywrightMcpMode {
  const value = (raw ?? 'auto').trim().toLowerCase()
  if (MODE_VALUES.has(value as PlaywrightMcpMode)) {
    return value as PlaywrightMcpMode
  }
  throw new Error(`Invalid Playwright MCP mode: ${raw}. Use auto, visible, or headless.`)
}

function resolveMode(
  requestedMode: PlaywrightMcpMode,
  environment: Required<Pick<PlaywrightMcpEnvironment, 'platform' | 'env'>>,
): ResolvedPlaywrightMcpMode {
  if (requestedMode === 'headless' || requestedMode === 'visible') {
    return requestedMode
  }

  if (environment.platform === 'win32' || environment.platform === 'darwin') {
    return 'visible'
  }

  return environment.env.DISPLAY || environment.env.WAYLAND_DISPLAY
    ? 'visible'
    : 'headless'
}

function defaultBrowser(osPlatform: NodeJS.Platform): string {
  return osPlatform === 'win32' ? 'msedge' : 'chromium'
}

function expandPath(input: string, env: Required<Pick<PlaywrightMcpEnvironment, 'homeDir' | 'cwd'>>): string {
  const expanded = input === '~'
    ? env.homeDir
    : input.startsWith('~/')
      ? join(env.homeDir, input.slice(2))
      : input
  return isAbsolute(expanded) ? expanded : resolve(env.cwd, expanded)
}

function normalizeCaps(raw: string | undefined): string | undefined {
  const values = raw
    ?.split(',')
    .map((part) => part.trim())
    .filter(Boolean) ?? []
  if (!values.length) return undefined

  const invalid = values.filter((value) => !CAPABILITY_VALUES.has(value))
  if (invalid.length) {
    throw new Error(
      `Invalid Playwright MCP capability: ${invalid.join(', ')}. Use vision, pdf, and/or devtools.`,
    )
  }
  return Array.from(new Set(values)).join(',')
}

function normalizeImageResponses(raw: string | undefined): string | undefined {
  const value = raw?.trim().toLowerCase()
  if (!value) return undefined
  if (IMAGE_RESPONSE_VALUES.has(value)) return value
  throw new Error(`Invalid Playwright MCP image response mode: ${raw}. Use allow or omit.`)
}

function nonEmpty(raw: string | undefined): string | undefined {
  const value = raw?.trim()
  return value ? value : undefined
}

function pushOptionalValueArg(args: string[], flag: string, raw: string | undefined): void {
  const value = nonEmpty(raw)
  if (value) args.push(`${flag}=${value}`)
}

export function buildPlaywrightMcpServerConfig(
  options: PlaywrightMcpOptions = {},
  environment: PlaywrightMcpEnvironment = {},
): PlaywrightMcpConfigResult {
  const env = environment.env ?? process.env
  const osPlatform = environment.platform ?? platform()
  const homeDir = environment.homeDir ?? homedir()
  const cwd = environment.cwd ?? process.cwd()
  const requestedMode = normalizeMode(options.mode)
  const resolvedMode = resolveMode(requestedMode, { platform: osPlatform, env })
  const name = options.name?.trim() || 'playwright'
  if (!/^[a-z0-9][a-z0-9-_.]*$/i.test(name)) {
    throw new Error(
      `Invalid MCP server name: ${name} (use letters, digits, '-', '_', '.' only; must start with a letter or digit)`,
    )
  }

  if (options.isolated && options.userDataDir) {
    throw new Error('Use either --isolated or --user-data-dir, not both.')
  }
  if (options.storageState && !options.isolated) {
    throw new Error('--storage-state requires --isolated.')
  }

  const browser = options.browser?.trim() || defaultBrowser(osPlatform)
  const mcpPackage = options.mcpPackage?.trim() || '@playwright/mcp@latest'
  const caps = normalizeCaps(options.caps)
  const imageResponses = normalizeImageResponses(options.imageResponses)
  const isolated = Boolean(options.isolated)
  const userDataDir = isolated
    ? undefined
    : options.userDataDir
      ? expandPath(options.userDataDir, { homeDir, cwd })
      : join(homeDir, '.sepilotd', 'browser', `${safePathSegment(name)}-${resolvedMode}`)

  const args = [
    '-y',
    mcpPackage,
    `--browser=${browser}`,
  ]

  if (resolvedMode === 'headless') args.push('--headless')
  if (isolated) args.push('--isolated')
  if (userDataDir) args.push(`--user-data-dir=${userDataDir}`)
  if (caps) args.push(`--caps=${caps}`)
  if (options.storageState) args.push(`--storage-state=${expandPath(options.storageState, { homeDir, cwd })}`)
  if (options.outputDir) args.push(`--output-dir=${expandPath(options.outputDir, { homeDir, cwd })}`)
  if (options.viewportSize) args.push(`--viewport-size=${options.viewportSize}`)
  pushOptionalValueArg(args, '--user-agent', options.userAgent)
  pushOptionalValueArg(args, '--device', options.device)
  pushOptionalValueArg(args, '--proxy-server', options.proxyServer)
  pushOptionalValueArg(args, '--proxy-bypass', options.proxyBypass)
  pushOptionalValueArg(args, '--timeout-action', options.timeoutAction)
  pushOptionalValueArg(args, '--timeout-navigation', options.timeoutNavigation)
  if (imageResponses) args.push(`--image-responses=${imageResponses}`)
  pushOptionalValueArg(args, '--allowed-origins', options.allowedOrigins)
  pushOptionalValueArg(args, '--blocked-origins', options.blockedOrigins)
  if (options.noSandbox) args.push('--no-sandbox')
  if (options.ignoreHttpsErrors) args.push('--ignore-https-errors')
  if (options.blockServiceWorkers) args.push('--block-service-workers')
  if (options.saveSession) args.push('--save-session')

  return {
    requestedMode,
    resolvedMode,
    browser,
    userDataDir,
    isolated,
    server: {
      name,
      enabled: options.disabled ? false : true,
      transport: 'stdio',
      command: 'npx',
      args,
      env: {},
    },
  }
}
