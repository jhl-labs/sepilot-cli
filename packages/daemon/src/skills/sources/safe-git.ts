import { isIP } from 'node:net'
import { devNull, platform } from 'node:os'
import simpleGit, { type SimpleGit } from 'simple-git'
import { assertPublicUrl, type PublicUrlResolution } from '../../utils/ssrf-guard.js'
import { SkillSourceUrlNotAllowedError } from '../errors.js'
import type { SkillSourceUrlPolicy } from '../source-url-policy.js'

const DEFAULT_TRUSTED_GIT_HOSTS = new Set(['github.com'])
const REMOTE_SCHEME = /^[a-z][a-z0-9+.-]*:\/\//iu
const WINDOWS_DRIVE_PATH = /^[a-z]:[\\/]/iu
const SCP_STYLE_REMOTE = /^[^/\\:]+(?:@[^/\\:]+)?:/u

export interface SafeGitRuntime {
  resolveUrl?: typeof assertPublicUrl
  urlPolicy?: SkillSourceUrlPolicy
}

export interface SafeGitTransport {
  source: string
  remote: boolean
  createGit(baseDir?: string): SimpleGit
}

/**
 * Build an isolated git execution context for a skill source.
 *
 * Remote git is deliberately narrower than a generic `git clone`: only HTTPS
 * is accepted, redirects and proxies are disabled, and libcurl is pinned to
 * the public address approved by the SSRF guard. GitHub is trusted because it
 * backs the built-in marketplace; other hosts require an explicit operator
 * allowlist. Local filesystem repositories remain available for tests and
 * operator-managed sources, but UNC and scp/remote-helper spellings are
 * rejected so they cannot silently become network access.
 */
export async function prepareSafeGitTransport(
  rawSource: string,
  runtime: SafeGitRuntime = {},
): Promise<SafeGitTransport> {
  const source = rawSource.trim()
  if (!source) {
    throw new SkillSourceUrlNotAllowedError(rawSource, 'git source is empty')
  }
  if (/[\0\r\n]/u.test(source) || source.startsWith('-')) {
    throw new SkillSourceUrlNotAllowedError(source, 'git source contains unsafe characters')
  }

  if (isLocalGitPath(source)) {
    runtime.urlPolicy?.assertAllowed(source, 'git')
    return createTransport(source, false, localGitConfig())
  }

  runtime.urlPolicy?.assertAllowed(source, 'git')
  const parsed = parsePublicGitUrl(source)
  if (
    !DEFAULT_TRUSTED_GIT_HOSTS.has(parsed.hostname.toLowerCase()) &&
    runtime.urlPolicy?.enforced !== true
  ) {
    throw new SkillSourceUrlNotAllowedError(
      source,
      `remote git host ${parsed.hostname} requires an explicit security.skillSources allowlist`,
    )
  }

  const resolveUrl = runtime.resolveUrl ?? assertPublicUrl
  let resolution: PublicUrlResolution
  try {
    resolution = await resolveUrl(parsed.toString())
  } catch (error) {
    throw new SkillSourceUrlNotAllowedError(
      source,
      `git source did not resolve to a public address: ${errorMessage(error)}`,
    )
  }
  assertMatchingResolution(parsed, resolution, source)

  const transport = createTransport(parsed.toString(), true, buildPinnedGitConfig(resolution))
  await assertPinnedDnsSupported(transport, source)
  return transport
}

export function isLocalGitPath(source: string): boolean {
  if (source.startsWith('\\\\') || source.startsWith('//')) {
    throw new SkillSourceUrlNotAllowedError(source, 'UNC git sources are not allowed')
  }
  if (WINDOWS_DRIVE_PATH.test(source)) return true
  if (REMOTE_SCHEME.test(source) || SCP_STYLE_REMOTE.test(source)) return false
  return true
}

function parsePublicGitUrl(source: string): URL {
  if (!/^https:\/\//iu.test(source)) {
    throw new SkillSourceUrlNotAllowedError(source, 'only https:// git sources are allowed')
  }
  let parsed: URL
  try {
    parsed = new URL(source)
  } catch {
    throw new SkillSourceUrlNotAllowedError(source, 'git source is not a valid URL')
  }
  if (parsed.protocol !== 'https:') {
    throw new SkillSourceUrlNotAllowedError(source, 'only https:// git sources are allowed')
  }
  if (parsed.username || parsed.password) {
    throw new SkillSourceUrlNotAllowedError(source, 'git source credentials are not allowed')
  }
  if (!parsed.hostname) {
    throw new SkillSourceUrlNotAllowedError(source, 'git source host is missing')
  }
  return parsed
}

function assertMatchingResolution(
  requested: URL,
  resolution: PublicUrlResolution,
  source: string,
): void {
  if (
    resolution.url.protocol !== 'https:' ||
    resolution.url.username ||
    resolution.url.password ||
    resolution.url.toString() !== requested.toString() ||
    normalizeHost(resolution.hostname) !== normalizeHost(requested.hostname)
  ) {
    throw new SkillSourceUrlNotAllowedError(
      source,
      'git source DNS validation returned a different destination',
    )
  }
}

function createTransport(source: string, remote: boolean, config: string[]): SafeGitTransport {
  return {
    source,
    remote,
    createGit(baseDir?: string): SimpleGit {
      const git = simpleGit({
        ...(baseDir ? { baseDir } : {}),
        config,
        timeout: {
          block: 30_000,
          stdErr: true,
          stdOut: true,
        },
        // Every protocol override above is a fixed internal value. Opting in
        // here only bypasses simple-git's generic argv guard; callers cannot
        // supply additional git config.
        unsafe: {
          allowUnsafeConfigPaths: true,
          allowUnsafeProtocolOverride: true,
        },
      })
      return git.env(isolatedGitEnvironment())
    },
  }
}

async function assertPinnedDnsSupported(
  transport: SafeGitTransport,
  source: string,
): Promise<void> {
  let version: Awaited<ReturnType<SimpleGit['version']>>
  try {
    version = await transport.createGit().version()
  } catch (error) {
    throw new SkillSourceUrlNotAllowedError(
      source,
      `unable to verify git supports pinned DNS: ${errorMessage(error)}`,
    )
  }
  if (!supportsCurloptResolve(version.major, version.minor)) {
    throw new SkillSourceUrlNotAllowedError(
      source,
      `git 2.34 or newer is required for pinned DNS (found ${version.major}.${version.minor})`,
    )
  }
}

export function supportsCurloptResolve(major: number, minor: number): boolean {
  // Ubuntu 22.04 ships Git 2.34, whose libcurl transport supports this
  // option.  Requiring 2.37 made the supported CI/runtime platform reject
  // every remote marketplace before it could use the pinned-DNS defence.
  return major > 2 || (major === 2 && minor >= 34)
}

function baseGitConfig(): string[] {
  return ['fetch.recurseSubmodules=false', 'submodule.recurse=false']
}

function localGitConfig(): string[] {
  return [...baseGitConfig(), 'protocol.allow=never', 'protocol.file.allow=always']
}

export function buildPinnedGitConfig(resolution: PublicUrlResolution): string[] {
  const config = [
    ...baseGitConfig(),
    'http.followRedirects=false',
    'http.proxy=',
    'http.sslVerify=true',
    'protocol.allow=never',
    'protocol.https.allow=always',
    'protocol.file.allow=never',
    'protocol.ext.allow=never',
    'protocol.ssh.allow=never',
    'protocol.git.allow=never',
  ]
  if (!isIP(resolution.hostname)) {
    const port = resolution.url.port || '443'
    const address = resolution.family === 6 ? `[${resolution.address}]` : resolution.address
    config.push(`http.curloptResolve=${resolution.hostname}:${port}:${address}`)
  }
  return config
}

function isolatedGitEnvironment(): NodeJS.ProcessEnv {
  const env = { ...process.env }
  const configNullDevice = platform() === 'win32' ? 'NUL' : devNull
  for (const key of Object.keys(env)) {
    if (
      /^GIT_CONFIG_(?:KEY|VALUE)_\d+$/u.test(key) ||
      [
        'GIT_ALTERNATE_OBJECT_DIRECTORIES',
        'GIT_COMMON_DIR',
        'GIT_CONFIG',
        'GIT_CONFIG_COUNT',
        'GIT_CONFIG_PARAMETERS',
        'GIT_DIR',
        'GIT_EXEC_PATH',
        'GIT_OBJECT_DIRECTORY',
        'GIT_PROXY_COMMAND',
        'GIT_PAGER',
        'GIT_SSH',
        'GIT_SSH_COMMAND',
        'GIT_SSL_NO_VERIFY',
        'GIT_WORK_TREE',
      ].includes(key)
    ) {
      delete env[key]
    }
  }
  // simple-git rejects inherited pager commands unless its unsafe pager mode
  // is enabled. An isolated non-interactive transport must never invoke one.
  delete env.PAGER
  env.GCM_INTERACTIVE = 'Never'
  env.GIT_CONFIG_GLOBAL = configNullDevice
  env.GIT_CONFIG_NOSYSTEM = '1'
  env.GIT_CONFIG_SYSTEM = configNullDevice
  env.GIT_PROTOCOL_FROM_USER = '0'
  env.GIT_TERMINAL_PROMPT = '0'
  return env
}

function normalizeHost(host: string): string {
  return host.replace(/^\[(.*)\]$/u, '$1').toLowerCase()
}

function errorMessage(error: unknown): string {
  return error instanceof Error && error.message ? error.message : String(error)
}
