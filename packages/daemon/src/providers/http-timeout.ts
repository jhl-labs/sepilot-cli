import { createHash, X509Certificate } from 'node:crypto'
import { readFileSync, statSync } from 'node:fs'
import { STATUS_CODES } from 'node:http'
import { isAbsolute, resolve } from 'node:path'
import * as tls from 'node:tls'
import {
  Agent,
  EnvHttpProxyAgent,
  fetch as undiciFetch,
  getGlobalDispatcher,
  request as undiciRequest,
  setGlobalDispatcher,
} from 'undici/index.js'
import type { Dispatcher } from 'undici'
import {
  DEFAULT_NETWORK_CONFIG,
  hasLegacyUnsafeProxyMigration,
  networkConfigSchema,
  type SepilotdConfig,
} from '../config/schema.js'

type NetworkConfig = SepilotdConfig['network']

export const MAX_CUSTOM_CA_BYTES = 1024 * 1024
export const CUSTOM_CA_UNAVAILABLE_DEGRADED_REASON = 'CUSTOM_CA_UNAVAILABLE'
export const LEGACY_PROXY_UNSAFE_DEGRADED_REASON = 'LEGACY_PROXY_UNSAFE'
export const NETWORK_PROXY_UNAVAILABLE_DEGRADED_REASON = 'NETWORK_PROXY_UNAVAILABLE'
const initialGlobalDispatcher = getGlobalDispatcher()
const retiredDispatchers = new Set<Promise<void>>()

let activeDispatcher: Dispatcher | null = null
let activePlan: ProviderDispatcherPlan | null = null
let fetchBeforeBunBridge: typeof globalThis.fetch | null = null

const bridgedUndiciFetch = undiciFetch as unknown as typeof globalThis.fetch

function responseHeadersFromUndici(
  source: Record<string, string | string[] | undefined>,
): Headers {
  const headers = new Headers()
  for (const [name, value] of Object.entries(source)) {
    if (Array.isArray(value)) {
      for (const item of value) headers.append(name, item)
    } else if (value !== undefined) {
      headers.append(name, value)
    }
  }
  return headers
}

function responseBodyFromUndici(
  body: AsyncIterable<Uint8Array> & { destroy(error?: Error): void },
): ReadableStream<Uint8Array> {
  const iterator = body[Symbol.asyncIterator]()
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const next = await iterator.next()
        if (next.done) {
          controller.close()
          return
        }
        const chunk = next.value
        controller.enqueue(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength))
      } catch (error) {
        controller.error(error)
      }
    },
    async cancel(reason) {
      body.destroy(reason instanceof Error ? reason : undefined)
      await iterator.return?.()
    },
  })
}

/**
 * Fetch adapter for provider SDKs running under Bun with the daemon's npm
 * Undici dispatcher policy.
 *
 * Bun can receive headers and bytes from `undici.fetch`, yet never deliver the
 * response-body EOF for Ollama's chunked `/api/chat` response. The SDK then
 * waits forever while health/model-list probes continue to look healthy.
 * `undici.request` does finish under Bun, so adapt its body iterator back into
 * a standards-shaped Response while retaining the configured dispatcher,
 * timeout, proxy, TLS, and cancellation policy.
 */
export const fetchWithProviderDispatcher = (async (
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> => {
  const requestInput = input instanceof Request ? input : null
  const url = requestInput?.url ?? input.toString()
  const method = (init?.method ?? requestInput?.method ?? 'GET').toUpperCase() as Dispatcher.HttpMethod
  const result = await undiciRequest(url, {
    dispatcher: activeDispatcher ?? getGlobalDispatcher(),
    method,
    headers: (init?.headers ?? requestInput?.headers) as never,
    body: (init?.body ?? requestInput?.body) as never,
    signal: init?.signal ?? requestInput?.signal,
    maxRedirections: init?.redirect === 'manual' ? 0 : 20,
  })
  const hasResponseBody = method !== 'HEAD'
    && ![101, 204, 205, 304].includes(result.statusCode)
  const response = new Response(
    hasResponseBody
      ? responseBodyFromUndici(result.body as typeof result.body & { destroy(error?: Error): void })
      : null,
    {
      status: result.statusCode,
      statusText: STATUS_CODES[result.statusCode] ?? '',
      headers: responseHeadersFromUndici(result.headers),
    },
  )
  Object.defineProperty(response, 'url', { value: url })
  return response
}) as typeof globalThis.fetch

/**
 * HTTP fetch preserving the daemon dispatcher policy and Bun body
 * compatibility. Use this for daemon-owned HTTP clients that consume a
 * response body. Under Bun, the process-wide bridge must use `undici.fetch`
 * to enforce the dispatcher, but that transport can receive headers and then
 * never deliver body EOF. The request-based adapter above preserves the same
 * network policy while providing a standards-shaped, cancellable Response.
 */
export function getDispatcherCompatibleFetch(): typeof globalThis.fetch {
  const bunRuntime = (globalThis as typeof globalThis & { Bun?: unknown }).Bun
  return bunRuntime ? fetchWithProviderDispatcher : globalThis.fetch
}

/** Backward-compatible provider SDK name for the same daemon HTTP contract. */
export function getProviderSdkFetch(): typeof globalThis.fetch {
  return getDispatcherCompatibleFetch()
}

/** Bun's native fetch does not consistently honor undici.setGlobalDispatcher.
 * Bridge only under Bun, before any configured provider/plugin can fetch. */
function ensureBunFetchDispatcherBridge(): void {
  const bunRuntime = (globalThis as typeof globalThis & { Bun?: unknown }).Bun
  if (!bunRuntime || fetchBeforeBunBridge) return
  fetchBeforeBunBridge = globalThis.fetch
  globalThis.fetch = bridgedUndiciFetch
}

function restoreBunFetchDispatcherBridge(): void {
  if (!fetchBeforeBunBridge) return
  if (globalThis.fetch === bridgedUndiciFetch) {
    globalThis.fetch = fetchBeforeBunBridge
  }
  fetchBeforeBunBridge = null
}

export class ProviderNetworkConfigurationError extends Error {
  constructor(
    readonly code: 'NETWORK_PROXY_INVALID' | 'NETWORK_CA_INVALID',
    message: string,
  ) {
    super(message)
    this.name = 'ProviderNetworkConfigurationError'
  }
}

export interface ProviderTlsOptions {
  ca?: string[]
  rejectUnauthorized: boolean
}

export interface ProviderDispatcherPlan {
  timeoutMs: number
  maxConcurrency: number
  proxyMode: NetworkConfig['proxyMode']
  useProxy: boolean
  proxyUrl: string | null
  httpProxy?: string
  httpsProxy?: string
  noProxy?: string
  customCaPath: string | null
  customCaFingerprint: string | null
  rejectUnauthorized: boolean
  degradedReason: string | null
  tlsOptions?: ProviderTlsOptions
  agentOptions: {
    headersTimeout: number
    bodyTimeout: number
    connections: number
    connect?: ProviderTlsOptions
  }
  fingerprint: string
}

interface DefaultCaSource {
  rootCertificates: readonly string[]
  getCACertificates?: (type: 'default') => string[]
}

/**
 * Node 22+ can include its effective system/default CA set in addition to the
 * bundled roots. Older runtimes expose only rootCertificates, so retain that
 * as a feature-detected fallback.
 */
export function resolveDefaultCaCertificates(
  source: DefaultCaSource = tls,
): string[] {
  if (typeof source.getCACertificates === 'function') {
    try {
      const certificates = source.getCACertificates('default')
      if (certificates.length > 0) return [...new Set(certificates)]
    } catch {
      // Fall through for early/partial Node implementations.
    }
  }
  return [...new Set(source.rootCertificates)]
}

export interface ProviderNetworkOverrideSources {
  tlsRejectUnauthorized: 'SEPILOTD_TLS_REJECT_UNAUTHORIZED' | 'NODE_TLS_REJECT_UNAUTHORIZED' | null
  timeoutMs: 'SEPILOTD_PROVIDER_HTTP_TIMEOUT_MS' | null
  customCaPath: 'SEPILOTD_EXTRA_CA_CERTS' | 'NODE_EXTRA_CA_CERTS' | null
}

export type ProviderNetworkIgnoredOverrideSources = ProviderNetworkOverrideSources

type TimeoutEnvironmentOverride = {
  source: 'SEPILOTD_PROVIDER_HTTP_TIMEOUT_MS'
  value: number | null
}

function resolveTimeoutEnvironmentOverride(
  env: NodeJS.ProcessEnv,
): TimeoutEnvironmentOverride | null {
  const raw = env.SEPILOTD_PROVIDER_HTTP_TIMEOUT_MS?.trim()
  if (!raw) return null
  const parsed = Number(raw)
  return {
    source: 'SEPILOTD_PROVIDER_HTTP_TIMEOUT_MS',
    value: Number.isFinite(parsed) && parsed >= 0 ? parsed : null,
  }
}

function resolveEnvironmentCaSource(
  config: NetworkConfig,
  env: NodeJS.ProcessEnv,
): ProviderNetworkOverrideSources['customCaPath'] {
  if (config.customCaPath) return null
  if (env.SEPILOTD_EXTRA_CA_CERTS?.trim()) return 'SEPILOTD_EXTRA_CA_CERTS'
  if (env.NODE_EXTRA_CA_CERTS?.trim()) return 'NODE_EXTRA_CA_CERTS'
  return null
}

/**
 * Return only the names of environment variables that override persisted
 * network settings. Values are deliberately omitted so diagnostics can never
 * expose proxy or filesystem secrets.
 */
export function getProviderNetworkOverrideSources(
  config: NetworkConfig,
  env: NodeJS.ProcessEnv = process.env,
  effectivePlan?: ProviderDispatcherPlan,
): ProviderNetworkOverrideSources {
  const scopedTls = env.SEPILOTD_TLS_REJECT_UNAUTHORIZED?.trim().toLowerCase()
  const tlsRejectUnauthorized = scopedTls === '0' || scopedTls === 'false'
    ? 'SEPILOTD_TLS_REJECT_UNAUTHORIZED'
    : env.NODE_TLS_REJECT_UNAUTHORIZED === '0'
      ? 'NODE_TLS_REJECT_UNAUTHORIZED'
      : null
  const timeoutOverride = resolveTimeoutEnvironmentOverride(env)
  const timeoutMs = timeoutOverride && timeoutOverride.value !== null
    ? timeoutOverride.source
    : null
  const declaredCaSource = resolveEnvironmentCaSource(config, env)
  const customCaPath = declaredCaSource && effectivePlan?.customCaPath === null
    ? null
    : declaredCaSource

  return { tlsRejectUnauthorized, timeoutMs, customCaPath }
}

/** Environment inputs that were declared but rejected, without their values. */
export function getProviderNetworkIgnoredOverrideSources(
  config: NetworkConfig,
  effectivePlan: ProviderDispatcherPlan,
  env: NodeJS.ProcessEnv = process.env,
): ProviderNetworkIgnoredOverrideSources {
  const timeoutOverride = resolveTimeoutEnvironmentOverride(env)
  const declaredCaSource = resolveEnvironmentCaSource(config, env)
  return {
    tlsRejectUnauthorized: null,
    timeoutMs: timeoutOverride && timeoutOverride.value === null
      ? timeoutOverride.source
      : null,
    customCaPath: declaredCaSource && effectivePlan.customCaPath === null
      ? declaredCaSource
      : null,
  }
}

function firstEnvironmentValue(
  env: NodeJS.ProcessEnv,
  lowercase: string,
  uppercase: string,
): string | undefined {
  const value = env[lowercase] ?? env[uppercase]
  return value?.trim() || undefined
}

function validateProxyUrl(
  raw: string,
  source: string,
  options: { allowUserInfo: boolean },
): string {
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    throw new ProviderNetworkConfigurationError(
      'NETWORK_PROXY_INVALID',
      `${source} must be a valid HTTP(S) proxy URL.`,
    )
  }

  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new ProviderNetworkConfigurationError(
      'NETWORK_PROXY_INVALID',
      `${source} must use http:// or https://.`,
    )
  }
  if (!options.allowUserInfo && (url.username || url.password)) {
    throw new ProviderNetworkConfigurationError(
      'NETWORK_PROXY_INVALID',
      `${source} must not contain a username or password.`,
    )
  }
  return url.toString()
}

function resolveEnvironmentProxy(
  env: NodeJS.ProcessEnv,
  configuredNoProxy: string | null,
): {
  httpProxy?: string
  httpsProxy?: string
  noProxy?: string
} {
  // undici 6's EnvHttpProxyAgent does not read ALL_PROXY itself. Normalize it
  // here so existing daemon deployments keep the documented fallback.
  const allProxy = firstEnvironmentValue(env, 'all_proxy', 'ALL_PROXY')
  const rawHttpProxy = firstEnvironmentValue(env, 'http_proxy', 'HTTP_PROXY') ?? allProxy
  const rawHttpsProxy = firstEnvironmentValue(env, 'https_proxy', 'HTTPS_PROXY') ?? allProxy
  const noProxy = configuredNoProxy
    ?? firstEnvironmentValue(env, 'no_proxy', 'NO_PROXY')

  return {
    ...(rawHttpProxy
      ? {
          httpProxy: validateProxyUrl(rawHttpProxy, 'HTTP_PROXY', {
            // Environment variables are an operator-controlled compatibility
            // surface. Manual API values remain credential-free.
            allowUserInfo: true,
          }),
        }
      : {}),
    ...(rawHttpsProxy
      ? {
          httpsProxy: validateProxyUrl(rawHttpsProxy, 'HTTPS_PROXY', {
            allowUserInfo: true,
          }),
        }
      : {}),
    ...(noProxy ? { noProxy } : {}),
  }
}

function resolveTimeoutMs(config: NetworkConfig, env: NodeJS.ProcessEnv): number {
  const override = resolveTimeoutEnvironmentOverride(env)
  // Keep the established env-only escape hatch where 0 disables undici's
  // headers/body timeout. The persisted UI contract stays bounded at >= 1s.
  return override?.value ?? config.timeoutMs
}

function shouldRejectUnauthorized(config: NetworkConfig, env: NodeJS.ProcessEnv): boolean {
  const scoped = env.SEPILOTD_TLS_REJECT_UNAUTHORIZED?.trim().toLowerCase()
  if (scoped === '0' || scoped === 'false') return false
  if (env.NODE_TLS_REJECT_UNAUTHORIZED === '0') return false
  return config.tlsRejectUnauthorized
}

function resolveCustomCaPath(config: NetworkConfig, env: NodeJS.ProcessEnv): {
  path: string | null
  configuredByApi: boolean
} {
  if (config.customCaPath) {
    return { path: config.customCaPath, configuredByApi: true }
  }
  const scopedPath = env.SEPILOTD_EXTRA_CA_CERTS?.trim()
  if (scopedPath) return { path: scopedPath, configuredByApi: false }
  const nodePath = env.NODE_EXTRA_CA_CERTS?.trim()
  return { path: nodePath || null, configuredByApi: false }
}

type LoadedCustomCa = {
  path: string | null
  certificates: string[]
  fingerprint: string | null
}

const EMPTY_CUSTOM_CA: LoadedCustomCa = {
  path: null,
  certificates: [],
  fingerprint: null,
}

function isUncOrNetworkPath(path: string): boolean {
  // Covers Windows UNC (\\server\share), slash-normalized UNC
  // (//server/share), and device/extended namespaces (\\?\ / \\.\). These
  // can make synchronous stat/read calls wait on a remote endpoint or driver.
  return /^(?:\\\\|\/\/)/.test(path)
}

function loadCustomCaStrict(
  configuredPath: string | null,
  configuredByApi: boolean,
): LoadedCustomCa {
  if (!configuredPath) {
    return EMPTY_CUSTOM_CA
  }
  if (isUncOrNetworkPath(configuredPath)) {
    throw new ProviderNetworkConfigurationError(
      'NETWORK_CA_INVALID',
      'Custom CA must use a local filesystem path; UNC, network, and device paths are not supported because they can block the daemon. Copy the PEM file to a local path.',
    )
  }
  if (configuredByApi && !isAbsolute(configuredPath)) {
    throw new ProviderNetworkConfigurationError(
      'NETWORK_CA_INVALID',
      'network.customCaPath must be an absolute path on the daemon host.',
    )
  }

  const absolutePath = resolve(configuredPath)
  let size: number
  try {
    const stats = statSync(absolutePath)
    if (!stats.isFile()) {
      throw new Error('path is not a regular file')
    }
    size = stats.size
  } catch (error) {
    throw new ProviderNetworkConfigurationError(
      'NETWORK_CA_INVALID',
      `Cannot read custom CA file "${absolutePath}": ${error instanceof Error ? error.message : String(error)}`,
    )
  }
  if (size > MAX_CUSTOM_CA_BYTES) {
    throw new ProviderNetworkConfigurationError(
      'NETWORK_CA_INVALID',
      `Custom CA file "${absolutePath}" exceeds the 1 MiB size limit.`,
    )
  }

  let pem: string
  try {
    pem = readFileSync(absolutePath, 'utf8')
  } catch (error) {
    throw new ProviderNetworkConfigurationError(
      'NETWORK_CA_INVALID',
      `Cannot read custom CA file "${absolutePath}": ${error instanceof Error ? error.message : String(error)}`,
    )
  }

  if (/-----BEGIN [^-\r\n]*PRIVATE KEY-----/.test(pem)) {
    throw new ProviderNetworkConfigurationError(
      'NETWORK_CA_INVALID',
      `Custom CA file "${absolutePath}" contains private key material; provide certificates only.`,
    )
  }

  const certificates = pem.match(/-----BEGIN CERTIFICATE-----[\s\S]*?-----END CERTIFICATE-----/g) ?? []
  if (certificates.length === 0) {
    throw new ProviderNetworkConfigurationError(
      'NETWORK_CA_INVALID',
      `Custom CA file "${absolutePath}" does not contain a PEM certificate.`,
    )
  }
  try {
    for (const certificate of certificates) {
      void new X509Certificate(certificate)
    }
  } catch (error) {
    throw new ProviderNetworkConfigurationError(
      'NETWORK_CA_INVALID',
      `Custom CA file "${absolutePath}" contains an invalid certificate: ${error instanceof Error ? error.message : String(error)}`,
    )
  }

  return {
    path: absolutePath,
    certificates,
    fingerprint: createHash('sha256').update(pem).digest('hex'),
  }
}

function loadCustomCa(
  configuredPath: string | null,
  configuredByApi: boolean,
): LoadedCustomCa {
  try {
    return loadCustomCaStrict(configuredPath, configuredByApi)
  } catch (error) {
    if (configuredByApi) throw error
    // Environment CA variables predate the settings contract and may point at
    // a removed/rotated file. Preserve upgrade availability by falling back to
    // Node's built-in roots. Persisted network.customCaPath stays strict so a
    // new API write receives a clear 400 and startup can explicitly enter its
    // fail-closed degraded state.
    return EMPTY_CUSTOM_CA
  }
}

/**
 * Resolve the process-wide undici dispatcher policy without changing global
 * state. The resulting dispatcher affects every daemon call made through
 * global `fetch`, including provider discovery/probes and non-LLM fetch users;
 * callers with an explicit dispatcher remain isolated from it.
 */
export function planProviderDispatcher(
  input: Partial<NetworkConfig> = DEFAULT_NETWORK_CONFIG,
  env: NodeJS.ProcessEnv = process.env,
): ProviderDispatcherPlan {
  const legacyUnsafeProxy = hasLegacyUnsafeProxyMigration(input)
  const config = networkConfigSchema.parse(input)
  const timeoutMs = resolveTimeoutMs(config, env)
  if (legacyUnsafeProxy) {
    return planBlockedProviderDispatcher(
      config,
      LEGACY_PROXY_UNSAFE_DEGRADED_REASON,
      timeoutMs,
    )
  }
  const rejectUnauthorized = shouldRejectUnauthorized(config, env)
  const customCa = loadCustomCa(...(() => {
    const resolved = resolveCustomCaPath(config, env)
    return [resolved.path, resolved.configuredByApi] as const
  })())

  // Supplying `ca` to tls.connect replaces Node's default trust roots. Merge
  // the custom bundle with the built-in roots so this remains an *extra* CA.
  const tlsOptions: ProviderTlsOptions | undefined =
    customCa.certificates.length > 0 || !rejectUnauthorized
      ? {
          ...(customCa.certificates.length > 0
            ? {
                ca: [
                  ...resolveDefaultCaCertificates(),
                  ...customCa.certificates,
                ],
              }
            : {}),
          rejectUnauthorized,
        }
      : undefined

  let proxyUrl: string | null = null
  let environmentProxy: ReturnType<typeof resolveEnvironmentProxy> = {}
  if (config.proxyMode === 'manual') {
    proxyUrl = validateProxyUrl(config.proxyUrl!, 'network.proxyUrl', {
      allowUserInfo: false,
    })
    // EnvHttpProxyAgent also gives a manual proxy a real NO_PROXY bypass
    // branch. A bare ProxyAgent cannot express per-target bypasses.
    environmentProxy = {
      httpProxy: proxyUrl,
      httpsProxy: proxyUrl,
      ...(config.noProxy ? { noProxy: config.noProxy } : {}),
    }
  } else if (config.proxyMode === 'environment') {
    environmentProxy = resolveEnvironmentProxy(env, config.noProxy)
  }

  const useProxy = config.proxyMode === 'manual'
    || Boolean(environmentProxy.httpProxy || environmentProxy.httpsProxy)
  const fingerprint = createHash('sha256')
    .update(JSON.stringify({
      timeoutMs,
      maxConcurrency: config.maxConcurrency,
      proxyMode: config.proxyMode,
      proxyUrl,
      ...environmentProxy,
      customCaPath: customCa.path,
      customCaFingerprint: customCa.fingerprint,
      rejectUnauthorized,
      degradedReason: null,
    }))
    .digest('hex')

  return {
    timeoutMs,
    maxConcurrency: config.maxConcurrency,
    proxyMode: config.proxyMode,
    useProxy,
    proxyUrl,
    ...environmentProxy,
    customCaPath: customCa.path,
    customCaFingerprint: customCa.fingerprint,
    rejectUnauthorized,
    degradedReason: null,
    tlsOptions,
    agentOptions: {
      headersTimeout: timeoutMs,
      bodyTimeout: timeoutMs,
      connections: config.maxConcurrency,
      ...(tlsOptions ? { connect: tlsOptions } : {}),
    },
    fingerprint,
  }
}

export class ProviderNetworkEgressBlockedError extends Error {
  readonly code = 'NETWORK_EGRESS_BLOCKED' as const

  constructor(readonly degradedReason: string) {
    super(`Provider network egress is blocked (${degradedReason}).`)
    this.name = 'ProviderNetworkEgressBlockedError'
  }
}

function createBlockedProviderDispatcher(degradedReason: string): Dispatcher {
  let closed = false
  return {
    dispatch(_options, handler) {
      if (typeof handler.onError !== 'function') {
        throw new TypeError('Blocked dispatcher requires an onError handler.')
      }
      const onError = handler.onError.bind(handler)
      const error = closed
        ? new ProviderNetworkEgressBlockedError('DISPATCHER_CLOSED')
        : new ProviderNetworkEgressBlockedError(degradedReason)
      queueMicrotask(() => onError(error))
      return false
    },
    async close() {
      closed = true
    },
    async destroy() {
      closed = true
    },
  } as Dispatcher
}

export function createProviderDispatcher(plan: ProviderDispatcherPlan): Dispatcher {
  if (plan.degradedReason) {
    return createBlockedProviderDispatcher(plan.degradedReason)
  }
  const commonOptions = {
    headersTimeout: plan.timeoutMs,
    bodyTimeout: plan.timeoutMs,
    connections: plan.maxConcurrency,
  }
  // Environment mode with no effective proxy is a direct connection. Using
  // EnvHttpProxyAgent anyway is not only unnecessary: under Bun's global
  // undici bridge it can leave streaming response bodies pending forever
  // (notably Ollama /api/chat), even though ordinary non-streaming probes such
  // as /api/tags still succeed. Keep EnvHttpProxyAgent only for plans that
  // actually have a proxy route; the plain Agent enforces the same timeout,
  // concurrency, and TLS policy for direct traffic.
  if (!plan.useProxy) {
    return new Agent({
      ...commonOptions,
      ...(plan.tlsOptions ? { connect: plan.tlsOptions } : {}),
    })
  }
  // EnvHttpProxyAgent provides NO_PROXY routing for both environment and
  // manual proxy modes. It forwards unknown agent options into its internal
  // ProxyAgent at runtime. Supplying both `connect` (direct/no_proxy) and
  // `requestTls`/`proxyTls` (proxied target/proxy) is required for custom CA
  // and rejectUnauthorized to work in every environment branch.
  const environmentOptions = {
    ...commonOptions,
    // Freeze the environment snapshot captured by planProviderDispatcher.
    // Explicit empty values prevent a later process.env mutation from leaking
    // into this dispatcher (and prevent manual mode from inheriting NO_PROXY).
    httpProxy: plan.httpProxy ?? '',
    httpsProxy: plan.httpsProxy ?? '',
    noProxy: plan.noProxy ?? '',
    ...(plan.tlsOptions
      ? {
          connect: plan.tlsOptions,
          requestTls: plan.tlsOptions,
          proxyTls: plan.tlsOptions,
        }
      : {}),
  }
  return new EnvHttpProxyAgent(
    environmentOptions as EnvHttpProxyAgent.Options & {
      requestTls?: ProviderTlsOptions
      proxyTls?: ProviderTlsOptions
    },
  )
}

function retireDispatcher(dispatcher: Dispatcher): void {
  const closing = Promise.resolve(dispatcher.close())
    .catch(() => undefined)
    .finally(() => retiredDispatchers.delete(closing))
  retiredDispatchers.add(closing)
}

/**
 * Atomically replace the daemon process-wide undici dispatcher. Existing
 * in-flight requests keep their old dispatcher while it drains; new global
 * fetch calls immediately use the new policy.
 *
 * Returns the applied timeout, or null when the effective policy is unchanged.
 * The legacy name is retained because startup/tests already import it.
 */
export function configureProviderHttpTimeout(
  config: Partial<NetworkConfig> = DEFAULT_NETWORK_CONFIG,
): number | null {
  const plan = planProviderDispatcher(config)
  ensureBunFetchDispatcherBridge()
  return activateProviderDispatcher(plan)
}

function activateProviderDispatcher(plan: ProviderDispatcherPlan): number | null {
  if (activePlan?.fingerprint === plan.fingerprint) return null

  // Build first. A malformed proxy/CA must not displace a working dispatcher.
  const nextDispatcher = createProviderDispatcher(plan)
  const previousDispatcher = activeDispatcher
  setGlobalDispatcher(nextDispatcher)
  activeDispatcher = nextDispatcher
  activePlan = plan
  if (previousDispatcher && previousDispatcher !== nextDispatcher) {
    retireDispatcher(previousDispatcher)
  }
  return plan.timeoutMs
}

/**
 * Install a fail-closed dispatcher while keeping the daemon control plane
 * reachable. No proxy/direct/environment path is consulted and every global
 * fetch dispatch fails locally until a valid network policy is saved.
 */
export function configureProviderNetworkBlocked(
  input: Partial<NetworkConfig>,
  degradedReason: string,
): number | null {
  const config = networkConfigSchema.parse(input)
  const timeoutMs = resolveTimeoutMs(config, process.env)
  const plan = planBlockedProviderDispatcher(config, degradedReason, timeoutMs)
  ensureBunFetchDispatcherBridge()
  return activateProviderDispatcher(plan)
}

function planBlockedProviderDispatcher(
  config: NetworkConfig,
  degradedReason: string,
  timeoutMs: number,
): ProviderDispatcherPlan {
  return {
    timeoutMs,
    maxConcurrency: config.maxConcurrency,
    proxyMode: config.proxyMode,
    useProxy: false,
    proxyUrl: config.proxyUrl,
    customCaPath: null,
    customCaFingerprint: null,
    rejectUnauthorized: true,
    degradedReason,
    agentOptions: {
      headersTimeout: timeoutMs,
      bodyTimeout: timeoutMs,
      connections: config.maxConcurrency,
    },
    fingerprint: createHash('sha256')
      .update(JSON.stringify({
        blocked: true,
        degradedReason,
        timeoutMs,
        maxConcurrency: config.maxConcurrency,
        proxyMode: config.proxyMode,
      }))
      .digest('hex'),
  }
}

export function getProviderHttpDispatcher(): Dispatcher | null {
  return activeDispatcher
}

export function getProviderDispatcherPlan(): ProviderDispatcherPlan | null {
  return activePlan
}

/** Restore the dispatcher that was present before daemon network setup. */
export async function closeProviderHttpDispatcher(): Promise<void> {
  const current = activeDispatcher
  activeDispatcher = null
  activePlan = null
  if (current) {
    setGlobalDispatcher(initialGlobalDispatcher)
    await Promise.resolve(current.close()).catch(() => undefined)
  }
  await Promise.allSettled([...retiredDispatchers])
  restoreBunFetchDispatcherBridge()
}
