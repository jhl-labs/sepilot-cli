import { promises as dns } from 'node:dns'
import { connect as connectNet, isIP, type LookupFunction } from 'node:net'
import { connect as connectTls } from 'node:tls'
import { Agent, fetch as undiciFetch, type Dispatcher } from 'undici/index.js'
import { raceWithAbort, throwIfAborted } from '../abort.js'

export interface PublicUrlResolution {
  url: URL
  hostname: string
  address: string
  family: 4 | 6
}

const pinnedDispatcherResolutions = new WeakMap<object, PublicUrlResolution>()

function isPrivateIpv4(ip: string): boolean {
  const parts = ip.split('.').map(Number)
  if (parts.length !== 4) return true
  if (parts.some((p) => Number.isNaN(p) || p < 0 || p > 255)) return true
  const [a, b, c] = parts as [number, number, number, number]
  if (a === 0) return true // 0.0.0.0/8
  if (a === 10) return true // 10.0.0.0/8
  if (a === 127) return true // loopback
  if (a === 169 && b === 254) return true // link-local + cloud metadata
  if (a === 172 && b >= 16 && b <= 31) return true // 172.16/12
  if (a === 192 && b === 168) return true // 192.168/16
  if (a === 100 && b >= 64 && b <= 127) return true // 100.64/10 CGNAT
  if (a === 192 && b === 0 && c === 0) return true // 192.0.0/24 IETF
  if (a === 192 && b === 0 && c === 2) return true // 192.0.2/24 documentation
  if (a === 192 && b === 88 && c === 99) return true // deprecated 6to4 relay anycast
  if (a === 198 && (b === 18 || b === 19)) return true // 198.18/15 benchmark
  if (a === 198 && b === 51 && c === 100) return true // 198.51.100/24 documentation
  if (a === 203 && b === 0 && c === 113) return true // 203.0.113/24 documentation
  if (a >= 224) return true // multicast/reserved
  return false
}

function isPrivateIpv6(ip: string): boolean {
  const lower = ip.toLowerCase()
  if (lower === '::1' || lower === '::') return true
  if (/^f[cd]/.test(lower)) return true // fc00::/7 ULA
  if (/^fe[89ab]/.test(lower)) return true // fe80::/10 link-local
  if (/^fe[c-f]/.test(lower)) return true // fec0::/10 deprecated site-local
  if (lower === '2001:db8' || lower.startsWith('2001:db8:')) return true // documentation
  if (lower === '2002' || lower.startsWith('2002:')) return true // deprecated 6to4 transition range
  if (lower === '64:ff9b:1::' || lower.startsWith('64:ff9b:1:')) return true // local-use NAT64
  if (lower.startsWith('ff')) return true // multicast
  const dottedMapped = lower.match(/^::ffff:([\d.]+)$/)
  if (dottedMapped?.[1]) return isPrivateIpv4(dottedMapped[1])
  // WHATWG URL parsing canonicalizes IPv4-mapped literals to hexadecimal
  // (for example ::ffff:127.0.0.1 becomes ::ffff:7f00:1). Treat the final
  // 32 bits as IPv4 so loopback/link-local addresses cannot bypass the guard.
  const hexMapped = lower.match(/^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/)
  if (hexMapped?.[1] && hexMapped[2]) {
    return isPrivateEmbeddedIpv4(hexMapped[1], hexMapped[2])
  }
  // RFC 6052 well-known NAT64 prefix. A translator can otherwise turn the
  // literal into a connection to loopback, metadata, or another private IPv4.
  const wellKnownNat64 = lower.match(/^64:ff9b::([0-9a-f]{1,4}):([0-9a-f]{1,4})$/u)
  if (wellKnownNat64?.[1] && wellKnownNat64[2]) {
    return isPrivateEmbeddedIpv4(wellKnownNat64[1], wellKnownNat64[2])
  }
  // IPv4-translatable addresses are reserved for translation within a host
  // or network and must not be accepted as public outbound destinations.
  if (/^::ffff:0:[0-9a-f]{1,4}:[0-9a-f]{1,4}$/u.test(lower)) return true
  // Deprecated IPv4-compatible IPv6 addresses (::/96) are not globally
  // routable IPv6 destinations. Reject the whole range instead of allowing
  // alternate spellings of loopback or link-local IPv4 targets.
  if (/^::[0-9a-f]{1,4}:[0-9a-f]{1,4}$/u.test(lower)) return true
  return false
}

function isPrivateEmbeddedIpv4(highWord: string, lowWord: string): boolean {
  const high = Number.parseInt(highWord, 16)
  const low = Number.parseInt(lowWord, 16)
  return isPrivateIpv4([high >>> 8, high & 0xff, low >>> 8, low & 0xff].join('.'))
}

export function isPrivateAddress(value: string): boolean {
  const kind = isIP(value)
  if (kind === 4) return isPrivateIpv4(value)
  if (kind === 6) return isPrivateIpv6(value)
  return false
}

/**
 * Block obvious SSRF targets when accepting a caller-controlled URL.
 * Resolves the hostname and rejects when any returned address is private,
 * loopback, link-local, CGNAT, or multicast. Returns the validated address
 * so callers can pin the subsequent connection to the same DNS result.
 * For redirect chains the caller must call `assertPublicUrl` on each hop.
 */
export async function assertPublicUrl(
  rawUrl: string,
  signal?: AbortSignal,
): Promise<PublicUrlResolution> {
  throwIfAborted(signal)
  let parsed: URL
  try {
    parsed = new URL(rawUrl)
  } catch {
    throw new Error('invalid url')
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error('only http(s) urls are allowed')
  }
  if (parsed.username || parsed.password) {
    throw new Error('url credentials are not allowed')
  }
  const host = parsed.hostname.replace(/^\[(.*)\]$/, '$1')
  if (!host) throw new Error('missing host')
  const literalFamily = isIP(host)
  if (literalFamily) {
    if (isPrivateAddress(host)) {
      throw new Error('refusing private/loopback host')
    }
    return {
      url: parsed,
      hostname: host,
      address: host,
      family: literalFamily === 4 ? 4 : 6,
    }
  }
  let resolved: { address: string }[]
  try {
    resolved = await raceWithAbort(dns.lookup(host, { all: true }), signal)
  } catch (error) {
    if (signal?.aborted) throw error
    throw new Error('dns lookup failed')
  }
  let selected: PublicUrlResolution | null = null
  for (const entry of resolved) {
    const family = isIP(entry.address)
    if (!family) {
      throw new Error('dns lookup failed')
    }
    if (isPrivateAddress(entry.address)) {
      throw new Error('refusing private/loopback host')
    }
    selected ??= {
      url: parsed,
      hostname: host,
      address: entry.address,
      family: family === 4 ? 4 : 6,
    }
  }
  if (!selected) throw new Error('dns lookup failed')
  return selected
}

/**
 * Pin an outbound connection to the address already approved by
 * `assertPublicUrl`. This closes the validation/connect DNS-rebinding window.
 * Callers own the returned dispatcher and must close it after consuming the
 * response body.
 */
export function createPinnedLookupDispatcher(resolution: PublicUrlResolution): Agent {
  const expectedHost = normalizeLookupHost(resolution.hostname)
  const lookup: LookupFunction = (hostname, options, callback) => {
    if (normalizeLookupHost(hostname) !== expectedHost) {
      const error = new Error('pinned DNS lookup host mismatch') as NodeJS.ErrnoException
      error.code = 'ERR_PINNED_DNS_HOST_MISMATCH'
      callback(error, '', 0)
      return
    }
    if (options.all) {
      callback(null, [{ address: resolution.address, family: resolution.family }])
      return
    }
    callback(null, resolution.address, resolution.family)
  }
  const dispatcher = new Agent({
    connect: {
      lookup,
    },
  })
  pinnedDispatcherResolutions.set(dispatcher, resolution)
  return dispatcher
}

/**
 * Build a direct dispatcher for a localhost origin that was independently
 * authorized by a managed-process session. The daemon's global dispatcher may
 * be a provider proxy, so inheriting it would make localhost unreachable and
 * could disclose a local URL to that proxy.
 */
export function createManagedLoopbackDispatcher(rawUrl: string): Agent {
  const url = new URL(rawUrl)
  const hostname = normalizeLookupHost(url.hostname)
  const address = hostname === '::1'
    ? '::1'
    : hostname === 'localhost' || hostname === '127.0.0.1'
      ? '127.0.0.1'
      : null
  if (!address) throw new Error('managed loopback dispatcher requires a localhost URL')
  return createPinnedLookupDispatcher({
    url,
    hostname,
    address,
    family: address === '::1' ? 6 : 4,
  })
}

type PinnedFetchInit = RequestInit & {
  dispatcher?: Dispatcher
  managedLoopbackSocketPath?: string
}
const MAX_PINNED_FETCH_BYTES = 32 * 1024 * 1024

function decodeChunkedHttpBody(body: Buffer): Buffer {
  const chunks: Buffer[] = []
  let offset = 0
  let totalBytes = 0
  while (offset < body.length) {
    const lineEnd = body.indexOf('\r\n', offset)
    if (lineEnd < 0) throw new Error('Malformed chunked localhost response')
    const sizeText = body.subarray(offset, lineEnd).toString('ascii').split(';', 1)[0]?.trim() ?? ''
    if (!/^[0-9a-f]+$/iu.test(sizeText)) throw new Error('Malformed chunk size')
    const size = Number.parseInt(sizeText, 16)
    offset = lineEnd + 2
    if (size === 0) break
    if (offset + size + 2 > body.length) throw new Error('Truncated chunked localhost response')
    const chunk = body.subarray(offset, offset + size)
    chunks.push(chunk)
    totalBytes += chunk.length
    offset += size
    if (body.subarray(offset, offset + 2).toString('ascii') !== '\r\n') {
      throw new Error('Malformed chunk terminator')
    }
    offset += 2
  }
  return Buffer.concat(chunks, totalBytes)
}

function parsePinnedHttpResponse(
  raw: Buffer,
  method: string,
): Response {
  const headerEnd = raw.indexOf('\r\n\r\n')
  if (headerEnd < 0) {
    throw new Error(`Malformed localhost HTTP response (${raw.length} bytes)`)
  }
  const headerLines = raw.subarray(0, headerEnd).toString('latin1').split('\r\n')
  const statusLine = headerLines.shift() ?? ''
  const statusMatch = statusLine.match(/^HTTP\/\d(?:\.\d)?\s+(\d{3})(?:\s+(.*))?$/u)
  if (!statusMatch?.[1]) throw new Error('Malformed localhost HTTP status line')
  const status = Number(statusMatch[1])
  const headers = new Headers()
  for (const line of headerLines) {
    const separator = line.indexOf(':')
    if (separator <= 0) continue
    headers.append(line.slice(0, separator).trim(), line.slice(separator + 1).trim())
  }
  let body = raw.subarray(headerEnd + 4)
  if (/\bchunked\b/iu.test(headers.get('transfer-encoding') ?? '')) {
    body = decodeChunkedHttpBody(body)
    headers.delete('transfer-encoding')
    headers.set('content-length', String(body.length))
  } else {
    const declaredLength = headers.get('content-length')
    if (declaredLength && /^\d+$/u.test(declaredLength)) {
      const expected = Number(declaredLength)
      if (body.length < expected) throw new Error('Truncated localhost HTTP response')
      body = body.subarray(0, expected)
    }
  }
  const hasBody = method !== 'HEAD' && ![204, 205, 304].includes(status)
  return new Response(hasBody ? body : null, {
    status,
    statusText: statusMatch[2] ?? '',
    headers,
  })
}

async function fetchPinnedWithSocket(
  parsed: URL,
  init: RequestInit | undefined,
  resolution: PublicUrlResolution,
  socketPath?: string,
): Promise<Response> {
  throwIfAborted(init?.signal, 'Pinned request aborted')
  const hostname = normalizeLookupHost(parsed.hostname)
  if (hostname !== normalizeLookupHost(resolution.hostname)) {
    throw new Error('pinned DNS lookup host mismatch')
  }
  const address = resolution.address
  const headers = new Headers(init?.headers)
  if (!headers.has('host')) headers.set('host', parsed.host)
  headers.set('connection', 'close')
  const body = init?.body == null
    ? undefined
    : typeof init.body === 'string' || init.body instanceof Uint8Array
      ? init.body
      : Buffer.from(await new Response(init.body).arrayBuffer())
  throwIfAborted(init?.signal, 'Pinned request aborted')

  const method = (init?.method ?? 'GET').toUpperCase()
  if (body && !headers.has('content-length')) headers.set('content-length', String(body.length))
  const requestHead = [
    `${method} ${parsed.pathname}${parsed.search} HTTP/1.1`,
    ...[...headers.entries()].map(([name, value]) => `${name}: ${value}`),
    '',
    '',
  ].join('\r\n')

  return new Promise<Response>((resolve, reject) => {
    const chunks: Buffer[] = []
    let totalBytes = 0
    let settled = false
    const baseSocket = socketPath ? connectNet({ path: socketPath }) : null
    const socket = parsed.protocol === 'https:'
      ? connectTls({
          ...(baseSocket
            ? { socket: baseSocket }
            : { host: address, port: Number(parsed.port || 443) }),
          servername: hostname,
        })
      : baseSocket ?? connectNet({ host: address, port: Number(parsed.port || 80) })
    const finishError = (error: Error) => {
      if (settled) return
      settled = true
      init?.signal?.removeEventListener('abort', onAbort)
      reject(error)
    }
    const onAbort = () => socket.destroy(init?.signal?.reason instanceof Error
      ? init.signal.reason
      : new Error('Pinned request aborted'))
    init?.signal?.addEventListener('abort', onAbort, { once: true })
    socket.on('data', (chunk: Buffer | Uint8Array | string) => {
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
      totalBytes += buffer.byteLength
      if (totalBytes > MAX_PINNED_FETCH_BYTES + 64 * 1024) {
        socket.destroy(new Error(
          `Pinned response exceeds ${MAX_PINNED_FETCH_BYTES} bytes`,
        ))
        return
      }
      chunks.push(buffer)
    })
    socket.once('error', finishError)
    socket.once('end', () => {
      // Bun's node:net compatibility layer can leave the writable half open
      // after the peer ends. Close it explicitly so the Unix proxy releases
      // the corresponding upstream connection before the next request.
      socket.destroy()
      init?.signal?.removeEventListener('abort', onAbort)
      if (settled) return
      try {
        const response = parsePinnedHttpResponse(
          Buffer.concat(chunks, totalBytes),
          method,
        )
        settled = true
        resolve(response)
      } catch (error) {
        finishError(error instanceof Error ? error : new Error(String(error)))
      }
    })
    const connectedEvent = parsed.protocol === 'https:' ? 'secureConnect' : 'connect'
    socket.once(connectedEvent, () => {
      socket.write(body ? Buffer.concat([Buffer.from(requestHead, 'latin1'), Buffer.from(body)]) : requestHead)
    })
  })
}

/**
 * Execute a request whose destination has already been validated and pinned by
 * the caller. Bun's npm-undici compatibility layer drops a later connection
 * made through a custom lookup dispatcher, so managed localhost requests use
 * a one-shot raw socket request with the validated literal address and no
 * pool. The same fallback is required for public pinned destinations in Bun:
 * its npm-undici compatibility layer can leave a request through a custom
 * lookup dispatcher pending forever. Node keeps using the supplied Undici
 * dispatcher normally.
 */
export const fetchWithPinnedNetworkPolicy: typeof fetch = async (input, init) => {
  const rawUrl = typeof input === 'string'
    ? input
    : input instanceof URL
      ? input.toString()
      : input.url
  const parsed = new URL(rawUrl)
  const hostname = normalizeLookupHost(parsed.hostname)
  const isManagedLoopback = hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1'
  const isBun = Boolean((globalThis as typeof globalThis & { Bun?: unknown }).Bun)
  const pinnedInit = (init ?? {}) as PinnedFetchInit
  const pinnedResolution = pinnedInit.dispatcher
    ? pinnedDispatcherResolutions.get(pinnedInit.dispatcher as object)
    : undefined

  if (
    (isManagedLoopback && (isBun || pinnedInit.managedLoopbackSocketPath))
    || (isBun && pinnedResolution)
  ) {
    const {
      dispatcher: _dispatcher,
      managedLoopbackSocketPath,
      ...directInit
    } = pinnedInit
    const resolution = pinnedResolution ?? {
      url: parsed,
      hostname,
      address: hostname === '::1' ? '::1' : '127.0.0.1',
      family: hostname === '::1' ? 6 as const : 4 as const,
    }
    return fetchPinnedWithSocket(parsed, directInit, resolution, managedLoopbackSocketPath)
  }

  const runtimeFetch = isBun ? undiciFetch as unknown as typeof fetch : globalThis.fetch
  return runtimeFetch(input, init)
}

function normalizeLookupHost(hostname: string): string {
  return hostname.replace(/^\[(.*)\]$/, '$1').toLowerCase()
}

/**
 * Synchronous variant: rejects only when the hostname is a literal private
 * IP. Hostnames pass through unchecked. Use where async DNS lookup is not
 * possible (e.g. sync schema validators). Defends against the cheap path
 * (`http://127.0.0.1/...`, `http://169.254.169.254/...`) without DNS work.
 */
export function rejectPrivateLiteralUrl(parsed: URL): void {
  const host = parsed.hostname.replace(/^\[(.*)\]$/, '$1')
  if (!host) throw new Error('missing host')
  if (isIP(host) && isPrivateAddress(host)) {
    throw new Error('refusing private/loopback host')
  }
}
