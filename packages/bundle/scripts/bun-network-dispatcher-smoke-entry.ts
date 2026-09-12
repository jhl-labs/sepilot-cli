import { DEFAULT_NETWORK_CONFIG } from '../../daemon/src/config/schema.ts'
import {
  closeProviderHttpDispatcher,
  configureProviderNetworkBlocked,
} from '../../daemon/src/providers/http-timeout.ts'

const EXPECTED_REASON = 'BUN_COMPILED_DISPATCHER_SMOKE'

function errorChainContainsBlockedReason(error: unknown): boolean {
  const visited = new Set<unknown>()
  let current = error
  while (current && !visited.has(current)) {
    visited.add(current)
    if (
      current instanceof Error
      && (
        current.message.includes(EXPECTED_REASON)
        || (current as Error & { code?: unknown }).code === 'NETWORK_EGRESS_BLOCKED'
      )
    ) {
      return true
    }
    current = typeof current === 'object' && current !== null && 'cause' in current
      ? (current as { cause?: unknown }).cause
      : undefined
  }
  return false
}

function describeErrorChain(error: unknown): string {
  const parts: string[] = []
  const visited = new Set<unknown>()
  let current = error
  while (current && !visited.has(current)) {
    visited.add(current)
    if (current instanceof Error) {
      parts.push(
        `${current.name}:${(current as Error & { code?: unknown }).code ?? ''}:${current.message}`,
      )
    } else {
      parts.push(String(current))
    }
    current = typeof current === 'object' && current !== null && 'cause' in current
      ? (current as { cause?: unknown }).cause
      : undefined
  }
  return parts.join(' <- ')
}

configureProviderNetworkBlocked(DEFAULT_NETWORK_CONFIG, EXPECTED_REASON)

let blockedByConfiguredDispatcher = false
let fetchError: unknown
try {
  // No listener is needed: the configured blocked dispatcher must reject
  // locally before Bun can attempt a native direct connection.
  await globalThis.fetch('http://127.0.0.1:65534/bun-dispatcher-smoke')
} catch (error) {
  fetchError = error
  blockedByConfiguredDispatcher = errorChainContainsBlockedReason(error)
} finally {
  await closeProviderHttpDispatcher()
}

if (!blockedByConfiguredDispatcher) {
  throw new Error(
    'Bun compiled fetch bypassed the npm Undici dispatcher; release network policy is unsafe. '
      + `Observed: ${describeErrorChain(fetchError)}`,
  )
}

process.stdout.write('BUN_NETWORK_DISPATCHER_OK\n')
