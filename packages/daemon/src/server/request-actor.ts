import type { RequestAuthContext } from './auth.js'

export function resolveApiActor(
  authContext: RequestAuthContext | undefined,
  fallback: string,
): string {
  if (authContext?.kind === 'master') {
    return 'api:master'
  }
  if (authContext?.kind === 'extension') {
    return `api:extension:${authContext.tokenId}`
  }
  return fallback
}
