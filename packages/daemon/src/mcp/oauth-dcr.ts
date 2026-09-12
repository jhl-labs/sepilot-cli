/**
 * RFC 7591 Dynamic Client Registration helper for MCP OAuth servers.
 *
 * Happy path:
 *   1. GET {authBase}/.well-known/oauth-authorization-server
 *   2. If `registration_endpoint` is present, POST a DCR request there.
 *   3. Return the issued client_id (and optional client_secret).
 *
 * All failures are returned as structured results rather than thrown — the
 * caller decides whether to fall back to a pre-registered credential flow.
 */

export interface DcrInput {
  redirectUris: readonly string[]
  clientName?: string
  scope?: string
  grantTypes?: readonly string[]
  responseTypes?: readonly string[]
  tokenEndpointAuthMethod?: 'none' | 'client_secret_basic' | 'client_secret_post'
}

export interface DcrCredentials {
  clientId: string
  clientSecret?: string
  registrationEndpoint: string
}

export type DcrFailureReason =
  | 'discovery-failed'
  | 'no-registration-endpoint'
  | 'register-failed'
  | 'no-client-id'
  | 'exception'

export type DcrResult =
  | { ok: true; credentials: DcrCredentials }
  | { ok: false; reason: DcrFailureReason; status?: number; detail?: string }

function wellKnownUrl(authBase: string): string {
  return authBase.replace(/\/+$/, '') + '/.well-known/oauth-authorization-server'
}

export async function tryDynamicClientRegistration(
  authBase: string,
  input: DcrInput,
): Promise<DcrResult> {
  try {
    const discoveryRes = await fetch(wellKnownUrl(authBase))
    if (!discoveryRes.ok) {
      return { ok: false, reason: 'discovery-failed', status: discoveryRes.status }
    }
    const discovery = await discoveryRes.json() as { registration_endpoint?: string }
    const endpoint = discovery.registration_endpoint
    if (typeof endpoint !== 'string' || endpoint.length === 0) {
      return { ok: false, reason: 'no-registration-endpoint' }
    }

    const body = {
      redirect_uris: [...input.redirectUris],
      client_name: input.clientName ?? 'sepilotd',
      token_endpoint_auth_method: input.tokenEndpointAuthMethod ?? 'none',
      grant_types: input.grantTypes ?? ['authorization_code', 'refresh_token'],
      response_types: input.responseTypes ?? ['code'],
      ...(input.scope ? { scope: input.scope } : {}),
    }

    const registerRes = await fetch(endpoint, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!registerRes.ok) {
      return { ok: false, reason: 'register-failed', status: registerRes.status }
    }
    const registered = await registerRes.json() as { client_id?: string; client_secret?: string }
    if (typeof registered.client_id !== 'string' || registered.client_id.length === 0) {
      return { ok: false, reason: 'no-client-id' }
    }
    return {
      ok: true,
      credentials: {
        clientId: registered.client_id,
        clientSecret: typeof registered.client_secret === 'string' ? registered.client_secret : undefined,
        registrationEndpoint: endpoint,
      },
    }
  } catch (err) {
    return { ok: false, reason: 'exception', detail: err instanceof Error ? err.message : String(err) }
  }
}
