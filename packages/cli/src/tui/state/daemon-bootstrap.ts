// Small async helpers extracted from App.tsx's 225-line connect-on-mount
// useEffect. Each one wraps a single daemon HTTP call with a "best-effort"
// contract: on success returns the parsed value, on failure returns the
// caller-supplied fallback. Pulling them out lets the effect read top-down
// instead of as a wall of try/catch blocks, and makes the per-call failure
// behaviour testable without driving a full bootstrap.
//
// The setters and dispatcher stay in App.tsx; this module only does the
// IO + parsing + fallback-on-failure pieces.

import type {
  DaemonAgentDescriptor,
  DaemonArtifact,
  DaemonConfig,
  DaemonProviderInfo,
  DaemonSessionDetail,
} from '@sepilotd/api-client'
import {
  detectWorkspaceProject,
  type WorkspaceProject,
} from '../utils/projects.js'

export interface DaemonBootstrapHttpClient {
  agents(): Promise<DaemonAgentDescriptor[]>
  providers(): Promise<DaemonProviderInfo[]>
  config(): Promise<DaemonConfig>
}

/**
 * Returns the daemon's registered agent descriptors. On any failure
 * (network, permission, unsupported daemon version) returns an empty
 * array so the caller can render a "no agents" experience instead of
 * blocking the splash.
 */
export async function loadAgentModesOrEmpty(
  httpClient: DaemonBootstrapHttpClient,
): Promise<DaemonAgentDescriptor[]> {
  try {
    return await httpClient.agents()
  } catch {
    return []
  }
}

export interface DaemonProvidersResult {
  providers: DaemonProviderInfo[]
  loaded: boolean
}

/**
 * Fetches the configured provider list. Tracks `loaded` separately so
 * the caller can distinguish "no providers configured" (loaded=true,
 * providers=[]) from "couldn't even reach the providers endpoint"
 * (loaded=false, providers=[]) — the startup preflight summary
 * branches on that.
 */
export async function loadProvidersOrEmpty(
  httpClient: DaemonBootstrapHttpClient,
): Promise<DaemonProvidersResult> {
  try {
    const providers = await httpClient.providers()
    return { providers, loaded: true }
  } catch {
    return { providers: [], loaded: false }
  }
}

export interface DaemonConfigDefaults {
  defaultProvider: string
  defaultModel: string
  agent: DaemonConfig['agent']
}

/**
 * Detect a workspace project rooted at the given directory. Returns
 * null on any failure (no recognised workspace marker, permission
 * denied, etc.) — the caller treats workspace detection as
 * best-effort and falls back to "no workspace" mode.
 */
export async function detectWorkspaceOrNull(
  rootDir: string,
): Promise<WorkspaceProject | null> {
  try {
    return await detectWorkspaceProject(rootDir)
  } catch {
    return null
  }
}

export interface InitialSessionHttpClient {
  session(id: string): Promise<DaemonSessionDetail>
  sessionArtifacts(id: string): Promise<DaemonArtifact[]>
}

export type InitialSessionResult =
  | { ok: true; session: DaemonSessionDetail; artifacts: DaemonArtifact[] }
  | { ok: false; error: unknown }

/**
 * Fetch the session detail + its artifacts in parallel for the
 * initial-session-on-mount path. Artifacts are best-effort (we still
 * boot the session if the artifacts endpoint fails); the session
 * call itself is not — its failure becomes the result error so the
 * caller can render the right message and clean up project bindings
 * for a missing session.
 */
export async function fetchInitialSession(opts: {
  httpClient: InitialSessionHttpClient
  sessionId: string
}): Promise<InitialSessionResult> {
  try {
    const [session, artifacts] = await Promise.all([
      opts.httpClient.session(opts.sessionId),
      opts.httpClient.sessionArtifacts(opts.sessionId).catch(
        (): DaemonArtifact[] => [],
      ),
    ])
    return { ok: true, session, artifacts }
  } catch (error) {
    return { ok: false, error }
  }
}

export interface WsConnectableClient {
  connect(): Promise<void>
  close(): void
}

/**
 * Connect a websocket client created by the caller, bailing out if
 * the bootstrap was cancelled while we were awaiting connect(). On
 * success returns the same instance; on connect failure or post-await
 * cancellation returns null and disposes the failed/extra connection
 * so we don't leak an open socket.
 */
export async function connectWsOrNull<T extends WsConnectableClient>(opts: {
  client: T
  cancelled: () => boolean
}): Promise<T | null> {
  try {
    await opts.client.connect()
  } catch {
    return null
  }
  if (opts.cancelled()) {
    opts.client.close()
    return null
  }
  return opts.client
}

/**
 * Resolve the daemon-wide defaults for provider + model from the
 * config payload. Falls back through agent.defaultProvider ->
 * providers[].default flag -> first provider so the caller always has
 * something concrete to seed the model picker / preflight summary.
 *
 * Returns null when the config endpoint fails — the caller treats
 * config as best-effort and continues bootstrapping without it.
 */
export async function loadDaemonConfigDefaults(
  httpClient: DaemonBootstrapHttpClient,
): Promise<DaemonConfigDefaults | null> {
  try {
    const config = await httpClient.config()
    const defaultProvider = config.agent.defaultProvider
      ?? config.providers.find((provider) => provider.default)?.id
      ?? config.providers[0]?.id
      ?? ''
    const defaultModel = config.agent.defaultModel
      ?? config.providers.find((provider) => provider.id === defaultProvider)
        ?.models?.[0]
      ?? ''
    return { defaultProvider, defaultModel, agent: config.agent }
  } catch {
    return null
  }
}
