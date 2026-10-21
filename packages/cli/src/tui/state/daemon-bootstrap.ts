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
