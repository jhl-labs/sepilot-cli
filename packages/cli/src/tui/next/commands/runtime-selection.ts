import type { DaemonAgentDescriptor, DaemonAgentMode, DaemonProviderInfo } from '@sepilotd/api-client'
import { findModeMatch } from '../../components/ModePicker.js'
import { findModelMatch } from '../../utils/provider-models.js'

export interface ModelCommandDeps {
  providers: DaemonProviderInfo[]
  currentProvider: string
  currentModel: string
  defaultProvider: string
  defaultModel: string
  fetchProviders(): Promise<DaemonProviderInfo[]>
  openPicker(target: 'session' | 'default'): void
  select(provider: string, model: string): void
  saveDefault(provider: string, model: string): Promise<void>
  setError(message: string): void
  showNotice(message: string): void
}

async function availableProviders(deps: ModelCommandDeps): Promise<DaemonProviderInfo[] | null> {
  if (deps.providers.length > 0) return deps.providers
  try {
    return await deps.fetchProviders()
  } catch (error) {
    deps.setError(error instanceof Error ? error.message : String(error))
    return null
  }
}

function reportModelMatchError(
  query: string,
  match: ReturnType<typeof findModelMatch>,
  setError: (message: string) => void,
): boolean {
  if (match.ambiguousMatches.length > 0) {
    setError(`Multiple models match "${query}": ${match.ambiguousMatches.slice(0, 8).map((item) => `${item.providerId}/${item.modelId}`).join(', ')}`)
    return true
  }
  if (!match.match) {
    setError(`Unknown model: ${query}`)
    return true
  }
  return false
}

export async function runModelCommand(args: string, deps: ModelCommandDeps): Promise<void> {
  if (!args) {
    deps.openPicker('session')
    return
  }
  if (args === 'current' || args === 'info') {
    deps.showNotice(`Current model: ${deps.currentProvider}/${deps.currentModel}\nDaemon default: ${deps.defaultProvider || '?'}/${deps.defaultModel || '?'}`)
    return
  }
  if (args === 'default') {
    deps.openPicker('default')
    return
  }
  if (args === 'default apply') {
    if (!deps.defaultProvider || !deps.defaultModel) {
      deps.setError('No daemon default model is configured.')
      return
    }
    deps.select(deps.defaultProvider, deps.defaultModel)
    deps.showNotice(`Model set to daemon default ${deps.defaultProvider}/${deps.defaultModel}.`)
    return
  }

  const isDefault = args.startsWith('default ')
  const query = isDefault ? args.slice('default '.length).trim() : args
  const providers = await availableProviders(deps)
  if (!providers) return
  const match = findModelMatch(
    providers,
    isDefault ? deps.defaultProvider || deps.currentProvider : deps.currentProvider,
    isDefault ? deps.defaultModel || deps.currentModel : deps.currentModel,
    query,
  )
  if (reportModelMatchError(query, match, deps.setError) || !match.match) return

  if (isDefault) {
    try {
      await deps.saveDefault(match.match.providerId, match.match.modelId)
      deps.showNotice(`Daemon default model: ${match.match.providerId}/${match.match.modelId}`)
    } catch (error) {
      deps.setError(error instanceof Error ? error.message : String(error))
    }
    return
  }
  deps.select(match.match.providerId, match.match.modelId)
}

export interface ModeCommandDeps {
  agents: DaemonAgentDescriptor[]
  fallbacks: DaemonAgentDescriptor[]
  currentMode: DaemonAgentMode
  fetchAgents(): Promise<DaemonAgentDescriptor[]>
  openPicker(): void
  select(mode: DaemonAgentMode): void
  setError(message: string): void
  showNotice(message: string): void
}

export async function runModeCommand(args: string, deps: ModeCommandDeps): Promise<void> {
  if (!args) {
    deps.openPicker()
    return
  }
  if (args === 'current' || args === 'info') {
    deps.showNotice(`Current mode: ${deps.currentMode}`)
    return
  }
  let agents: DaemonAgentDescriptor[]
  try {
    const fetched = await deps.fetchAgents()
    agents = fetched.length > 0 ? fetched : deps.fallbacks
  } catch {
    agents = deps.agents.length > 0 ? deps.agents : deps.fallbacks
  }
  const match = findModeMatch(agents, deps.currentMode, args)
  if (match.ambiguousMatches.length > 0) {
    deps.setError(`Multiple modes match "${args}": ${match.ambiguousMatches.map((agent) => agent.id).join(', ')}`)
    return
  }
  if (!match.match) {
    deps.setError(`Unknown mode: ${args}`)
    return
  }
  deps.select(match.match.id)
}
