import type { IDecisionProvider } from '@sepilotd/core'

export class DecisionProviderRegistry {
  private readonly providers = new Map<string, IDecisionProvider>()
  private defaultId?: string

  register(provider: IDecisionProvider, options?: { default?: boolean }): void {
    this.providers.set(provider.id, provider)
    if (options?.default || !this.defaultId) {
      this.defaultId = provider.id
    }
  }

  get(id: string): IDecisionProvider | undefined {
    return this.providers.get(id)
  }

  getDefault(): IDecisionProvider | undefined {
    return this.defaultId ? this.providers.get(this.defaultId) : undefined
  }

  list(): IDecisionProvider[] {
    return [...this.providers.values()]
  }

  get size(): number {
    return this.providers.size
  }
}
