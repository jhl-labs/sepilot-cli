import type { ILLMProvider } from '@sepilotd/core'

class RegisteredProvider implements ILLMProvider {
  readonly id: string
  readonly name: string
  readonly models: ILLMProvider['models']
  readonly implementationId: string
  readonly refreshModelCatalog?: ILLMProvider['refreshModelCatalog']
  readonly chat: ILLMProvider['chat']
  readonly stream: ILLMProvider['stream']
  readonly embed?: ILLMProvider['embed']
  readonly countTokens?: ILLMProvider['countTokens']

  constructor(id: string, private readonly provider: ILLMProvider) {
    this.id = id
    this.name = provider.name
    this.models = provider.models
    this.implementationId = provider.id
    this.refreshModelCatalog = provider.refreshModelCatalog?.bind(provider)
    this.chat = provider.chat.bind(provider)
    this.stream = provider.stream.bind(provider)
    this.embed = provider.embed?.bind(provider)
    this.countTokens = provider.countTokens?.bind(provider)
  }

  get modelCatalogAuthority(): ILLMProvider['modelCatalogAuthority'] {
    return this.provider.modelCatalogAuthority
  }
}

export class ProviderRegistry {
  private providers = new Map<string, RegisteredProvider>()
  private defaultProviderId?: string

  register(provider: ILLMProvider, options?: { default?: boolean }): RegisteredProvider
  register(id: string, provider: ILLMProvider, options?: { default?: boolean }): RegisteredProvider
  register(
    idOrProvider: string | ILLMProvider,
    providerOrOptions?: ILLMProvider | { default?: boolean },
    maybeOptions?: { default?: boolean },
  ): RegisteredProvider {
    const id = typeof idOrProvider === 'string' ? idOrProvider : idOrProvider.id
    const provider = typeof idOrProvider === 'string'
      ? providerOrOptions as ILLMProvider
      : idOrProvider
    const options = typeof idOrProvider === 'string'
      ? maybeOptions
      : providerOrOptions as { default?: boolean } | undefined

    const registered = new RegisteredProvider(id, provider)
    this.providers.set(id, registered)

    if (options?.default || !this.defaultProviderId) {
      this.defaultProviderId = id
    }

    return registered
  }

  get(id: string): ILLMProvider | undefined {
    return this.providers.get(id)
  }

  getDefault(): ILLMProvider | undefined {
    if (this.defaultProviderId) {
      return this.providers.get(this.defaultProviderId)
    }
    return this.providers.values().next().value
  }

  list(): ILLMProvider[] {
    return Array.from(this.providers.values())
  }

  has(id: string): boolean {
    return this.providers.has(id)
  }
}
