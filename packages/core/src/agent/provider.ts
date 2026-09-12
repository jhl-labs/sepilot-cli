import type { ChatRequest, ChatResponse, StreamChunk, ModelInfo, Message } from './types.js'

export interface LLMRequestOptions {
  signal?: AbortSignal
}

export type ModelCatalogAuthority = 'configured' | 'endpoint'

export interface ModelCatalogRefreshResult {
  modelIds: string[]
  added: string[]
  removed: string[]
}

export interface ILLMProvider {
  readonly id: string
  readonly name: string
  readonly models: ModelInfo[]
  /**
   * Describes whether `models` is only a configured hint or the latest
   * successful inventory returned by the provider endpoint. Callers may pass
   * through an unknown explicit model only for non-authoritative catalogs.
   */
  readonly modelCatalogAuthority?: ModelCatalogAuthority
  /**
   * Refresh the executable model inventory without persisting provider
   * configuration. A successful refresh must replace the runtime snapshot,
   * including endpoint-advertised capability metadata when available;
   * inventory failures leave the previous snapshot intact and per-model
   * metadata failures must not erase successful sibling refreshes.
   */
  refreshModelCatalog?(): Promise<ModelCatalogRefreshResult>
  chat(request: ChatRequest, options?: LLMRequestOptions): Promise<ChatResponse>
  stream(request: ChatRequest, options?: LLMRequestOptions): AsyncIterable<StreamChunk>
  embed?(texts: string[], model?: string): Promise<number[][]>
  countTokens?(messages: Message[]): Promise<number>
}
