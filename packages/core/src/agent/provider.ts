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
  readonly modelCatalogAuthority?: ModelCatalogAuthority
  refreshModelCatalog?(): Promise<ModelCatalogRefreshResult>
  chat(request: ChatRequest, options?: LLMRequestOptions): Promise<ChatResponse>
  stream(request: ChatRequest, options?: LLMRequestOptions): AsyncIterable<StreamChunk>
  embed?(texts: string[], model?: string): Promise<number[][]>
  countTokens?(messages: Message[]): Promise<number>
}
