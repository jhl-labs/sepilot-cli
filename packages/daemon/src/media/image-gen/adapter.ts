export interface ProviderInfo {
  id: string
  label: string
  enabled: boolean
  operations?: MediaOperation[]
  recommendedModels?: RecommendedMediaModel[]
  hardware?: {
    supportsDeviceSelection?: boolean
    devicesEndpoint?: string
  }
}

export type MediaOperation =
  | 'text-to-image'
  | 'image-to-image'
  | 'inpaint'
  | 'text-to-video'
  | 'image-to-video'

export type MediaOutputKind = 'image' | 'video'

export interface RecommendedMediaModel {
  id: string
  label: string
  operation: MediaOperation
  modelId?: string
  notes?: string
  minVramMiB?: number
  recommendedVramMiB?: number
  deviceKinds?: Array<'cuda' | 'mps' | 'cpu'>
  tags?: string[]
  workflow?: {
    id: string
    label: string
    description?: string
  }
  params?: Record<string, unknown>
}

export interface ProviderRunInput {
  jobId: string
  prompt: string
  params?: Record<string, unknown>
  onProgress: (progress: number) => void
  signal?: AbortSignal
}

export interface ProviderRunOutput {
  outputs: {
    id: string
    mime: string
    bytes: Buffer
    path?: string
    kind?: MediaOutputKind
  }[]
}

export interface Provider {
  info: ProviderInfo
  run(input: ProviderRunInput): Promise<ProviderRunOutput>
}
