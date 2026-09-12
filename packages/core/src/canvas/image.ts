export const IMAGE_CANVAS_OPERATIONS = [
  'text-to-image',
  'image-to-image',
  'inpaint',
  'text-to-video',
  'image-to-video',
] as const

export type ImageCanvasOperation = (typeof IMAGE_CANVAS_OPERATIONS)[number]

export type ImageCanvasOutputKind = 'image' | 'video'

export interface ImageCanvasSize {
  width: number
  height: number
}

export interface ImageCanvasWorkflowRef {
  id: string
  label: string
  description?: string
}

export interface ImageCanvasRecommendedModel {
  id: string
  label: string
  operation: ImageCanvasOperation
  modelId?: string
  notes?: string
  minVramMiB?: number
  recommendedVramMiB?: number
  deviceKinds?: Array<'cuda' | 'mps' | 'cpu'>
  tags?: string[]
  workflow?: ImageCanvasWorkflowRef
  params?: Record<string, unknown>
}

export interface ImageCanvasProviderLike {
  id: string
  label?: string
  enabled?: boolean
  operations?: readonly ImageCanvasOperation[]
  recommendedModels?: readonly ImageCanvasRecommendedModel[]
  hardware?: {
    supportsDeviceSelection?: boolean
    devicesEndpoint?: string
  }
}

export interface ImageCanvasAssetRef {
  id: string
  label?: string
  kind?: ImageCanvasOutputKind | 'mask'
  mime?: string
  fileId?: string
  url?: string
  dataUrl?: string
  width?: number
  height?: number
  prompt?: string
  jobId?: string
  outputIndex?: number
}

export interface ImageCanvasJobDraft {
  providerId: string
  prompt: string
  operation: ImageCanvasOperation
  model?: string
  negativePrompt?: string
  size?: ImageCanvasSize
  steps?: number
  guidanceScale?: number
  seed?: number
  batchSize?: number
  strength?: number
  frames?: number
  fps?: number
  outputFormat?: 'png' | 'webp' | 'jpg' | 'mp4' | 'gif'
  image?: ImageCanvasAssetRef
  mask?: ImageCanvasAssetRef
  params?: Record<string, unknown>
}

export interface ImageCanvasHistoryStep {
  id: string
  operation: ImageCanvasOperation
  prompt: string
  providerId?: string
  model?: string
  jobId?: string
  inputAssetIds?: string[]
  outputAssetIds?: string[]
  createdAt: number
}

export interface ImageCanvasProject {
  kind: 'image-canvas-project'
  version: 1
  id: string
  title?: string
  createdAt: number
  updatedAt: number
  assets: ImageCanvasAssetRef[]
  history: ImageCanvasHistoryStep[]
  activeDraft?: ImageCanvasJobDraft
}

export function isImageCanvasOperation(value: unknown): value is ImageCanvasOperation {
  return (
    typeof value === 'string' &&
    (IMAGE_CANVAS_OPERATIONS as readonly string[]).includes(value)
  )
}

export function imageCanvasOperationNeedsImage(operation: ImageCanvasOperation): boolean {
  return operation === 'image-to-image' || operation === 'inpaint' || operation === 'image-to-video'
}

export function imageCanvasOperationNeedsMask(operation: ImageCanvasOperation): boolean {
  return operation === 'inpaint'
}

export function imageCanvasOperationIsVideo(operation: ImageCanvasOperation): boolean {
  return operation === 'text-to-video' || operation === 'image-to-video'
}

export function providerSupportsImageCanvasOperation(
  provider: Pick<ImageCanvasProviderLike, 'operations'> | null | undefined,
  operation: ImageCanvasOperation,
): boolean {
  if (!provider?.operations?.length) return true
  return provider.operations.includes(operation)
}

export function enabledImageCanvasProvidersFor<T extends ImageCanvasProviderLike>(
  providers: readonly T[],
  operation: ImageCanvasOperation,
): T[] {
  return providers.filter(
    (provider) =>
      provider.enabled !== false && providerSupportsImageCanvasOperation(provider, operation),
  )
}

export function imageCanvasRecommendedModelsFor<
  T extends Pick<ImageCanvasRecommendedModel, 'operation'>,
>(
  provider: { recommendedModels?: readonly T[] } | null | undefined,
  operation: ImageCanvasOperation,
): T[] {
  return (provider?.recommendedModels ?? []).filter((model) => model.operation === operation)
}

export function clampImageCanvasSteps(value: unknown, fallback = 20): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return Math.max(1, Math.min(120, Math.round(parsed)))
}

export function clampImageCanvasStrength(value: unknown, fallback = 0.65): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return Math.max(0.05, Math.min(1, parsed))
}

export function normalizeImageCanvasJobDraft(draft: ImageCanvasJobDraft): ImageCanvasJobDraft {
  return {
    ...draft,
    prompt: draft.prompt.trim(),
    model: draft.model?.trim() || undefined,
    negativePrompt: draft.negativePrompt?.trim() || undefined,
    steps: draft.steps == null ? undefined : clampImageCanvasSteps(draft.steps),
    strength: draft.strength == null ? undefined : clampImageCanvasStrength(draft.strength),
    batchSize:
      draft.batchSize == null
        ? undefined
        : Math.max(1, Math.min(8, Math.round(Number(draft.batchSize) || 1))),
    frames:
      draft.frames == null
        ? undefined
        : Math.max(1, Math.min(240, Math.round(Number(draft.frames) || 16))),
    fps:
      draft.fps == null
        ? undefined
        : Math.max(1, Math.min(60, Math.round(Number(draft.fps) || 8))),
  }
}
