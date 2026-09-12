import type { ChatResponse, ILLMProvider } from '@sepilotd/core'

export interface ImagePromptPreparerInput {
  providerId: string
  prompt: string
  params: Record<string, unknown>
  signal?: AbortSignal
}

export interface PreparedImagePrompt {
  prompt: string
  params: Record<string, unknown>
}

export type ImagePromptPreparer = (input: ImagePromptPreparerInput) => Promise<PreparedImagePrompt>

interface ImagePromptTranslationRuntime {
  config?: {
    agent?: {
      defaultModel?: string
    }
  }
  providerRegistry?: {
    getDefault(): ILLMProvider | undefined
  }
}

type PromptTranslationMode = 'auto' | 'always' | 'never'

const HANGUL_RE = /[\u3131-\u318E\uAC00-\uD7A3]/
const TRANSLATION_TIMEOUT_MS = 10_000

const PROMPT_TRANSLATION_SYSTEM_PROMPT = `You adapt user prompts for local diffusion image, image editing, inpainting, and video generation.
Return JSON only with this exact shape: {"prompt":"...","negativePrompt":"..."}.
Rules:
- Translate Korean or mixed-language user instructions into concise natural English suitable for diffusion models.
- Preserve quoted text, visible text to render, names, brands, file paths, numbers, aspect ratios, and technical model terms exactly.
- Do not add objects, styles, lighting, camera details, or constraints that are not present in the user's prompt.
- For image-to-image, inpaint, and image/video edit operations, keep the edit intent explicit and preserve unchanged image content.
- Translate the negative prompt into concise comma-separated English only when it is provided. If none is provided, use an empty string.`

function hasHangul(value: string | undefined): boolean {
  return Boolean(value && HANGUL_RE.test(value))
}

function stringParam(params: Record<string, unknown>, key: string): string | undefined {
  const value = params[key]
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function parseTranslationMode(params: Record<string, unknown>): PromptTranslationMode {
  const raw = params.translatePrompt ?? params.promptTranslation
  if (typeof raw === 'boolean') return raw ? 'always' : 'never'
  if (typeof raw !== 'string') return 'auto'

  const value = raw.trim().toLowerCase()
  if (['1', 'always', 'on', 'true', 'yes'].includes(value)) return 'always'
  if (['0', 'false', 'never', 'no', 'none', 'off'].includes(value)) return 'never'
  return 'auto'
}

function shouldTranslate(input: ImagePromptPreparerInput): boolean {
  const mode = parseTranslationMode(input.params)
  if (mode === 'never') return false
  if (mode === 'always') return true
  return hasHangul(input.prompt) || hasHangul(stringParam(input.params, 'negativePrompt'))
}

function firstChatModel(
  provider: ILLMProvider,
  configuredModel: string | undefined,
): string | null {
  if (configuredModel?.trim()) return configuredModel.trim()
  return (
    provider.models.find((model) => !model.capabilities.embedding)?.id ??
    provider.models[0]?.id ??
    null
  )
}

function responseText(response: ChatResponse): string {
  const { content } = response.message
  if (typeof content === 'string') return content
  return content
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

function stripJsonFence(text: string): string {
  const trimmed = text.trim()
  const match = /^```(?:json)?\s*([\s\S]*?)\s*```$/i.exec(trimmed)
  return (match?.[1] ?? trimmed).trim()
}

function parseJsonObject(text: string): Record<string, unknown> | null {
  const cleaned = stripJsonFence(text)
  try {
    const parsed = JSON.parse(cleaned)
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null
  } catch {
    const match = /\{[\s\S]*\}/.exec(cleaned)
    if (!match) return null
    try {
      const parsed = JSON.parse(match[0])
      return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
        ? (parsed as Record<string, unknown>)
        : null
    } catch {
      return null
    }
  }
}

function translatedText(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

export function createImagePromptPreparer(
  runtime: () => ImagePromptTranslationRuntime | undefined,
): ImagePromptPreparer {
  return async (input) => {
    if (!shouldTranslate(input)) {
      return { prompt: input.prompt, params: input.params }
    }

    const currentRuntime = runtime()
    const provider = currentRuntime?.providerRegistry?.getDefault()
    const model = provider
      ? firstChatModel(provider, currentRuntime?.config?.agent?.defaultModel)
      : null
    if (!provider || !model) {
      return { prompt: input.prompt, params: input.params }
    }

    const negativePrompt = stringParam(input.params, 'negativePrompt')
    const payload = {
      providerId: input.providerId,
      operation: stringParam(input.params, 'operation') ?? 'text-to-image',
      diffusionModel:
        stringParam(input.params, 'model') ??
        stringParam(input.params, 'modelId') ??
        stringParam(input.params, 'pipeline') ??
        null,
      pipeline: stringParam(input.params, 'pipeline') ?? null,
      prompt: input.prompt,
      negativePrompt: negativePrompt ?? '',
    }

    try {
      const response = await provider.chat(
        {
          model,
          systemPrompt: PROMPT_TRANSLATION_SYSTEM_PROMPT,
          messages: [
            {
              role: 'user',
              content: JSON.stringify(payload),
            },
          ],
          temperature: 0.1,
          maxTokens: 700,
          timeoutMs: TRANSLATION_TIMEOUT_MS,
        },
        { signal: input.signal },
      )
      const parsed = parseJsonObject(responseText(response))
      const prompt = translatedText(parsed?.prompt)
      if (!prompt) return { prompt: input.prompt, params: input.params }

      const translatedNegativePrompt = translatedText(parsed?.negativePrompt)
      const params =
        negativePrompt && translatedNegativePrompt
          ? { ...input.params, negativePrompt: translatedNegativePrompt }
          : input.params
      return { prompt, params }
    } catch {
      return { prompt: input.prompt, params: input.params }
    }
  }
}
