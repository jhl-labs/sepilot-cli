import type { ContentPart } from '@sepilotd/core'

export const MULTIMODAL_INPUT_SENTINEL = 'MULTIMODAL_INPUT'

export function encodeMultimodalInput(parts: ContentPart[]): string {
  return MULTIMODAL_INPUT_SENTINEL + JSON.stringify(parts)
}

export function requiresNativeMultimodalInput(parts: readonly ContentPart[]): boolean {
  return parts.some((part) => part.type !== 'text')
}

export function primaryInstructionFromContentParts(parts: readonly ContentPart[]): string | null {
  const first = parts[0]
  return first?.type === 'text' ? first.text : null
}

export function decodeMultimodalInput(input: string): ContentPart[] | null {
  if (!input.startsWith(MULTIMODAL_INPUT_SENTINEL)) return null
  const json = input.slice(MULTIMODAL_INPUT_SENTINEL.length)
  let parsed: unknown
  try {
    parsed = JSON.parse(json)
  } catch {
    return null
  }
  if (!Array.isArray(parsed) || parsed.length === 0) return null
  for (const part of parsed) {
    if (!part || typeof part !== 'object') return null
    if (typeof (part as { type?: unknown }).type !== 'string') return null
  }
  return parsed as ContentPart[]
}
