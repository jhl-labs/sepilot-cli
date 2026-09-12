// Bridge that lets HTTP routes hand a multimodal user turn to the agent
// engine while the engine's run() signature stays `string`.
//
// The chat-stream route assembles a `ContentPart[]` (text + image +
// document parts) when the request includes attachments, but the engine's
// `run(input: string, …)` API and downstream graph plumbing all expect a
// plain string. Without a contract for "this string actually represents
// content parts", `engine.buildInitialMessages` would push the JSON blob
// into a user message as raw text — the LLM then sees base64 image
// bytes as a wall of unreadable characters and, predictably, answers a
// completely unrelated question.
//
// The sentinel below is the contract: routes that need to send multimodal
// input call `encodeMultimodalInput(parts)` to wrap the JSON, and the
// engine calls `decodeMultimodalInput(input)` to recover the array. Plain
// text inputs flow through untouched.

import type { ContentPart } from '@sepilotd/core'

/**
 * Prefix tag used to mark a JSON-stringified `ContentPart[]` as
 * multimodal user input. Chosen to be a literal byte sequence that
 * cannot appear naturally in a Unicode-encoded user prompt: U+0001 is
 * a control character and is not produced by any ordinary keyboard or
 * paste flow, so the false-positive risk on real prompts is zero.
 */
export const MULTIMODAL_INPUT_SENTINEL = 'MULTIMODAL_INPUT'

export function encodeMultimodalInput(parts: ContentPart[]): string {
  return MULTIMODAL_INPUT_SENTINEL + JSON.stringify(parts)
}

/**
 * Whether these content parts require a provider-native content-part request.
 * Extracted source/Markdown/plain-text attachments are ordinary text context
 * and can safely use graph-backed modes. Images and binary documents still
 * need the react bridge until graph providers accept native content parts.
 */
export function requiresNativeMultimodalInput(parts: readonly ContentPart[]): boolean {
  return parts.some((part) => part.type !== 'text')
}

/**
 * Routes always place the user's instruction in the first text part, followed
 * by attachment parts. Keep that instruction as the policy/routing input while
 * the full array remains available to the model as structured context.
 */
export function primaryInstructionFromContentParts(parts: readonly ContentPart[]): string | null {
  const first = parts[0]
  return first?.type === 'text' ? first.text : null
}

/**
 * If `input` carries the multimodal sentinel and decodes to a valid
 * `ContentPart[]`, return that array; otherwise return `null` so the
 * caller falls back to treating `input` as plain text.
 */
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
  // Shape check is intentionally lax: each entry just needs a string
  // `type` field. The provider adapters validate the rest, and we'd
  // rather forward a slightly malformed part than silently drop a user
  // attachment.
  for (const part of parsed) {
    if (!part || typeof part !== 'object') return null
    if (typeof (part as { type?: unknown }).type !== 'string') return null
  }
  return parsed as ContentPart[]
}
