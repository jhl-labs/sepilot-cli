// Maintained per-model price table (USD per 1K tokens).
//
// This is the single source of truth for cost accounting. Rates are best-effort
// public list prices and MUST be kept honest: a model that is not in the table
// resolves to `costKnown:false` with `costUsd:0` rather than a fabricated $0
// that pretends to be a known cost. Local models (Ollama etc.) have no API
// billing, so they legitimately fall through to `costKnown:false` — reports
// distinguish "known $0" from "unknown / not billed" via the flag.

export interface CostTokens {
  inputTokens: number
  outputTokens: number
  cacheReadTokens?: number
  cacheCreationTokens?: number
  thinkingTokens?: number
}

export interface CostResult {
  costUsd: number
  costKnown: boolean
}

// Rates are per 1K tokens. cacheRead defaults to 0.1x input, cacheWrite to
// 1.25x input when a provider does not price them separately.
interface ModelRate {
  input: number
  output: number
  cacheRead?: number
  cacheWrite?: number
}

// Keys are provider-agnostic model-id prefixes. Longest matching prefix wins so
// `claude-sonnet-5-20260101` resolves to the `claude-sonnet-5` entry.
const PRICING: Record<string, ModelRate> = {
  // Anthropic Claude (per 1K tokens)
  'claude-opus-4': { input: 0.015, output: 0.075, cacheRead: 0.0015, cacheWrite: 0.01875 },
  'claude-sonnet-5': { input: 0.003, output: 0.015, cacheRead: 0.0003, cacheWrite: 0.00375 },
  'claude-sonnet-4': { input: 0.003, output: 0.015, cacheRead: 0.0003, cacheWrite: 0.00375 },
  'claude-haiku-4': { input: 0.0008, output: 0.004, cacheRead: 0.00008, cacheWrite: 0.001 },
  'claude-3-5-sonnet': { input: 0.003, output: 0.015, cacheRead: 0.0003, cacheWrite: 0.00375 },
  'claude-3-5-haiku': { input: 0.0008, output: 0.004, cacheRead: 0.00008, cacheWrite: 0.001 },
  'claude-3-opus': { input: 0.015, output: 0.075, cacheRead: 0.0015, cacheWrite: 0.01875 },
  // OpenAI
  'gpt-4o-mini': { input: 0.00015, output: 0.0006 },
  'gpt-4o': { input: 0.0025, output: 0.01 },
  'gpt-4.1-mini': { input: 0.0004, output: 0.0016 },
  'gpt-4.1': { input: 0.002, output: 0.008 },
  'o1-mini': { input: 0.0011, output: 0.0044 },
  'o1': { input: 0.015, output: 0.06 },
  // Google Gemini
  'gemini-1.5-flash': { input: 0.000075, output: 0.0003 },
  'gemini-1.5-pro': { input: 0.00125, output: 0.005 },
  'gemini-2.0-flash': { input: 0.0001, output: 0.0004 },
}

function normalizeModelId(model: string): string {
  return model.trim().toLowerCase()
}

function findRate(model: string): ModelRate | undefined {
  const id = normalizeModelId(model)
  let best: { key: string; rate: ModelRate } | undefined
  for (const [key, rate] of Object.entries(PRICING)) {
    if (id === key || id.startsWith(`${key}-`) || id.startsWith(`${key}:`) || id.startsWith(key)) {
      if (!best || key.length > best.key.length) {
        best = { key, rate }
      }
    }
  }
  return best?.rate
}

/**
 * Compute the USD cost of a token breakdown for a provider/model pair.
 *
 * Unknown models return `{ costUsd: 0, costKnown: false }` — never a fabricated
 * $0 that reads as a known cost. Thinking tokens are billed at the output rate.
 */
export function computeCostUsd(_provider: string, model: string, tokens: CostTokens): CostResult {
  const rate = findRate(model)
  if (!rate) {
    return { costUsd: 0, costKnown: false }
  }
  const cacheReadRate = rate.cacheRead ?? rate.input * 0.1
  const cacheWriteRate = rate.cacheWrite ?? rate.input * 1.25
  const input = tokens.inputTokens ?? 0
  const output = tokens.outputTokens ?? 0
  const thinking = tokens.thinkingTokens ?? 0
  const cacheRead = tokens.cacheReadTokens ?? 0
  const cacheWrite = tokens.cacheCreationTokens ?? 0
  const costUsd =
    (input / 1000) * rate.input +
    ((output + thinking) / 1000) * rate.output +
    (cacheRead / 1000) * cacheReadRate +
    (cacheWrite / 1000) * cacheWriteRate
  return { costUsd, costKnown: true }
}

/** Whether a provider/model pair has a maintained price. */
export function isCostKnown(_provider: string, model: string): boolean {
  return findRate(model) !== undefined
}

/**
 * Per-1K input/output rate for a model, or undefined when unpriced. Used by the
 * estimate route so it shares this one price table instead of a private copy.
 */
export function getRatePer1k(_provider: string, model: string): { input: number; output: number } | undefined {
  const rate = findRate(model)
  return rate ? { input: rate.input, output: rate.output } : undefined
}
