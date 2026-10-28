export interface CostTokens {
  inputTokens: number
  outputTokens: number
  cacheReadTokens?: number
  cacheCreationTokens?: number
  thinkingTokens?: number
  inputTokensIncludeCache?: boolean
}

export interface CostResult {
  costUsd: number
  costKnown: boolean
}

interface ModelRate {
  input: number
  output: number
  cacheRead?: number
  cacheWrite?: number
}

const PRICING: Record<string, ModelRate> = {
  'claude-opus-4': { input: 0.015, output: 0.075, cacheRead: 0.0015, cacheWrite: 0.01875 },
  'claude-sonnet-5': { input: 0.003, output: 0.015, cacheRead: 0.0003, cacheWrite: 0.00375 },
  'claude-sonnet-4': { input: 0.003, output: 0.015, cacheRead: 0.0003, cacheWrite: 0.00375 },
  'claude-haiku-4': { input: 0.0008, output: 0.004, cacheRead: 0.00008, cacheWrite: 0.001 },
  'claude-3-5-sonnet': { input: 0.003, output: 0.015, cacheRead: 0.0003, cacheWrite: 0.00375 },
  'claude-3-5-haiku': { input: 0.0008, output: 0.004, cacheRead: 0.00008, cacheWrite: 0.001 },
  'claude-3-opus': { input: 0.015, output: 0.075, cacheRead: 0.0015, cacheWrite: 0.01875 },
  'gpt-4o-mini': { input: 0.00015, output: 0.0006 },
  'gpt-4o': { input: 0.0025, output: 0.01 },
  'gpt-4.1-mini': { input: 0.0004, output: 0.0016 },
  'gpt-4.1': { input: 0.002, output: 0.008 },
  'o1-mini': { input: 0.0011, output: 0.0044 },
  'o1': { input: 0.015, output: 0.06 },
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

export function computeCostUsd(_provider: string, model: string, tokens: CostTokens): CostResult {
  const rate = findRate(model)
  if (!rate) {
    return { costUsd: 0, costKnown: false }
  }
  const input = tokens.inputTokens ?? 0
  const output = tokens.outputTokens ?? 0
  const thinking = tokens.thinkingTokens ?? 0
  const cacheRead = tokens.cacheReadTokens ?? 0
  const cacheWrite = tokens.cacheCreationTokens ?? 0
  if ((cacheRead > 0 && rate.cacheRead === undefined) || (cacheWrite > 0 && rate.cacheWrite === undefined)) {
    return { costUsd: 0, costKnown: false }
  }
  const cacheReadRate = rate.cacheRead ?? 0
  const cacheWriteRate = rate.cacheWrite ?? 0
  const freshInput = tokens.inputTokensIncludeCache ? Math.max(0, input - cacheRead - cacheWrite) : input
  const costUsd =
    (freshInput / 1000) * rate.input +
    ((output + thinking) / 1000) * rate.output +
    (cacheRead / 1000) * cacheReadRate +
    (cacheWrite / 1000) * cacheWriteRate
  return { costUsd, costKnown: true }
}

export function isCostKnown(_provider: string, model: string): boolean {
  return findRate(model) !== undefined
}

export function getRatePer1k(_provider: string, model: string): { input: number; output: number } | undefined {
  const rate = findRate(model)
  return rate ? { input: rate.input, output: rate.output } : undefined
}
