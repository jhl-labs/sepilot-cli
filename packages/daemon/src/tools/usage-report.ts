import type { UsageTracker, ProviderUsageSummary } from '../memory/usage-tracker.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

interface UsageReportInput {
  provider?: string
  days?: number
}

interface UsageReportRow {
  date?: string
  provider: string
  model: string
  inputTokens: number
  outputTokens: number
  costUsd: number
  requestCount: number
}

function parseInput(input: Record<string, unknown>): UsageReportInput {
  const provider = typeof input.provider === 'string'
    ? input.provider.trim()
    : undefined
  const days = typeof input.days === 'number' && Number.isFinite(input.days)
    ? Math.max(1, Math.min(365, Math.floor(input.days)))
    : undefined
  return {
    provider: provider || undefined,
    days,
  }
}

function matchesProvider(row: UsageReportRow, provider: string | undefined): boolean {
  if (!provider) return true
  const needle = provider.toLowerCase()
  return row.provider.toLowerCase().includes(needle)
    || row.model.toLowerCase().includes(needle)
}

function aggregate(rows: UsageReportRow[]): Omit<ProviderUsageSummary, 'provider' | 'model'> {
  return rows.reduce(
    (total, row) => ({
      inputTokens: total.inputTokens + row.inputTokens,
      outputTokens: total.outputTokens + row.outputTokens,
      costUsd: total.costUsd + row.costUsd,
      requestCount: total.requestCount + row.requestCount,
    }),
    { inputTokens: 0, outputTokens: 0, costUsd: 0, requestCount: 0 },
  )
}

export function createUsageReportTool(usageTracker: UsageTracker): ToolDefinitionRuntime {
  return {
    name: 'usage.report',
    description: [
      'Query recorded LLM provider/model usage from the daemon usage database.',
      'Use for questions about Claude, Anthropic, OpenAI, GPT, Gemini, model token counts, model costs, billing, quotas, or LLM request usage.',
      'Do not use this for host CPU, GPU, memory, disk, or operating-system resource usage; use system.info for host resource questions.',
    ].join(' '),
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'usage-report' },
    inputSchema: {
      type: 'object',
      properties: {
        provider: {
          type: 'string',
          description: 'Optional provider or model filter, for example "anthropic", "claude", "openai", "gpt", or "gemini".',
        },
        days: {
          type: 'number',
          description: 'Optional lookback window in days. Omit for all recorded usage.',
          minimum: 1,
          maximum: 365,
        },
      },
    },
    async execute(input: Record<string, unknown>): Promise<ToolResult> {
      const start = Date.now()
      try {
        const parsed = parseInput(input)
        const rows: UsageReportRow[] = parsed.days
          ? usageTracker.getDailySummaries(parsed.days).map((row) => ({
              date: row.date,
              provider: row.provider,
              model: row.model,
              inputTokens: row.totalInputTokens,
              outputTokens: row.totalOutputTokens,
              costUsd: row.totalCostUsd,
              requestCount: row.requestCount,
            }))
          : usageTracker.getProviderSummaries().map((row) => ({ ...row }))
        const filtered = rows.filter((row) => matchesProvider(row, parsed.provider))
        const total = aggregate(filtered)
        return {
          output: JSON.stringify({
            providerFilter: parsed.provider ?? null,
            days: parsed.days ?? null,
            total,
            rows: filtered,
          }, null, 2),
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (error) {
        return {
          output: error instanceof Error ? error.message : String(error),
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
    },
  }
}
