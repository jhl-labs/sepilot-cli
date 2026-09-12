import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

const YAHOO_CHART_BASE = 'https://query1.finance.yahoo.com/v8/finance/chart'
const STALE_QUOTE_MS = 4 * 24 * 60 * 60 * 1000

// Hard timeout for the upstream fetch. Without this the Telegram
// channel pipeline (which calls market.quote without an abort
// signal in the context) holds its `activeChannelRuns` slot for up
// to ACTIVE_CHANNEL_RUN_STALE_MS (30 min) when Yahoo doesn't
// respond, blocking every follow-up Telegram message in the same
// chat. 10s matches the 10s timeout used by other outbound
// daemon HTTP calls (outbound-webhook).
const QUOTE_FETCH_TIMEOUT_MS = 10_000

const KNOWN_SYMBOLS = new Map<string, string>([
  ['samsungelectronics', '005930.KS'],
  ['samsung', '005930.KS'],
  ['삼성전자', '005930.KS'],
  ['005930', '005930.KS'],
  ['skhynix', '000660.KS'],
  ['hynix', '000660.KS'],
  ['sk하이닉스', '000660.KS'],
  ['하이닉스', '000660.KS'],
  ['000660', '000660.KS'],
  ['kodex반도체', '091160.KS'],
  ['kodexsemiconductor', '091160.KS'],
  ['091160', '091160.KS'],
  ['kodex미국s&p500', '379800.KS'],
  ['kodex미국sp500', '379800.KS'],
  ['kodexuss&p500', '379800.KS'],
  ['kodexussp500', '379800.KS'],
  ['379800', '379800.KS'],
  ['kodexai반도체', '395160.KS'],
  ['kodexaisemiconductor', '395160.KS'],
  ['395160', '395160.KS'],
])

interface YahooChartResponse {
  chart?: {
    result?: Array<{
      meta?: {
        currency?: string
        symbol?: string
        exchangeName?: string
        instrumentType?: string
        regularMarketPrice?: number
        previousClose?: number
        regularMarketTime?: number
        shortName?: string
        longName?: string
        timezone?: string
      }
      timestamp?: number[]
      indicators?: {
        quote?: Array<{
          close?: Array<number | null>
        }>
      }
    }>
    error?: {
      code?: string
      description?: string
    }
  }
}

function lookupKey(value: string): string {
  return value
    .toLowerCase()
    .replace(/\s+/g, '')
    .replace(/[().,_-]/g, '')
}

function normalizeSymbol(raw: string, market?: string): string {
  const trimmed = raw.trim()
  const key = lookupKey(trimmed)
  const known = KNOWN_SYMBOLS.get(key)
  if (known) return known

  if (/^\d{6}$/.test(trimmed)) {
    return `${trimmed}.${market?.toUpperCase() === 'KQ' ? 'KQ' : 'KS'}`
  }

  if (/^[A-Za-z0-9.-]+$/.test(trimmed)) {
    return trimmed.toUpperCase()
  }

  return trimmed
}

function lastFinite(values: Array<number | null> | undefined): number | undefined {
  if (!values) return undefined
  for (let index = values.length - 1; index >= 0; index -= 1) {
    const value = values[index]
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value
    }
  }
  return undefined
}

function secondsToIso(seconds: number | undefined): string | undefined {
  return typeof seconds === 'number' && Number.isFinite(seconds)
    ? new Date(seconds * 1000).toISOString()
    : undefined
}

export function createMarketQuoteTool(): ToolDefinitionRuntime {
  return {
    name: 'market.quote',
    description: [
      'Fetch a current or latest available market quote for a stock/ETF using Yahoo Finance chart data.',
      'Use this for current prices, today prices, portfolio valuation, or profit/loss calculations.',
      'For Korean listed securities, pass a 6-digit code or a known Korean name; common KRX symbols are normalized to .KS automatically.',
      'Return values include price, currency, source URL, and as-of timestamp. Do not invent prices when this tool fails.',
    ].join(' '),
    resumeSafety: 'replay-safe',
    scheduling: {
      mode: 'parallel-safe',
      resource: 'market-quote',
      key: (input) => typeof input.symbol === 'string' ? input.symbol : null,
    },
    inputSchema: {
      type: 'object',
      properties: {
        symbol: {
          type: 'string',
          description: 'Ticker, 6-digit KRX code, or known security name such as 005930, 000660, 삼성전자, SK하이닉스.',
        },
        market: {
          type: 'string',
          description: 'Optional market hint. Use KR or KS for KOSPI/KRX, KQ for KOSDAQ, or leave empty for symbols with suffixes.',
        },
      },
      required: ['symbol'],
    },
    async execute(input: Record<string, unknown>, context): Promise<ToolResult> {
      const start = Date.now()
      const rawSymbol = typeof input.symbol === 'string' ? input.symbol.trim() : ''
      if (!rawSymbol) {
        return {
          output: 'symbol must be a non-empty string',
          status: 'error',
          durationMs: Date.now() - start,
          code: 'INVALID_SYMBOL_PERMANENT',
        }
      }

      const market = typeof input.market === 'string' ? input.market.trim() : undefined
      const symbol = normalizeSymbol(rawSymbol, market)
      const url = `${YAHOO_CHART_BASE}/${encodeURIComponent(symbol)}?range=1d&interval=1m`

      // Combine the caller's abort signal (if any) with our own
      // timeout so the fetch can never hang past QUOTE_FETCH_TIMEOUT_MS,
      // even when the channel pipeline calls without a context.
      const timeoutSignal = AbortSignal.timeout(QUOTE_FETCH_TIMEOUT_MS)
      const fetchSignal = context?.signal
        ? AbortSignal.any([context.signal, timeoutSignal])
        : timeoutSignal

      try {
        const res = await fetch(url, {
          signal: fetchSignal,
          headers: { 'user-agent': 'sepilotd-market-quote/1' },
        })
        if (!res.ok) {
          const transient = res.status >= 500 || res.status === 429 || res.status === 408
          return {
            output: `Quote fetch failed: HTTP ${res.status}`,
            status: 'error',
            durationMs: Date.now() - start,
            code: transient ? `${res.status}_TRANSIENT` : `${res.status}_PERMANENT`,
          }
        }

        const data = await res.json() as YahooChartResponse
        const chartError = data.chart?.error
        if (chartError) {
          return {
            output: `Quote fetch failed: ${chartError.code ?? 'ERROR'} ${chartError.description ?? ''}`.trim(),
            status: 'error',
            durationMs: Date.now() - start,
            code: 'QUOTE_LOOKUP_PERMANENT',
          }
        }

        const quote = data.chart?.result?.[0]
        const meta = quote?.meta
        const close = lastFinite(quote?.indicators?.quote?.[0]?.close)
        const price = typeof meta?.regularMarketPrice === 'number'
          ? meta.regularMarketPrice
          : close ?? meta?.previousClose
        if (typeof price !== 'number' || !Number.isFinite(price)) {
          return {
            output: `No numeric quote returned for ${symbol}`,
            status: 'error',
            durationMs: Date.now() - start,
            code: 'QUOTE_NOT_FOUND_PERMANENT',
          }
        }

        const timestampSeconds = meta?.regularMarketTime ?? quote?.timestamp?.at(-1)
        const asOf = secondsToIso(timestampSeconds)
        const stale = asOf
          ? Date.now() - new Date(asOf).getTime() > STALE_QUOTE_MS
          : true

        return {
          output: JSON.stringify({
            symbol,
            requestedSymbol: rawSymbol,
            name: meta?.longName ?? meta?.shortName,
            exchange: meta?.exchangeName,
            instrumentType: meta?.instrumentType,
            price,
            previousClose: meta?.previousClose,
            currency: meta?.currency,
            asOf,
            timezone: meta?.timezone,
            stale,
            source: 'Yahoo Finance chart API',
            sourceUrl: url,
            warning: stale
              ? 'Quote timestamp is stale or missing. Do not present it as a current live price without warning.'
              : undefined,
          }, null, 2),
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        // Either signal aborting (caller cancel OR our internal
        // timeout) classifies as TIMEOUT_TRANSIENT so the agent's
        // retry policy treats it as worth one more try at a
        // different upstream, instead of marking the symbol
        // permanently broken.
        const aborted = context?.signal?.aborted || timeoutSignal.aborted
        return {
          output: `Quote fetch error: ${message}`,
          status: 'error',
          durationMs: Date.now() - start,
          code: aborted ? 'TIMEOUT_TRANSIENT' : 'NETWORK_TRANSIENT',
        }
      }
    },
  }
}
