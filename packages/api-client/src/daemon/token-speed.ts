import type { DaemonUsage } from './types.js'

export interface TokenSpeedUsage {
  inputTokens: number
  outputTokens: number
}

export interface TokenSpeedRunRecordInput {
  startedAt: number
  finishedAt: number
  usage: TokenSpeedUsage | Pick<DaemonUsage, 'inputTokens' | 'outputTokens'>
}

export interface TokenSpeedSample {
  startedAt: number
  finishedAt: number
  durationMs: number
  inputTokens: number
  outputTokens: number
  totalTokens: number
  inputTokensPerSecond: number
  outputTokensPerSecond: number
  totalTokensPerSecond: number
}

export interface TokenSpeedSessionStats {
  sampleCount: number
  durationMs: number
  inputTokens: number
  outputTokens: number
  totalTokens: number
  inputTokensPerSecond: number
  outputTokensPerSecond: number
  totalTokensPerSecond: number
}

export interface TokenSpeedSnapshot {
  last: TokenSpeedSample | null
  session: TokenSpeedSessionStats
}

export interface TokenSpeedTracker {
  recordRun(input: TokenSpeedRunRecordInput): TokenSpeedSample
  snapshot(): TokenSpeedSnapshot
  reset(): void
}

export interface FormatTokenSpeedStatsOptions {
  locale?: 'en' | 'ko'
}

const MIN_DURATION_MS = 1

function emptySessionStats(): TokenSpeedSessionStats {
  return {
    sampleCount: 0,
    durationMs: 0,
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    inputTokensPerSecond: 0,
    outputTokensPerSecond: 0,
    totalTokensPerSecond: 0,
  }
}

function tokensPerSecond(tokens: number, durationMs: number): number {
  return tokens / (Math.max(durationMs, MIN_DURATION_MS) / 1000)
}

function buildSample(input: TokenSpeedRunRecordInput): TokenSpeedSample {
  const startedAt = Number.isFinite(input.startedAt) ? input.startedAt : Date.now()
  const finishedAt = Number.isFinite(input.finishedAt) ? input.finishedAt : Date.now()
  const durationMs = Math.max(MIN_DURATION_MS, finishedAt - startedAt)
  const inputTokens = Math.max(0, input.usage.inputTokens)
  const outputTokens = Math.max(0, input.usage.outputTokens)
  const totalTokens = inputTokens + outputTokens

  return {
    startedAt,
    finishedAt,
    durationMs,
    inputTokens,
    outputTokens,
    totalTokens,
    inputTokensPerSecond: tokensPerSecond(inputTokens, durationMs),
    outputTokensPerSecond: tokensPerSecond(outputTokens, durationMs),
    totalTokensPerSecond: tokensPerSecond(totalTokens, durationMs),
  }
}

function recalculateRates(stats: TokenSpeedSessionStats): TokenSpeedSessionStats {
  return {
    ...stats,
    inputTokensPerSecond: tokensPerSecond(stats.inputTokens, stats.durationMs || MIN_DURATION_MS),
    outputTokensPerSecond: tokensPerSecond(stats.outputTokens, stats.durationMs || MIN_DURATION_MS),
    totalTokensPerSecond: tokensPerSecond(stats.totalTokens, stats.durationMs || MIN_DURATION_MS),
  }
}

export function createTokenSpeedTracker(): TokenSpeedTracker {
  let last: TokenSpeedSample | null = null
  let session = emptySessionStats()

  return {
    recordRun(input) {
      const sample = buildSample(input)
      last = sample
      session = recalculateRates({
        sampleCount: session.sampleCount + 1,
        durationMs: session.durationMs + sample.durationMs,
        inputTokens: session.inputTokens + sample.inputTokens,
        outputTokens: session.outputTokens + sample.outputTokens,
        totalTokens: session.totalTokens + sample.totalTokens,
        inputTokensPerSecond: 0,
        outputTokensPerSecond: 0,
        totalTokensPerSecond: 0,
      })
      return sample
    },
    snapshot() {
      return {
        last,
        session: { ...session },
      }
    },
    reset() {
      last = null
      session = emptySessionStats()
    },
  }
}

function formatCount(value: number, locale: 'en' | 'ko'): string {
  return value.toLocaleString(locale === 'ko' ? 'ko-KR' : 'en-US')
}

function formatRate(value: number, locale: 'en' | 'ko'): string {
  return value.toLocaleString(locale === 'ko' ? 'ko-KR' : 'en-US', {
    maximumFractionDigits: 1,
    minimumFractionDigits: 1,
  })
}

function formatSeconds(durationMs: number, locale: 'en' | 'ko'): string {
  const seconds = durationMs / 1000
  return seconds.toLocaleString(locale === 'ko' ? 'ko-KR' : 'en-US', {
    maximumFractionDigits: seconds < 10 ? 2 : 1,
    minimumFractionDigits: seconds < 10 ? 2 : 1,
  })
}

function formatSample(sample: TokenSpeedSample, locale: 'en' | 'ko'): string[] {
  const duration = formatSeconds(sample.durationMs, locale)
  const output = `${formatCount(sample.outputTokens, locale)} output tokens / ${duration}s = ${formatRate(sample.outputTokensPerSecond, locale)} tokens/sec`
  const input = `${formatCount(sample.inputTokens, locale)} input tokens`
  const total = `${formatCount(sample.totalTokens, locale)} total billed tokens`

  if (locale === 'ko') {
    return [
      `최근 응답 출력 속도: ${formatCount(sample.outputTokens, locale)} 출력 토큰 / ${duration}초 = ${formatRate(sample.outputTokensPerSecond, locale)} tokens/sec`,
      `  입력 토큰: ${formatCount(sample.inputTokens, locale)}개 (처리량 계산에 넣지 않음)`,
      `  총 과금 토큰: ${formatCount(sample.totalTokens, locale)}개`,
    ]
  }

  return [
    `Last response output speed: ${output}`,
    `  Input: ${input} (not counted as throughput)`,
    `  Total: ${total}`,
  ]
}

function formatSession(stats: TokenSpeedSessionStats, locale: 'en' | 'ko'): string[] {
  const duration = formatSeconds(stats.durationMs, locale)
  const runLabel = locale === 'ko'
    ? `${formatCount(stats.sampleCount, locale)}회`
    : `${formatCount(stats.sampleCount, locale)} run${stats.sampleCount === 1 ? '' : 's'}`

  if (locale === 'ko') {
    return [
      `현재 세션 출력 평균: ${runLabel}, ${formatCount(stats.outputTokens, locale)} 출력 토큰 / ${duration}초 = ${formatRate(stats.outputTokensPerSecond, locale)} tokens/sec`,
      `  입력 토큰 합계: ${formatCount(stats.inputTokens, locale)}개`,
      `  총 과금 토큰: ${formatCount(stats.totalTokens, locale)}개`,
    ]
  }

  return [
    `Current session output average: ${runLabel}, ${formatCount(stats.outputTokens, locale)} output tokens / ${duration}s = ${formatRate(stats.outputTokensPerSecond, locale)} tokens/sec`,
    `  Input total: ${formatCount(stats.inputTokens, locale)} tokens`,
    `  Total billed: ${formatCount(stats.totalTokens, locale)} tokens`,
  ]
}

export function formatTokenSpeedStats(
  snapshot: TokenSpeedSnapshot,
  options: FormatTokenSpeedStatsOptions = {},
): string {
  const locale = options.locale ?? 'en'
  const lines = ['TPS']

  if (!snapshot.last || snapshot.session.sampleCount === 0) {
    lines.push(
      locale === 'ko'
        ? '아직 완료된 LLM 응답이 없습니다. 대화를 한 번 완료한 뒤 /tps를 실행하세요.'
        : 'No completed LLM response has been observed yet. Run /tps after a response completes.',
    )
  } else {
    lines.push(...formatSample(snapshot.last, locale))
    lines.push(...formatSession(snapshot.session, locale))
  }

  lines.push(
    locale === 'ko'
      ? '범위: 이 화면에서 관측한 완료 응답 기준입니다. 입력 토큰은 컨텍스트 크기이며 TPS 처리량으로 계산하지 않습니다. /tps reset으로 초기화할 수 있습니다.'
      : 'Scope: completed responses observed in this client surface. Input tokens are context size, not throughput. Use /tps reset to clear it.',
  )
  return lines.join('\n')
}
