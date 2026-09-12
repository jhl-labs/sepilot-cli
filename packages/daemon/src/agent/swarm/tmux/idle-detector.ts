export interface IdleConfig {
  idlePatterns: RegExp[]
  busyPatterns: RegExp[]
  minWaitMs: number
  stableSeconds: number
}

export type IdleDecision = 'waiting-min' | 'busy' | 'not-yet' | 'idle'

function testPattern(pattern: RegExp, text: string): boolean {
  pattern.lastIndex = 0
  return pattern.test(text)
}

function signalLines(text: string): string[] {
  const lines = text
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .split('\n')
    .map((line) => line.trimEnd())
    .filter((line) => line.trim().length > 0)
  return lines.slice(-12)
}

function lastMatchingLine(lines: string[], patterns: RegExp[]): number {
  for (let i = lines.length - 1; i >= 0; i--) {
    if (patterns.some((pattern) => testPattern(pattern, lines[i] ?? ''))) return i
  }
  return -1
}

export class IdleDetector {
  private lastText = ''
  private stableSince: number | null = null

  constructor(private cfg: IdleConfig, private startedAt: number) {}

  observe(text: string, nowMs: number): IdleDecision {
    if (nowMs - this.startedAt < this.cfg.minWaitMs) return 'waiting-min'
    const lines = signalLines(text)
    const signal = lines.join('\n')
    const lastIdle = lastMatchingLine(lines, this.cfg.idlePatterns)
    const lastBusy = lastMatchingLine(lines, this.cfg.busyPatterns)
    const hasIdle = lastIdle >= 0 || this.cfg.idlePatterns.some((p) => testPattern(p, signal))
    const hasBusy = lastBusy >= 0 || this.cfg.busyPatterns.some((p) => testPattern(p, signal))
    const busyIsCurrent = hasBusy && (!hasIdle || lastBusy >= lastIdle)
    if (text !== this.lastText) {
      this.lastText = text
      this.stableSince = nowMs
      return busyIsCurrent ? 'busy' : 'not-yet'
    }
    if (this.stableSince === null) {
      this.stableSince = nowMs
      return busyIsCurrent ? 'busy' : 'not-yet'
    }
    const elapsed = nowMs - this.stableSince
    if (elapsed < this.cfg.stableSeconds * 1000) return 'not-yet'
    if (hasIdle && (!hasBusy || lastIdle >= lastBusy)) return 'idle'
    return busyIsCurrent ? 'busy' : 'not-yet'
  }
}
