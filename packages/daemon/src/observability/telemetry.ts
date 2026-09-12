export interface TelemetryConfig {
  enabled: boolean
  otlpEndpoint?: string
  serviceName?: string
}

export interface Span {
  end(): void
  setStatus(status: 'ok' | 'error', message?: string): void
  setAttribute(key: string, value: string | number | boolean): void
}

class NoopSpan implements Span {
  end() {}
  setStatus() {}
  setAttribute() {}
}

export class TelemetryManager {
  private enabled: boolean
  private counters = new Map<string, number>()
  private histograms = new Map<string, number[]>()

  constructor(config: TelemetryConfig) {
    this.enabled = config.enabled
  }

  async init(): Promise<void> {
    if (!this.enabled) return
    // OTel SDK init would go here when @opentelemetry packages are installed
  }

  startSpan(name: string, _attributes?: Record<string, string | number>): Span {
    if (!this.enabled) return new NoopSpan()
    // In real impl, this would create an OTel span
    const start = Date.now()
    return {
      end: () => { this.recordHistogram(`${name}.duration`, Date.now() - start) },
      setStatus: () => {},
      setAttribute: () => {},
    }
  }

  incrementCounter(name: string, value?: number): void {
    const current = this.counters.get(name) ?? 0
    this.counters.set(name, current + (value ?? 1))
  }

  recordHistogram(name: string, value: number): void {
    const values = this.histograms.get(name) ?? []
    values.push(value)
    this.histograms.set(name, values)
  }

  getCounter(name: string): number {
    return this.counters.get(name) ?? 0
  }

  getHistogram(name: string): { count: number; min: number; max: number; avg: number } {
    const values = this.histograms.get(name) ?? []
    if (values.length === 0) return { count: 0, min: 0, max: 0, avg: 0 }
    return {
      count: values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      avg: values.reduce((a, b) => a + b, 0) / values.length,
    }
  }

  getMetrics(): Record<string, unknown> {
    return {
      counters: Object.fromEntries(this.counters),
      histograms: Object.fromEntries([...this.histograms].map(([k, _v]) => [k, this.getHistogram(k)])),
    }
  }

  async shutdown(): Promise<void> {}
}
