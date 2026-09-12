export type LspDiagnosticSeverity = 'error' | 'warning' | 'info' | 'hint'

export interface LspDiagnostic {
  message: string
  line: number
  severity?: LspDiagnosticSeverity
  source?: string
  code?: string | number
}

export class DiagnosticCollector {
  private readonly byUri = new Map<string, LspDiagnostic[]>()

  update(uri: string, diagnostics: LspDiagnostic[]): void {
    if (diagnostics.length === 0) {
      this.byUri.delete(uri)
      return
    }
    this.byUri.set(uri, [...diagnostics])
  }

  get(uri: string): LspDiagnostic[] {
    const existing = this.byUri.get(uri)
    return existing ? [...existing] : []
  }

  all(): Record<string, LspDiagnostic[]> {
    const out: Record<string, LspDiagnostic[]> = {}
    for (const [uri, list] of this.byUri) {
      out[uri] = [...list]
    }
    return out
  }

  clear(uri?: string): void {
    if (uri === undefined) {
      this.byUri.clear()
      return
    }
    this.byUri.delete(uri)
  }
}
