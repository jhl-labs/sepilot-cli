export interface LlmTurnRequest {
  ruleId: string
  eventId: string
  sessionId: string
  title: string
  message: string
  bot?: string
  projectId?: string
  network?: boolean
  timeoutMs: number
  signal: AbortSignal
}

export type LlmTurnResult =
  | { ok: true; text: string }
  | { ok: false; retryable: boolean; error: string }

export type LlmTurnRunner = (request: LlmTurnRequest) => Promise<LlmTurnResult>
