export type CtrlCIntent =
  | { kind: 'cancel-stream'; hint: string }
  | { kind: 'clear-input'; hint: string }
  | { kind: 'arm'; hint: string }
  | { kind: 'exit' }

export interface CtrlCContext {
  isStreaming: boolean
  inputLength: number
  exitArmed: boolean
  immediateExitWhenIdle?: boolean
}

export const EXIT_HINT_TIMEOUT_MS = 2000

export function resolveCtrlCIntent(ctx: CtrlCContext): CtrlCIntent {
  if (ctx.isStreaming) {
    return {
      kind: 'cancel-stream',
      hint: 'canceled · press Ctrl+C again to exit',
    }
  }
  if (ctx.inputLength > 0) {
    return {
      kind: 'clear-input',
      hint: 'input cleared · press Ctrl+C again to exit',
    }
  }
  if (ctx.immediateExitWhenIdle) {
    return { kind: 'exit' }
  }
  if (ctx.exitArmed) {
    return { kind: 'exit' }
  }
  return { kind: 'arm', hint: 'press Ctrl+C again to exit' }
}
