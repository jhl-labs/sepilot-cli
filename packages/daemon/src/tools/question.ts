import { randomUUID } from 'node:crypto'
import type { ToolDefinitionRuntime, ToolResult, ToolExecutionContext } from './registry.js'

export interface PendingQuestion {
  id: string
  sessionId: string
  prompt: string
  choices?: string[]
  resolve: (answer: string) => void
}

export type PendingQuestionListener = (q: PendingQuestion) => void
export type PendingQuestionAnswerListener = (
  q: PendingQuestion,
  answer: string,
) => void

export interface QuestionRequestInput {
  sessionId: string
  prompt: string
  choices?: string[]
}

export type QuestionEnqueueListener = (
  question: Omit<PendingQuestion, 'resolve'>,
) => void | Promise<void>

export class PendingQuestionStore {
  private byId = new Map<string, PendingQuestion>()
  private onEnqueueListener: PendingQuestionListener | null = null
  private onAnswerListener: PendingQuestionAnswerListener | null = null

  enqueue(q: PendingQuestion): void {
    this.byId.set(q.id, q)
    try {
      this.onEnqueueListener?.(q)
    } catch {
      // Listener must not affect question persistence.
    }
  }

  list(sessionId: string): PendingQuestion[] {
    return Array.from(this.byId.values()).filter(q => q.sessionId === sessionId)
  }

  listAll(): PendingQuestion[] {
    return Array.from(this.byId.values())
  }

  cancelForSession(
    sessionId: string,
    answer = 'Session was deleted before this question was answered.',
  ): number {
    let cancelled = 0
    for (const q of Array.from(this.byId.values())) {
      if (q.sessionId !== sessionId) {
        continue
      }
      this.byId.delete(q.id)
      q.resolve(answer)
      try {
        this.onAnswerListener?.(q, answer)
      } catch {
        // Listener must not affect question cancellation.
      }
      cancelled += 1
    }
    return cancelled
  }

  answer(id: string, answer: string): boolean {
    const q = this.byId.get(id)
    if (!q) return false
    this.byId.delete(id)
    q.resolve(answer)
    try {
      this.onAnswerListener?.(q, answer)
    } catch {
      // Listener must not affect question resolution.
    }
    return true
  }

  setOnEnqueue(listener: PendingQuestionListener | null): void {
    this.onEnqueueListener = listener
  }

  setOnAnswer(listener: PendingQuestionAnswerListener | null): void {
    this.onAnswerListener = listener
  }
}

export interface QuestionToolDeps {
  store: PendingQuestionStore
}

export function createQuestionRequester(
  store: Pick<PendingQuestionStore, 'enqueue'>,
  onEnqueue?: QuestionEnqueueListener,
): (input: QuestionRequestInput) => Promise<string> {
  return async ({ sessionId, prompt, choices }: QuestionRequestInput) => {
    const id = randomUUID()
    const answer = new Promise<string>((resolve) => {
      store.enqueue({
        id,
        sessionId,
        prompt,
        choices,
        resolve,
      })
    })
    if (onEnqueue) {
      void Promise.resolve(onEnqueue({ id, sessionId, prompt, choices })).catch(() => {
        // Question visibility must not affect the blocking question itself.
      })
    }
    return answer
  }
}

export function createQuestionTool(deps: QuestionToolDeps): ToolDefinitionRuntime {
  const requestQuestion = createQuestionRequester(deps.store)

  return {
    name: 'question',
    description:
      'Ask the user a free-form question (optionally with choices) and wait for the reply. The run BLOCKS until the user answers — never call this in autonomous mode (the agent will hang) and never to confirm something the user already said. Prefer making a reasonable assumption and stating it; only fall back to question() when you genuinely cannot proceed.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        prompt: { type: 'string' },
        choices: { type: 'array', items: { type: 'string' } },
      },
      required: ['prompt'],
    },
    async execute(input, ctx?: ToolExecutionContext): Promise<ToolResult> {
      const start = Date.now()
      if (!ctx?.sessionId) {
        return { output: 'no session context', status: 'error', durationMs: Date.now() - start }
      }
      const answer = await requestQuestion({
        sessionId: ctx.sessionId,
        prompt: String(input.prompt),
        choices: Array.isArray(input.choices) ? (input.choices as string[]) : undefined,
      })
      return { output: answer, status: 'success', durationMs: Date.now() - start }
    },
  }
}
