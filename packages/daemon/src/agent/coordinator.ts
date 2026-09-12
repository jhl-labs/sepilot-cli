import { AgentEngine, type AgentEngineOptions } from './engine.js'
import { createAgentOutputTracker } from './event-output.js'
import type { AgentEvent, AgentContext, TokenUsage } from '@sepilotd/core'
import { randomUUID } from 'node:crypto'

export interface SubTask {
  id: string
  instruction: string
  context?: AgentContext
}

export interface SubTaskResult {
  id: string
  content: string
  usage: TokenUsage
  events: AgentEvent[]
  status: 'completed' | 'error'
  error?: string
}

export class AgentCoordinator {
  private engineOptions: AgentEngineOptions

  constructor(options: AgentEngineOptions) {
    this.engineOptions = options
  }

  /** Run a single sub-task with its own AgentEngine */
  async runSubTask(task: SubTask): Promise<SubTaskResult> {
    const engine = new AgentEngine(this.engineOptions)
    const context: AgentContext = task.context ?? {
      sessionId: randomUUID(),
      provider: this.engineOptions.provider.id,
      model: this.engineOptions.provider.models[0]?.id ?? 'default',
    }

    const events: AgentEvent[] = []
    const outputTracker = createAgentOutputTracker()
    let usage: TokenUsage = { inputTokens: 0, outputTokens: 0 }

    try {
      for await (const event of engine.run(task.instruction, context)) {
        events.push(event)
        outputTracker.consume(event)
        if (event.type === 'done') usage = event.usage
        if (event.type === 'error') {
          return { id: task.id, content: '', usage, events, status: 'error', error: event.error.message }
        }
      }
      return {
        id: task.id,
        content: outputTracker.finalContent(),
        usage,
        events,
        status: 'completed',
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      return { id: task.id, content: '', usage, events, status: 'error', error: message }
    }
  }

  /** Run multiple sub-tasks sequentially */
  async runSequential(tasks: SubTask[]): Promise<SubTaskResult[]> {
    const results: SubTaskResult[] = []
    for (const task of tasks) {
      results.push(await this.runSubTask(task))
    }
    return results
  }

  /** Run multiple sub-tasks in parallel */
  async runParallel(tasks: SubTask[]): Promise<SubTaskResult[]> {
    return Promise.all(tasks.map(task => this.runSubTask(task)))
  }

  /** Plan and execute: use LLM to decompose a complex task, then execute sub-tasks */
  async *planAndExecute(input: string, context: AgentContext): AsyncIterable<AgentEvent> {
    // Phase 1: Planning — use the LLM to decompose
    const planEngine = new AgentEngine({ ...this.engineOptions, maxIterations: 3 })
    const planPrompt = `Break down this task into 2-5 independent sub-tasks. Return a JSON array of objects with "id" and "instruction" fields only. Task: ${input}`

    let planJson = ''
    for await (const event of planEngine.run(planPrompt, context)) {
      if (event.type === 'message') planJson = event.content
    }

    // Try to parse plan
    let subTasks: SubTask[]
    try {
      const match = planJson.match(/\[[\s\S]*\]/)
      subTasks = match ? JSON.parse(match[0]) : [{ id: '1', instruction: input }]
    } catch {
      // Fallback: run as single task
      subTasks = [{ id: '1', instruction: input }]
    }

    yield { type: 'state_change', state: 'thinking' }
    yield { type: 'thinking', content: `Decomposed into ${subTasks.length} sub-tasks` }

    // Phase 2: Execute sub-tasks
    const totalUsage: TokenUsage = { inputTokens: 0, outputTokens: 0 }

    for (const task of subTasks) {
      yield { type: 'thinking', content: `Executing sub-task: ${task.instruction.slice(0, 80)}` }
      const result = await this.runSubTask(task)
      totalUsage.inputTokens += result.usage.inputTokens
      totalUsage.outputTokens += result.usage.outputTokens

      if (result.status === 'error') {
        yield { type: 'thinking', content: `Sub-task ${task.id} failed: ${result.error}` }
      }
    }

    // Phase 3: Synthesize results
    const results = await this.runSequential(subTasks)
    const summary = results.map(r => `[${r.id}] ${r.content}`).join('\n\n')

    yield { type: 'message', content: summary }
    yield { type: 'state_change', state: 'done' }
    yield { type: 'done', usage: totalUsage }
  }
}
