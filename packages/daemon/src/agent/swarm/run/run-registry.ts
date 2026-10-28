import { randomBytes } from 'node:crypto'
import { createLogger } from '../../../logger.js'
import { SwarmRun } from './swarm-run.js'
import type { SwarmRunStore } from './run-store.js'
import type { TmuxSessionPool } from '../tmux/pool.js'
import type { SwarmAgentRuntimeAdapter } from '../launcher/runtime-adapter.js'
import type { SwarmAgentHandle } from '@sepilotd/core'
import type { Worktree } from '../launcher/worktree.js'
const log = createLogger('swarm:journal')

export interface CreateRunOpts {
  goal: string
  worktreePath: string
  worktreeCreatedByDaemon: boolean
}

type SwarmShutdownRuntime =
  | Pick<SwarmAgentRuntimeAdapter, 'stop'>
  | Pick<TmuxSessionPool, 'destroy'>

type SwarmShutdownWorktree = (worktree: Worktree) => void | Promise<void>

function hasStopRuntime(runtime: SwarmShutdownRuntime): runtime is Pick<SwarmAgentRuntimeAdapter, 'stop'> {
  return typeof (runtime as { stop?: unknown }).stop === 'function'
}

async function stopAgent(runtime: SwarmShutdownRuntime, agent: SwarmAgentHandle): Promise<void> {
  if (hasStopRuntime(runtime)) {
    await runtime.stop(agent)
  } else {
    await runtime.destroy(agent.tmuxSessionName)
  }
}

export class SwarmRunRegistry {
  private runs = new Map<string, SwarmRun>()
  private pendingAppends = new Set<Promise<void>>()
  private appendTails = new Map<string, Promise<void>>()

  constructor(
    private store: SwarmRunStore,
    private shutdownRuntime?: SwarmShutdownRuntime,
    private shutdownWorktree?: SwarmShutdownWorktree,
  ) {}

  create(opts: CreateRunOpts): SwarmRun {
    const id = `swarm_${randomBytes(4).toString('hex')}`
    const run = new SwarmRun({
      id,
      goal: opts.goal,
      worktree: { path: opts.worktreePath, createdByDaemon: opts.worktreeCreatedByDaemon },
    })
    run.on('event', (e) => {
      const previous = this.appendTails.get(id) ?? Promise.resolve()
      const p = previous.then(() => this.store.append(id, e)).catch(() => {
        try { log.error('Swarm event could not be persisted; retained history may be incomplete', { runId: id, eventType: e.type }) } catch {}
      })
      this.appendTails.set(id, p)
      this.pendingAppends.add(p)
      void p.finally(() => {
        this.pendingAppends.delete(p)
        if (this.appendTails.get(id) === p) this.appendTails.delete(id)
      })
    })
    this.runs.set(id, run)
    return run
  }

  get(id: string): SwarmRun | undefined {
    return this.runs.get(id)
  }

  list(): SwarmRun[] {
    return [...this.runs.values()]
  }

  deregister(id: string): void {
    this.runs.delete(id)
  }

  async shutdownAll(): Promise<void> {
    const all = [...this.runs.values()]
    this.runs.clear()
    for (const run of all) {
      try {
        run.finish('interrupted')
      } catch {
      }
      if (this.shutdownRuntime) {
        for (const agent of run.snapshot().agents) {
          try {
            await stopAgent(this.shutdownRuntime, agent)
          } catch {
          }
        }
      }
      const worktree = run.snapshot().worktree as Worktree
      if (this.shutdownWorktree && worktree.createdByDaemon) {
        try {
          await this.shutdownWorktree(worktree)
        } catch {
        }
      }
    }
    if (this.pendingAppends.size > 0) {
      await Promise.allSettled([...this.pendingAppends])
    }
  }
}
