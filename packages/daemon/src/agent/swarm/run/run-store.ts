import { appendFile, readFile, readdir, mkdir } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import type { SwarmEvent, SwarmRun, SwarmAgentHandle } from '@sepilotd/core'

export class SwarmRunStore {
  constructor(private sessionsDir: string) {}

  private path(runId: string): string {
    return join(this.sessionsDir, `${runId}.jsonl`)
  }

  async append(runId: string, event: SwarmEvent): Promise<void> {
    if (!existsSync(this.sessionsDir)) {
      await mkdir(this.sessionsDir, { recursive: true })
    }
    await appendFile(this.path(runId), JSON.stringify(event) + '\n', 'utf-8')
  }

  async readAll(runId: string): Promise<SwarmEvent[]> {
    if (!existsSync(this.path(runId))) return []
    const raw = await readFile(this.path(runId), 'utf-8')
    return raw
      .split('\n')
      .filter(Boolean)
      .map((line) => JSON.parse(line) as SwarmEvent)
  }

  async readSince(runId: string, sinceMs: number): Promise<SwarmEvent[]> {
    const all = await this.readAll(runId)
    return all.filter((e) => e.ts > sinceMs)
  }

  /**
   * Reconstruct a SwarmRun snapshot from jsonl events. Used to surface
   * already-completed runs (which have been deregistered from the active
   * map) via the same shape the active /runs endpoints return.
   */
  async reconstruct(runId: string): Promise<SwarmRun | null> {
    const events = await this.readAll(runId)
    if (events.length === 0) return null
    const started = events.find((e) => e.type === 'run.started')
    if (!started || started.type !== 'run.started') return null
    const ended = [...events].reverse().find((e) => e.type === 'run.ended')
    const agents = new Map<string, SwarmAgentHandle>()
    let activeHandle: string | undefined
    for (const e of events) {
      if (e.type === 'agent.spawned') agents.set(e.agent.handle, { ...e.agent })
      else if (e.type === 'agent.killed') agents.delete(e.handle)
      else if (e.type === 'agent.startup') {
        const a = agents.get(e.handle)
        if (a) {
          a.startupEvidence = { ...e.evidence }
          a.runtime = e.evidence.runtime
        }
      }
      else if (e.type === 'agent.status') {
        const a = agents.get(e.handle)
        if (a) a.status = e.status
      } else if (e.type === 'agent.active') activeHandle = e.handle
    }
    return {
      id: started.runId,
      goal: started.goal,
      status: ended && ended.type === 'run.ended' ? ended.status : 'running',
      worktree: { path: '', createdByDaemon: false },
      activeHandle,
      agents: [...agents.values()],
      createdAt: started.ts,
      endedAt: ended?.ts,
    }
  }

  /** List runs in the sessions dir, newest-first. Does NOT load events. */
  async listIds(): Promise<string[]> {
    if (!existsSync(this.sessionsDir)) return []
    const entries = await readdir(this.sessionsDir, { withFileTypes: true })
    return entries
      .filter((d) => d.isFile() && d.name.startsWith('swarm_') && d.name.endsWith('.jsonl'))
      .map((d) => d.name.slice(0, -'.jsonl'.length))
  }

  /**
   * Reconstruct snapshots for the most-recently-modified `limit` runs.
   * Used to power /swarm history without loading the entire on-disk corpus.
   */
  async listHistory(limit = 50): Promise<SwarmRun[]> {
    const ids = await this.listIds()
    if (ids.length === 0) return []
    const withMtime = await Promise.all(
      ids.map(async (id) => {
        const path = this.path(id)
        const { stat } = await import('node:fs/promises')
        const s = await stat(path).catch(() => null)
        return { id, mtime: s?.mtimeMs ?? 0 }
      }),
    )
    withMtime.sort((a, b) => b.mtime - a.mtime)
    const top = withMtime.slice(0, limit)
    const runs = await Promise.all(top.map((e) => this.reconstruct(e.id)))
    return runs.filter((r): r is SwarmRun => r !== null)
  }
}
