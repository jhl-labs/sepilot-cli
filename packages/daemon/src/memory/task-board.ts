import { appendFile, mkdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { TodoItem } from '@sepilotd/core'
import { createLogger } from '../logger.js'

/**
 * Cross-session project task board (the durable half of the P022 task layer,
 * ~ Claude Code's `~/.claude/tasks/`).
 *
 * TodoWrite is in-session scratch state; this store is the durable spine: it
 * persists project-scoped tasks to `<stateDir>/<projectHash>.tasks.jsonl`
 * (projectHash from `memory/scope.ts#hashProjectPath` — never a raw cwd, which
 * embeds the OS username) so a new session can reload the still-open tasks
 * into its board. Records are append-only; the latest record per task id is
 * the current state, so history is preserved and writes never race-clobber.
 */

const log = createLogger('task-board')

export type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'blocked' | 'cancelled'

export interface ProjectTask {
  id: string
  content: string
  status: TaskStatus
  /** ISO timestamp of this record. */
  updatedAt: string
}

export interface TaskBoardWriteEvent {
  projectHash: string
  path: string
  /** Tasks whose state changed in this write. */
  changed: ProjectTask[]
}

export type TaskBoardBroadcaster = (event: TaskBoardWriteEvent) => void

const OPEN_STATUSES = new Set<TaskStatus>(['pending', 'in_progress', 'blocked'])

/** projectHash is used directly as a filename segment — must be a bare token. */
const SAFE_PROJECT_HASH = /^[a-z0-9_-]+$/

function assertSafeProjectHash(projectHash: string): void {
  if (!SAFE_PROJECT_HASH.test(projectHash)) {
    throw new Error(`Unsafe projectHash for task-board filename: ${JSON.stringify(projectHash)}`)
  }
}

function isTaskStatus(value: unknown): value is TaskStatus {
  return (
    value === 'pending'
    || value === 'in_progress'
    || value === 'completed'
    || value === 'blocked'
    || value === 'cancelled'
  )
}

export class TaskBoardStore {
  constructor(
    private readonly stateDir: string,
    private readonly broadcast?: TaskBoardBroadcaster,
  ) {}

  filePath(projectHash: string): string {
    assertSafeProjectHash(projectHash)
    return join(this.stateDir, `${projectHash}.tasks.jsonl`)
  }

  /** Latest state of every task (append-only log reduced by id). */
  async loadTasks(projectHash: string): Promise<ProjectTask[]> {
    const raw = await this.readRaw(projectHash)
    if (!raw) return []
    const latest = new Map<string, ProjectTask>()
    for (const line of raw.split('\n')) {
      const trimmed = line.trim()
      if (!trimmed) continue
      let parsed: unknown
      try {
        parsed = JSON.parse(trimmed)
      } catch {
        continue
      }
      const task = this.coerceTask(parsed)
      if (task) latest.set(task.id, task)
    }
    return [...latest.values()]
  }

  /** Tasks a new session should resume: pending or in_progress. */
  async loadOpenTasks(projectHash: string): Promise<ProjectTask[]> {
    return (await this.loadTasks(projectHash)).filter((task) => OPEN_STATUSES.has(task.status))
  }

  /**
   * Upsert the given items into the durable store. Only tasks whose content or
   * status actually differs from the current latest record are appended (so a
   * no-op todowrite does not grow the log), and only those are broadcast.
   * Returns the tasks that changed.
   */
  async upsertTasks(
    projectHash: string,
    items: Array<Pick<ProjectTask, 'id' | 'content' | 'status'>>,
    now: () => string = () => new Date().toISOString(),
  ): Promise<ProjectTask[]> {
    const current = new Map((await this.loadTasks(projectHash)).map((task) => [task.id, task]))
    const changed: ProjectTask[] = []
    for (const item of items) {
      if (!item.id || !isTaskStatus(item.status)) continue
      const existing = current.get(item.id)
      if (existing && existing.content === item.content && existing.status === item.status) continue
      changed.push({ id: item.id, content: item.content, status: item.status, updatedAt: now() })
    }
    if (changed.length === 0) return []
    await mkdir(this.stateDir, { recursive: true })
    const payload = changed.map((task) => JSON.stringify(task)).join('\n') + '\n'
    await appendFile(this.filePath(projectHash), payload, 'utf-8')
    this.emit(projectHash, changed)
    return changed
  }

  private coerceTask(value: unknown): ProjectTask | null {
    if (typeof value !== 'object' || value === null) return null
    const { id, content, status, updatedAt } = value as Record<string, unknown>
    if (typeof id !== 'string' || typeof content !== 'string') return null
    if (!isTaskStatus(status)) return null
    return { id, content, status, updatedAt: typeof updatedAt === 'string' ? updatedAt : '' }
  }

  private async readRaw(projectHash: string): Promise<string | undefined> {
    try {
      return await readFile(this.filePath(projectHash), 'utf-8')
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
      throw error
    }
  }

  private emit(projectHash: string, changed: ProjectTask[]): void {
    if (!this.broadcast) return
    try {
      this.broadcast({ projectHash, path: this.filePath(projectHash), changed })
    } catch (error) {
      log.warn('task-board broadcast failed', { projectHash, error: String(error) })
    }
  }
}

/** Convert durable tasks back into in-session TodoItem shape for board load. */
export function projectTasksToTodoItems(tasks: ProjectTask[]): TodoItem[] {
  return tasks
    .filter((task) => task.status !== 'cancelled')
    .map((task) => ({
      id: task.id,
      content: task.content,
      status: task.status,
    }))
}
