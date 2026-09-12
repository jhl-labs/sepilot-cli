import { randomUUID } from 'node:crypto'
import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { A2ATaskStore, type A2ATaskStoreOptions } from './server.js'
import type { A2AMessage, A2ATask, A2ATaskPushNotificationConfig, A2ATaskState } from './types.js'

interface PersistedA2ATasks {
  version: 1
  tasks: A2ATask[]
  pushNotificationConfigs?: A2ATaskPushNotificationConfig[]
}

const TERMINAL_STATES = new Set<A2ATaskState>([
  'TASK_STATE_COMPLETED',
  'TASK_STATE_FAILED',
  'TASK_STATE_CANCELED',
  'TASK_STATE_REJECTED',
])

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function isTask(value: unknown): value is A2ATask {
  if (!isRecord(value)) return false
  if (typeof value.id !== 'string' || !value.id) return false
  return isRecord(value.status) && typeof value.status.state === 'string'
}

function isTaskPushNotificationConfig(value: unknown): value is A2ATaskPushNotificationConfig {
  if (!isRecord(value)) return false
  if (typeof value.taskId !== 'string' || !value.taskId) return false
  const config = value.pushNotificationConfig
  return isRecord(config)
    && typeof config.id === 'string'
    && typeof config.url === 'string'
}

function nowIso(): string {
  return new Date().toISOString()
}

function restartFailureMessage(task: A2ATask): A2AMessage {
  return {
    messageId: randomUUID(),
    contextId: task.contextId,
    taskId: task.id,
    role: 'ROLE_AGENT',
    parts: [{
      text: 'A2A task did not finish before sepilotd restarted.',
      mediaType: 'text/plain',
    }],
  }
}

function recoverInFlightTask(task: A2ATask): A2ATask {
  if (TERMINAL_STATES.has(task.status.state)) return task
  const message = restartFailureMessage(task)
  return {
    ...task,
    status: {
      state: 'TASK_STATE_FAILED',
      message,
      timestamp: nowIso(),
    },
    history: [...(task.history ?? []), message],
    metadata: {
      ...(task.metadata ?? {}),
      recoveredAfterRestart: true,
    },
  }
}

function quarantineUnreadableTaskFile(filePath: string): void {
  try {
    renameSync(filePath, `${filePath}.invalid.${Date.now()}`)
  } catch {}
}

function loadPersistedPayload(filePath: string): PersistedA2ATasks {
  try {
    const parsed = JSON.parse(readFileSync(filePath, 'utf8')) as unknown
    if (!isRecord(parsed) || !Array.isArray(parsed.tasks)) {
      return { version: 1, tasks: [] }
    }
    return {
      version: 1,
      tasks: parsed.tasks.filter(isTask).map(recoverInFlightTask),
      pushNotificationConfigs: Array.isArray(parsed.pushNotificationConfigs)
        ? parsed.pushNotificationConfigs.filter(isTaskPushNotificationConfig)
        : [],
    }
  } catch (error) {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') {
      return { version: 1, tasks: [] }
    }
    quarantineUnreadableTaskFile(filePath)
    return { version: 1, tasks: [] }
  }
}

export interface A2AFileTaskStoreOptions extends A2ATaskStoreOptions {
  /**
   * When > 0, coalesce persists that happen within the window into a single
   * whole-file write (mitigates O(n^2) rewrite on bursty save+update). Default
   * 0 keeps the durable synchronous write after every mutation.
   */
  persistDebounceMs?: number
}

function resolveMaxTasks(env: NodeJS.ProcessEnv, fallback?: number): number | undefined {
  const raw = Number(env.SEPILOTD_A2A_MAX_TASKS)
  return Number.isFinite(raw) && raw >= 1 ? Math.floor(raw) : fallback
}

export class A2AFileTaskStore extends A2ATaskStore {
  private readonly debounceMs: number
  private persistTimer: ReturnType<typeof setTimeout> | undefined

  constructor(private readonly filePath: string, options: A2AFileTaskStoreOptions = {}) {
    const payload = loadPersistedPayload(filePath)
    super(payload.tasks, {
      ...options,
      maxTasks: resolveMaxTasks(process.env, options.maxTasks),
    })
    this.debounceMs = options.persistDebounceMs ?? 0
    this.restorePushConfigs(payload.pushNotificationConfigs ?? [])
    if (this.snapshotTasks().some((task) => task.metadata?.recoveredAfterRestart === true)) {
      this.persistNow()
    }
  }

  override save(task: A2ATask): A2ATask {
    const saved = super.save(task)
    this.schedulePersist()
    return saved
  }

  override update(id: string, mutator: (task: A2ATask) => A2ATask): A2ATask | null {
    const updated = super.update(id, mutator)
    if (updated) this.schedulePersist()
    return updated
  }

  protected override afterPushConfigMutation(): void {
    this.schedulePersist()
  }

  /** Force any pending debounced write to disk immediately. */
  flush(): void {
    if (this.persistTimer) {
      clearTimeout(this.persistTimer)
      this.persistTimer = undefined
    }
    this.persistNow()
  }

  private schedulePersist(): void {
    if (this.debounceMs <= 0) {
      this.persistNow()
      return
    }
    if (this.persistTimer) return
    this.persistTimer = setTimeout(() => {
      this.persistTimer = undefined
      this.persistNow()
    }, this.debounceMs)
    this.persistTimer.unref?.()
  }

  private persistNow(): void {
    mkdirSync(dirname(this.filePath), { recursive: true })
    const payload: PersistedA2ATasks = {
      version: 1,
      tasks: this.snapshotTasks(),
      pushNotificationConfigs: this.snapshotPushConfigs(),
    }
    const tmpPath = `${this.filePath}.${process.pid}.${Date.now()}.tmp`
    writeFileSync(tmpPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8')
    renameSync(tmpPath, this.filePath)
  }
}
