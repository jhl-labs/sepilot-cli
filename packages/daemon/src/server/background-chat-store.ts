import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import type { RunStopReason } from '@sepilotd/core'
import { dirname, join } from 'node:path'
import type { SessionEvent } from '@sepilotd/core'

export type BackgroundChatStatus = 'running' | 'completed' | 'failed' | 'cancelled'

export interface BackgroundChatJob {
  jobId: string
  sessionId: string
  status: BackgroundChatStatus
  messageId?: string
  content?: string
  progress?: BackgroundChatProgress
  error?: {
    code?: string
    message: string
  }
  /** Structured termination reason of the completed run, when known. */
  stopReason?: RunStopReason
  createdAt: string
  updatedAt: string
}

export interface BackgroundChatProgress {
  eventType: string
  label: string
  detail?: string
  action?: BackgroundChatProgressAction
  eventCount: number
  updatedAt: string
  partialContent?: string
}

export type BackgroundChatProgressAction =
  | { type: 'approval'; requestId: string; toolName: string; preview?: string }
  | { type: 'question'; questionId: string; choices?: string[] }

type PersistedBackgroundChatJob = Omit<BackgroundChatJob, 'content'>

function cloneProgressAction(
  action: BackgroundChatProgressAction | undefined,
): BackgroundChatProgressAction | undefined {
  if (!action) return undefined
  if (action.type === 'approval') {
    return { ...action }
  }
  return {
    ...action,
    choices: action.choices ? [...action.choices] : undefined,
  }
}

function cloneProgress(progress: BackgroundChatProgress | undefined): BackgroundChatProgress | undefined {
  if (!progress) return undefined
  return {
    ...progress,
    action: cloneProgressAction(progress.action),
  }
}

function cloneJob(job: BackgroundChatJob): BackgroundChatJob {
  return {
    ...job,
    progress: cloneProgress(job.progress),
    error: job.error ? { ...job.error } : undefined,
  }
}

function toPersistedJob(job: BackgroundChatJob): PersistedBackgroundChatJob {
  const { content: _content, ...persisted } = job
  return {
    ...persisted,
    error: persisted.error ? { ...persisted.error } : undefined,
  }
}

function parseStopReason(value: unknown): RunStopReason | undefined {
  if (!value || typeof value !== 'object') return undefined
  const candidate = value as Partial<RunStopReason>
  if (typeof candidate.kind !== 'string' || typeof candidate.code !== 'string') return undefined
  return {
    kind: candidate.kind,
    code: candidate.code,
    summary: typeof candidate.summary === 'string' ? candidate.summary : undefined,
    detail: candidate.detail && typeof candidate.detail === 'object' ? { ...candidate.detail } : undefined,
    resumable: candidate.resumable === true,
    nextActions: Array.isArray(candidate.nextActions)
      ? candidate.nextActions.filter((entry): entry is RunStopReason['nextActions'][number] => typeof entry === 'string')
      : [],
  }
}

function parsePersistedJobs(raw: string): BackgroundChatJob[] {
  const parsed = JSON.parse(raw) as unknown
  if (!parsed || typeof parsed !== 'object') return []
  const items = Array.isArray((parsed as { jobs?: unknown }).jobs)
    ? (parsed as { jobs: unknown[] }).jobs
    : []
  const jobs: BackgroundChatJob[] = []
  for (const item of items) {
    if (!item || typeof item !== 'object') continue
    const candidate = item as Partial<PersistedBackgroundChatJob>
    if (
      typeof candidate.jobId !== 'string' ||
      typeof candidate.sessionId !== 'string' ||
      (candidate.status !== 'running' &&
        candidate.status !== 'completed' &&
        candidate.status !== 'failed' &&
        candidate.status !== 'cancelled') ||
      typeof candidate.createdAt !== 'string' ||
      typeof candidate.updatedAt !== 'string'
    ) {
      continue
    }
    jobs.push({
      jobId: candidate.jobId,
      sessionId: candidate.sessionId,
      status: candidate.status,
      messageId: typeof candidate.messageId === 'string' ? candidate.messageId : undefined,
      progress: parseProgress(candidate.progress),
      error:
        candidate.error && typeof candidate.error === 'object'
          ? {
              code: typeof candidate.error.code === 'string' ? candidate.error.code : undefined,
              message:
                typeof candidate.error.message === 'string'
                  ? candidate.error.message
                  : 'Background chat failed.',
            }
          : undefined,
      stopReason: parseStopReason(candidate.stopReason),
      createdAt: candidate.createdAt,
      updatedAt: candidate.updatedAt,
    })
  }
  return jobs
}

function parseProgress(value: unknown): BackgroundChatProgress | undefined {
  if (!value || typeof value !== 'object') return undefined
  const progress = value as Partial<BackgroundChatProgress>
  if (
    typeof progress.eventType !== 'string' ||
    typeof progress.label !== 'string' ||
    typeof progress.eventCount !== 'number' ||
    typeof progress.updatedAt !== 'string'
  ) {
    return undefined
  }
  const action = parseProgressAction(progress.action)
  return {
    eventType: progress.eventType.slice(0, 80),
    label: progress.label.slice(0, 160),
    ...(typeof progress.detail === 'string' ? { detail: progress.detail.slice(0, 240) } : {}),
    ...(action ? { action } : {}),
    ...(typeof progress.partialContent === 'string'
      ? { partialContent: progress.partialContent.slice(0, PARTIAL_CONTENT_LIMIT) }
      : {}),
    eventCount: Math.max(0, Math.floor(progress.eventCount)),
    updatedAt: progress.updatedAt,
  }
}

function parseProgressAction(value: unknown): BackgroundChatProgressAction | undefined {
  if (!value || typeof value !== 'object') return undefined
  const candidate = value as Partial<BackgroundChatProgressAction>
  if (
    candidate.type === 'approval' &&
    typeof candidate.requestId === 'string' &&
    typeof candidate.toolName === 'string'
  ) {
    return {
      type: 'approval',
      requestId: candidate.requestId.slice(0, 120),
      toolName: candidate.toolName.slice(0, 120),
      ...(typeof candidate.preview === 'string' ? { preview: candidate.preview.slice(0, 240) } : {}),
    }
  }
  if (
    candidate.type === 'question' &&
    typeof candidate.questionId === 'string'
  ) {
    return {
      type: 'question',
      questionId: candidate.questionId.slice(0, 120),
      ...(Array.isArray(candidate.choices)
        ? {
            choices: candidate.choices
              .filter((choice): choice is string => typeof choice === 'string')
              .slice(0, 8)
              .map((choice) => choice.slice(0, 120)),
          }
        : {}),
    }
  }
  return undefined
}

export interface BackgroundChatJobStore {
  init(): Promise<void>
  get(jobId: string): Promise<BackgroundChatJob | undefined>
  set(jobId: string, job: BackgroundChatJob): Promise<void>
  touchRunning(jobId: string, updatedAt?: string): Promise<BackgroundChatJob | undefined>
  updateProgress(
    jobId: string,
    progress: BackgroundChatProgress,
  ): Promise<BackgroundChatJob | undefined>
  delete(jobId: string): Promise<void>
  entries(): Promise<BackgroundChatJob[]>
  cleanup(now?: number): Promise<void>
}

export interface BackgroundChatJobStoreOptions {
  dataDir?: string
  terminalRetentionMs?: number
  runningRetentionMs?: number
  now?: () => number
}

export interface BackgroundChatSessionReader {
  getEvents(sessionId: string): Promise<SessionEvent[]>
}

const HOUR_MS = 60 * 60 * 1000
const DEFAULT_TERMINAL_RETENTION_MS = 24 * HOUR_MS
const DEFAULT_RUNNING_RETENTION_MS = 6 * HOUR_MS
const PARTIAL_CONTENT_LIMIT = 32_000

export class JsonBackgroundChatJobStore implements BackgroundChatJobStore {
  private readonly filePath?: string
  private readonly terminalRetentionMs: number
  private readonly runningRetentionMs: number
  private readonly now: () => number
  private jobs = new Map<string, BackgroundChatJob>()
  private initialized = false
  private initPromise?: Promise<void>
  private writeQueue = Promise.resolve()
  private writeSequence = 0

  constructor(options: BackgroundChatJobStoreOptions = {}) {
    this.filePath = options.dataDir
      ? join(options.dataDir, 'sessions', 'background-chat-jobs.json')
      : undefined
    this.terminalRetentionMs = options.terminalRetentionMs ?? DEFAULT_TERMINAL_RETENTION_MS
    this.runningRetentionMs = options.runningRetentionMs ?? DEFAULT_RUNNING_RETENTION_MS
    this.now = options.now ?? (() => Date.now())
  }

  async init(): Promise<void> {
    if (this.initialized) return
    if (!this.initPromise) {
      this.initPromise = this.load().then(
        () => {
          this.initialized = true
        },
        (error) => {
          this.initPromise = undefined
          throw error
        },
      )
    }
    await this.initPromise
  }

  private async load(): Promise<void> {
    if (!this.filePath) return
    try {
      const raw = await readFile(this.filePath, 'utf-8')
      const nowIso = new Date(this.now()).toISOString()
      for (const job of parsePersistedJobs(raw)) {
        this.jobs.set(
          job.jobId,
          job.status === 'running'
            ? {
                ...job,
                status: 'failed',
                error: {
                  code: 'BACKGROUND_CHAT_INTERRUPTED',
                  message: 'Background chat was interrupted by daemon restart before completion.',
                },
                updatedAt: nowIso,
              }
            : job,
        )
      }
      await this.persist()
    } catch (error) {
      if ((error as { code?: unknown }).code !== 'ENOENT') throw error
    }
  }

  async get(jobId: string): Promise<BackgroundChatJob | undefined> {
    await this.init()
    const job = this.jobs.get(jobId)
    return job ? cloneJob(job) : undefined
  }

  async set(jobId: string, job: BackgroundChatJob): Promise<void> {
    await this.init()
    this.jobs.set(jobId, cloneJob(job))
    await this.persist()
  }

  async touchRunning(
    jobId: string,
    updatedAt = new Date(this.now()).toISOString(),
  ): Promise<BackgroundChatJob | undefined> {
    await this.init()
    const job = this.jobs.get(jobId)
    if (!job) return undefined
    if (job.status !== 'running') return cloneJob(job)
    const next = { ...job, updatedAt }
    this.jobs.set(jobId, next)
    await this.persist()
    return cloneJob(next)
  }

  async updateProgress(
    jobId: string,
    progress: BackgroundChatProgress,
  ): Promise<BackgroundChatJob | undefined> {
    await this.init()
    const job = this.jobs.get(jobId)
    if (!job) return undefined
    if (job.status !== 'running') return cloneJob(job)
    const next = {
      ...job,
      progress: cloneProgress(progress),
      updatedAt: progress.updatedAt,
    }
    this.jobs.set(jobId, next)
    await this.persist()
    return cloneJob(next)
  }

  async delete(jobId: string): Promise<void> {
    await this.init()
    this.jobs.delete(jobId)
    await this.persist()
  }

  async entries(): Promise<BackgroundChatJob[]> {
    await this.init()
    return [...this.jobs.values()].map(cloneJob)
  }

  async cleanup(now = this.now()): Promise<void> {
    await this.init()
    let changed = false
    for (const [jobId, job] of this.jobs.entries()) {
      const updatedAt = Date.parse(job.updatedAt)
      const createdAt = Date.parse(job.createdAt)
      if (
        job.status !== 'running' &&
        Number.isFinite(updatedAt) &&
        now - updatedAt > this.terminalRetentionMs
      ) {
        this.jobs.delete(jobId)
        changed = true
        continue
      }
      if (
        job.status === 'running' &&
        Number.isFinite(createdAt) &&
        now - createdAt > this.runningRetentionMs
      ) {
        this.jobs.set(jobId, {
          ...job,
          status: 'failed',
          error: {
            code: 'BACKGROUND_CHAT_STALE',
            message: 'Background chat job exceeded its retention window.',
          },
          updatedAt: new Date(now).toISOString(),
        })
        changed = true
      }
    }
    if (changed) await this.persist()
  }

  private async persist(): Promise<void> {
    if (!this.filePath) return
    const write = this.writeQueue.then(() => this.writeSnapshot())
    this.writeQueue = write.catch(() => undefined)
    await write
  }

  private async writeSnapshot(): Promise<void> {
    if (!this.filePath) return
    await mkdir(dirname(this.filePath), { recursive: true })
    const payload = JSON.stringify(
      {
        version: 1,
        jobs: [...this.jobs.values()].map(toPersistedJob),
      },
      null,
      2,
    )
    this.writeSequence += 1
    const tmpPath = `${this.filePath}.${process.pid}.${Date.now()}.${this.writeSequence}.tmp`
    await writeFile(tmpPath, `${payload}\n`, { mode: 0o600 })
    await rename(tmpPath, this.filePath)
  }
}

export async function hydrateBackgroundChatJobContent(
  job: BackgroundChatJob,
  sessions?: BackgroundChatSessionReader | null,
): Promise<BackgroundChatJob> {
  if (job.status !== 'completed' || job.content || !job.messageId || !sessions?.getEvents) {
    return cloneJob(job)
  }
  try {
    const events = await sessions.getEvents(job.sessionId)
    const assistant = events.find(
      (event) =>
        event.type === 'assistant_message' &&
        event.id === job.messageId &&
        typeof event.content === 'string',
    )
    return assistant && assistant.type === 'assistant_message'
      ? { ...cloneJob(job), content: assistant.content }
      : cloneJob(job)
  } catch {
    return cloneJob(job)
  }
}
