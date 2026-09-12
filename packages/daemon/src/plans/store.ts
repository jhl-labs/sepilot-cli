import { mkdir, readFile } from 'node:fs/promises'
import { randomUUID } from 'node:crypto'
import { join } from 'node:path'
import { writeFileAtomic } from '../utils/atomic-write.js'

export type WorkPlanStatus = 'draft' | 'ready' | 'running' | 'completed' | 'archived'
export type WorkPlanStepStatus = 'pending' | 'in_progress' | 'done' | 'blocked' | 'skipped'

export interface WorkPlanStep {
  id: string
  title: string
  status: WorkPlanStepStatus
  detail?: string
  children?: WorkPlanStep[]
}

export interface WorkPlan {
  id: string
  title: string
  goal: string
  status: WorkPlanStatus
  steps: WorkPlanStep[]
  acceptanceCriteria: string[]
  risks: string[]
  decisions: string[]
  sourceSessionId?: string
  lastJobId?: string
  lastStartedAt?: string
  createdAt: string
  updatedAt: string
}

export interface WorkPlanCreateInput {
  title?: string
  goal: string
  steps?: WorkPlanStep[]
  acceptanceCriteria?: string[]
  risks?: string[]
  decisions?: string[]
  sourceSessionId?: string
  status?: WorkPlanStatus
}

export interface WorkPlanPatchInput {
  title?: string
  goal?: string
  status?: WorkPlanStatus
  steps?: WorkPlanStep[]
  acceptanceCriteria?: string[]
  risks?: string[]
  decisions?: string[]
}

export interface WorkPlanStore {
  init(): Promise<void>
  create(input: WorkPlanCreateInput): Promise<WorkPlan>
  list(): Promise<WorkPlan[]>
  get(id: string): Promise<WorkPlan | null>
  patch(id: string, input: WorkPlanPatchInput): Promise<WorkPlan | null>
  markStarted(id: string, jobId: string): Promise<WorkPlan | null>
}

interface StoredPlansFile {
  version: 1
  plans: WorkPlan[]
}

const mutationQueues = new Map<string, Promise<void>>()

function createEmptyPlansFile(): StoredPlansFile {
  return { version: 1, plans: [] }
}

function isErrnoCode(error: unknown, code: string): boolean {
  return (error as NodeJS.ErrnoException | undefined)?.code === code
}

async function serializeMutation<T>(
  key: string,
  mutation: () => Promise<T>,
): Promise<T> {
  const previous = mutationQueues.get(key) ?? Promise.resolve()
  let release!: () => void
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  const tail = previous.catch(() => undefined).then(() => gate)
  mutationQueues.set(key, tail)
  await previous.catch(() => undefined)
  try {
    return await mutation()
  } finally {
    release()
    if (mutationQueues.get(key) === tail) {
      mutationQueues.delete(key)
    }
  }
}

function nowIso(): string {
  return new Date().toISOString()
}

function titleFromGoal(goal: string): string {
  return goal.trim().replace(/\s+/g, ' ').slice(0, 80) || 'Untitled plan'
}

function normalizeStrings(values: string[] | undefined): string[] {
  return (values ?? []).map((value) => value.trim()).filter(Boolean)
}

function normalizeStep(step: WorkPlanStep, fallbackId: string): WorkPlanStep {
  return {
    id: step.id?.trim() || fallbackId,
    title: step.title.trim(),
    status: step.status ?? 'pending',
    detail: step.detail?.trim() || undefined,
    children: step.children?.map((child, index) =>
      normalizeStep(child, `${fallbackId}.${index + 1}`),
    ),
  }
}

function normalizeSteps(steps: WorkPlanStep[] | undefined): WorkPlanStep[] {
  return (steps ?? [])
    .filter((step) => step.title?.trim())
    .map((step, index) => normalizeStep(step, String(index + 1)))
}

export class JsonWorkPlanStore implements WorkPlanStore {
  private readonly filePath: string

  constructor(private readonly dir: string) {
    this.filePath = join(dir, 'plans.json')
  }

  async init(): Promise<void> {
    await mkdir(this.dir, { recursive: true })
    await serializeMutation(this.filePath, async () => {
      try {
        await readFile(this.filePath, 'utf-8')
      } catch (error) {
        if (!isErrnoCode(error, 'ENOENT')) throw error
        await this.write(createEmptyPlansFile())
      }
    })
  }

  async create(input: WorkPlanCreateInput): Promise<WorkPlan> {
    return serializeMutation(this.filePath, async () => {
      const file = await this.read()
      const timestamp = nowIso()
      const goal = input.goal.trim()
      const plan: WorkPlan = {
        id: randomUUID(),
        title: input.title?.trim() || titleFromGoal(goal),
        goal,
        status: input.status ?? 'draft',
        steps: normalizeSteps(input.steps),
        acceptanceCriteria: normalizeStrings(input.acceptanceCriteria),
        risks: normalizeStrings(input.risks),
        decisions: normalizeStrings(input.decisions),
        sourceSessionId: input.sourceSessionId?.trim() || undefined,
        createdAt: timestamp,
        updatedAt: timestamp,
      }
      file.plans.unshift(plan)
      await this.write(file)
      return plan
    })
  }

  async list(): Promise<WorkPlan[]> {
    const file = await this.read()
    return [...file.plans].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
  }

  async get(id: string): Promise<WorkPlan | null> {
    const file = await this.read()
    return file.plans.find((plan) => plan.id === id) ?? null
  }

  async patch(id: string, input: WorkPlanPatchInput): Promise<WorkPlan | null> {
    return serializeMutation(this.filePath, async () => {
      const file = await this.read()
      const index = file.plans.findIndex((plan) => plan.id === id)
      if (index < 0) return null
      const current = file.plans[index]!
      const next: WorkPlan = {
        ...current,
        title: input.title?.trim() || current.title,
        goal: input.goal?.trim() || current.goal,
        status: input.status ?? current.status,
        steps: input.steps ? normalizeSteps(input.steps) : current.steps,
        acceptanceCriteria: input.acceptanceCriteria
          ? normalizeStrings(input.acceptanceCriteria)
          : current.acceptanceCriteria,
        risks: input.risks ? normalizeStrings(input.risks) : current.risks,
        decisions: input.decisions ? normalizeStrings(input.decisions) : current.decisions,
        updatedAt: nowIso(),
      }
      file.plans[index] = next
      await this.write(file)
      return next
    })
  }

  async markStarted(id: string, jobId: string): Promise<WorkPlan | null> {
    return serializeMutation(this.filePath, async () => {
      const file = await this.read()
      const index = file.plans.findIndex((plan) => plan.id === id)
      if (index < 0) return null
      const timestamp = nowIso()
      const next: WorkPlan = {
        ...file.plans[index]!,
        status: 'running',
        lastJobId: jobId,
        lastStartedAt: timestamp,
        updatedAt: timestamp,
      }
      file.plans[index] = next
      await this.write(file)
      return next
    })
  }

  private async read(): Promise<StoredPlansFile> {
    try {
      const raw = await readFile(this.filePath, 'utf-8')
      const parsed = JSON.parse(raw) as StoredPlansFile
      if (parsed.version !== 1 || !Array.isArray(parsed.plans)) {
        throw new Error(`Unsupported work plan store format: ${this.filePath}`)
      }
      return parsed
    } catch (error) {
      if (isErrnoCode(error, 'ENOENT')) return createEmptyPlansFile()
      throw error
    }
  }

  private async write(file: StoredPlansFile): Promise<void> {
    await mkdir(this.dir, { recursive: true })
    await writeFileAtomic(this.filePath, `${JSON.stringify(file, null, 2)}\n`)
  }
}
