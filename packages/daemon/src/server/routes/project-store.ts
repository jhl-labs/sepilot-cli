import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { assertSafeId } from '../../utils/safe-id.js'

export interface Project {
  id: string
  name: string
  description: string
  instructions: string
  workingDirectory: string
  sessionIds: string[]
  fileIds: string[]
  createdAt: string
  updatedAt: string
}

function normalizeProject(value: unknown): Project {
  const record = (value && typeof value === 'object')
    ? value as Partial<Project>
    : {}

  return {
    id: typeof record.id === 'string' ? record.id : '',
    name: typeof record.name === 'string' ? record.name : '',
    description: typeof record.description === 'string' ? record.description : '',
    instructions: typeof record.instructions === 'string' ? record.instructions : '',
    workingDirectory: typeof record.workingDirectory === 'string'
      ? record.workingDirectory
      : '',
    sessionIds: Array.isArray(record.sessionIds)
      ? record.sessionIds.filter((entry): entry is string => typeof entry === 'string')
      : [],
    fileIds: Array.isArray(record.fileIds)
      ? record.fileIds.filter((entry): entry is string => typeof entry === 'string')
      : [],
    createdAt: typeof record.createdAt === 'string' ? record.createdAt : '',
    updatedAt: typeof record.updatedAt === 'string' ? record.updatedAt : '',
  }
}

export function getProjectsDir(dataDir: string): string {
  return join(dataDir, 'projects')
}

export async function loadProject(
  dir: string,
  id: string,
): Promise<Project | null> {
  try {
    const safeId = assertSafeId(id, 'project id')
    const raw = await readFile(join(dir, `${safeId}.json`), 'utf-8')
    return normalizeProject(JSON.parse(raw))
  } catch {
    return null
  }
}

export async function saveProject(
  dir: string,
  project: Project,
): Promise<void> {
  const safeId = assertSafeId(project.id, 'project id')
  await mkdir(dir, { recursive: true })
  await writeFile(
    join(dir, `${safeId}.json`),
    JSON.stringify(project, null, 2),
  )
}
