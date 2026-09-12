import { join } from 'node:path'
import type { ProjectContext } from '../../agent/system-prompt.js'
import { resolveUploadedFileNames } from './file-registry.js'
import { loadProject } from './project-store.js'
import { getRuntimeDataDir } from './utils.js'

export async function loadProjectContext(
  runtime: { dataDir?: string },
  projectId?: string,
): Promise<ProjectContext | undefined> {
  if (!projectId) return undefined

  const project = await loadProject(
    join(getRuntimeDataDir(runtime), 'projects'),
    projectId,
  )
  if (!project) return undefined

  return {
    name: project.name,
    description: project.description || undefined,
    instructions: project.instructions || undefined,
    workingDirectory: project.workingDirectory || undefined,
    fileNames: resolveUploadedFileNames(project.fileIds),
  }
}
