import { useState } from 'react'
import type { DaemonProject } from '@sepilotd/api-client'
import type { WorkspaceProject } from '../utils/projects.js'

export type ProjectSelectionSource = 'workspace' | 'session' | 'manual'

export interface UseProjectSelectionResult {
  projects: DaemonProject[]
  setProjects: React.Dispatch<React.SetStateAction<DaemonProject[]>>
  workspaceProject: WorkspaceProject | null
  setWorkspaceProject: React.Dispatch<
    React.SetStateAction<WorkspaceProject | null>
  >
  projectSelectionSource: ProjectSelectionSource | null
  setProjectSelectionSource: React.Dispatch<
    React.SetStateAction<ProjectSelectionSource | null>
  >
}

export function useProjectSelection(): UseProjectSelectionResult {
  const [projects, setProjects] = useState<DaemonProject[]>([])
  const [workspaceProject, setWorkspaceProject] =
    useState<WorkspaceProject | null>(null)
  const [projectSelectionSource, setProjectSelectionSource] =
    useState<ProjectSelectionSource | null>(null)

  return {
    projects,
    setProjects,
    workspaceProject,
    setWorkspaceProject,
    projectSelectionSource,
    setProjectSelectionSource,
  }
}
