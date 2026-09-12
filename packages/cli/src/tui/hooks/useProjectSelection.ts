// Three project-selection state slots App.tsx kept inline:
//
// - projects — the daemon's configured project list (refreshed via
//   /project commands and on hydrate)
// - workspaceProject — what detectWorkspaceProject() found for the
//   current cwd, used to suggest auto-selection
// - projectSelectionSource — provenance tag for the *current* project
//   selection: was it detected from the workspace, derived from a
//   loaded session, or explicitly chosen by the operator. Drives the
//   /project info readout.
//
// The hook owns nothing but state; consumers in App.tsx still read via
// the same destructured names.

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
