import type { DaemonProject } from '@sepilotd/api-client'
import { findProjectBySession } from '../utils/projects.js'
import type { ProjectSelectionSource } from '../hooks/useProjectSelection.js'

export type AutoProjectFromSessionDecision =
  | { kind: 'noop' }
  | { kind: 'apply'; project: DaemonProject }

export function decideAutoProjectFromSession(opts: {
  sessionId: string | null
  projects: ReadonlyArray<DaemonProject>
  selectedProjectId: string | null
  projectSelectionSource: ProjectSelectionSource | null
}): AutoProjectFromSessionDecision {
  const { sessionId, projects, selectedProjectId, projectSelectionSource } = opts

  if (!sessionId || projects.length === 0) {
    return { kind: 'noop' }
  }

  const linkedProject = findProjectBySession([...projects], sessionId)
  if (!linkedProject) {
    return { kind: 'noop' }
  }
  if (selectedProjectId === linkedProject.id) {
    return { kind: 'noop' }
  }
  if (projectSelectionSource === 'manual') {
    return { kind: 'noop' }
  }

  return { kind: 'apply', project: linkedProject }
}
