// Pulled out of App.tsx so the "if the current session is already
// linked to a project, auto-select that project unless the operator
// has manually picked one" rule has a single named home with unit
// tests. The helper is pure: the caller threads the result back into
// applyProjectSelection.

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

  // findProjectBySession is typed against DaemonProject; spread to a
  // mutable array because the public ReadonlyArray signature is
  // narrower than what the lookup utility currently asks for.
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
