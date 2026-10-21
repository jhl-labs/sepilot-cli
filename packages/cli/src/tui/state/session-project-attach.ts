import type { ProjectSelectionSource } from '../hooks/useProjectSelection.js'

interface ProjectLike {
  id: string
  sessionIds: string[]
}

export type SessionProjectAttachDecision =
  | { kind: 'reset'; nextRef: null }
  | { kind: 'already-attached'; nextRef: string }
  | { kind: 'skip'; reason: 'linked-elsewhere' | 'duplicate-fire' }
  | { kind: 'attach'; attachmentKey: string; projectId: string; sessionId: string }

export function decideSessionProjectAttach(opts: {
  sessionId: string | null
  selectedProject: ProjectLike | null
  projects: ReadonlyArray<ProjectLike>
  projectSelectionSource: ProjectSelectionSource | null
  lastAttachedKey: string | null
}): SessionProjectAttachDecision {
  const {
    sessionId,
    selectedProject,
    projects,
    projectSelectionSource,
    lastAttachedKey,
  } = opts

  if (!sessionId || !selectedProject) {
    return { kind: 'reset', nextRef: null }
  }

  if (selectedProject.sessionIds.includes(sessionId)) {
    return {
      kind: 'already-attached',
      nextRef: `${selectedProject.id}:${sessionId}`,
    }
  }

  const linkedElsewhere = projects.some(
    (project) =>
      project.id !== selectedProject.id
      && project.sessionIds.includes(sessionId),
  )
  if (linkedElsewhere && projectSelectionSource !== 'manual') {
    return { kind: 'skip', reason: 'linked-elsewhere' }
  }

  const attachmentKey = `${selectedProject.id}:${sessionId}`
  if (lastAttachedKey === attachmentKey) {
    return { kind: 'skip', reason: 'duplicate-fire' }
  }

  return {
    kind: 'attach',
    attachmentKey,
    projectId: selectedProject.id,
    sessionId,
  }
}
