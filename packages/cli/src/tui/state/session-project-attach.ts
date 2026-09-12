// Pulled out of App.tsx so the "should the current session be
// attached to the selected project right now?" decision can be
// reasoned about without the surrounding effect/closure noise. The
// helper is pure: it inspects state and returns a tagged action the
// effect should perform — reset the dedup ref, record an existing
// link, skip, or fire the attach call.
//
// The actual attach call + state updates stay in the effect because
// they touch React-side setters (setProjects, dispatch) and the
// outcome interleaves with the dedup ref. Keeping the decision
// pure-functional makes the gate logic unit-testable in isolation
// while leaving the effect body small and top-down.

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
