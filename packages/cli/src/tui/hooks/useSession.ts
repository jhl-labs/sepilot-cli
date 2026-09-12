import type {
  DaemonArtifact,
  DaemonSessionDetail,
} from '@sepilotd/api-client'
import {
  buildHydratedCliSessionState,
  type HydratedCliSessionState,
} from '../utils/surface.js'
export type HydratedSessionState = HydratedCliSessionState

export function hydrateSession(
  session: DaemonSessionDetail,
  artifacts: DaemonArtifact[] = [],
): HydratedSessionState {
  return buildHydratedCliSessionState(session, artifacts)
}
