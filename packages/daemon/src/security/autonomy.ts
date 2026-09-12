import { AutonomyLevel } from '@sepilotd/core'

/**
 * Autonomy levels do NOT form a single permissiveness line — they vary on
 * two independent axes:
 *
 *  - approval axis: supervised requires a human prompt for every
 *    non-read tool, while accept-edits / workspace-write auto-allow file
 *    edits without one. From configured `supervised`, requesting
 *    `accept-edits` would *remove* the operator's approval gate.
 *  - write-scope axis: accept-edits / workspace-write / autonomous
 *    enforce the workspace write boundary, supervised does not (any path
 *    with approval). From configured `workspace-write`, requesting
 *    `supervised` would *widen* the writable scope.
 *
 * This partial order is used only where the product declares a real ceiling,
 * such as an untrusted channel ACL. `agent.autonomy` is the local session
 * default, not such a ceiling; conflating the two made an explicit
 * `/autonomy workspace-write` selection silently collapse back to readonly.
 */
export const ALLOWED_DOWNGRADES: Readonly<Record<AutonomyLevel, readonly AutonomyLevel[]>> = {
  [AutonomyLevel.ReadOnly]: [AutonomyLevel.ReadOnly],
  [AutonomyLevel.AcceptEdits]: [AutonomyLevel.ReadOnly, AutonomyLevel.AcceptEdits],
  [AutonomyLevel.WorkspaceWrite]: [
    AutonomyLevel.ReadOnly,
    AutonomyLevel.AcceptEdits,
    AutonomyLevel.WorkspaceWrite,
  ],
  [AutonomyLevel.Supervised]: [AutonomyLevel.ReadOnly, AutonomyLevel.Supervised],
  [AutonomyLevel.Autonomous]: [
    AutonomyLevel.ReadOnly,
    AutonomyLevel.AcceptEdits,
    AutonomyLevel.WorkspaceWrite,
    AutonomyLevel.Supervised,
    AutonomyLevel.Autonomous,
  ],
}

/**
 * Backward-compatible alias retained for callers that imported the old name.
 * `configured` is a default, so any valid explicit per-turn selection wins.
 * Actual ceilings are enforced separately by channel ACL and policy layers.
 */
export function clampRequestedAutonomy(
  configured: AutonomyLevel,
  requested: string | undefined,
): AutonomyLevel {
  return resolveRequestedAutonomy(configured, requested).effective
}

export interface AutonomyResolution {
  configured: AutonomyLevel
  requested?: AutonomyLevel
  effective: AutonomyLevel
  clamped: boolean
  reason?: string
}

/** Resolve autonomy and retain the evidence surfaces need to explain clamps. */
export function resolveRequestedAutonomy(
  configured: AutonomyLevel,
  requested: string | undefined,
): AutonomyResolution {
  if (!requested) {
    return {
      configured,
      effective: configured,
      clamped: false,
    }
  }
  if (Object.values(AutonomyLevel).includes(requested as AutonomyLevel)) {
    return {
      configured,
      requested: requested as AutonomyLevel,
      effective: requested as AutonomyLevel,
      clamped: false,
    }
  }
  return {
    configured,
    effective: configured,
    clamped: true,
    reason: `Requested autonomy '${requested}' is not a recognized autonomy level`,
  }
}
