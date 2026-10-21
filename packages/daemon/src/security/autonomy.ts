import { AutonomyLevel } from '@sepilotd/core'

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
