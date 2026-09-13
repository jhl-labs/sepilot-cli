import { AutonomyLevel } from '@sepilotd/core'

/** A restart never silently upgrades the authority that created a checkpoint. */
export function resolveResumeAutonomy(saved: unknown, current: AutonomyLevel): AutonomyLevel {
  if (saved === undefined) return current // Legacy checkpoints retain their historical contract.
  if (!Object.values(AutonomyLevel).includes(saved as AutonomyLevel)) return AutonomyLevel.ReadOnly
  if (saved === AutonomyLevel.ReadOnly || current === AutonomyLevel.ReadOnly) return AutonomyLevel.ReadOnly
  if (saved === current) return current
  if (saved === AutonomyLevel.Autonomous) return current
  if (current === AutonomyLevel.Autonomous) return saved as AutonomyLevel
  // Different restricted modes have non-identical capabilities. Require human
  // decisions instead of selecting the more permissive side of that boundary.
  return AutonomyLevel.Supervised
}
