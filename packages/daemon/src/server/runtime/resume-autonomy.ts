import { AutonomyLevel } from '@sepilotd/core'

export function resolveResumeAutonomy(saved: unknown, current: AutonomyLevel): AutonomyLevel {
  if (saved === undefined) return current
  if (!Object.values(AutonomyLevel).includes(saved as AutonomyLevel)) return AutonomyLevel.ReadOnly
  if (saved === AutonomyLevel.ReadOnly || current === AutonomyLevel.ReadOnly) return AutonomyLevel.ReadOnly
  if (saved === current) return current
  if (saved === AutonomyLevel.Autonomous) return current
  if (current === AutonomyLevel.Autonomous) return saved as AutonomyLevel
  return AutonomyLevel.Supervised
}
