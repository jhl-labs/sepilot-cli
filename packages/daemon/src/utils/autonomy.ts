import { AutonomyLevel } from '@sepilotd/core'

const SATISFIES: Readonly<Record<AutonomyLevel, readonly AutonomyLevel[]>> = {
  [AutonomyLevel.ReadOnly]: [
    AutonomyLevel.ReadOnly,
    AutonomyLevel.AcceptEdits,
    AutonomyLevel.WorkspaceWrite,
    AutonomyLevel.Supervised,
    AutonomyLevel.Autonomous,
  ],
  [AutonomyLevel.AcceptEdits]: [
    AutonomyLevel.AcceptEdits,
    AutonomyLevel.WorkspaceWrite,
    AutonomyLevel.Supervised,
    AutonomyLevel.Autonomous,
  ],
  [AutonomyLevel.WorkspaceWrite]: [
    AutonomyLevel.WorkspaceWrite,
    AutonomyLevel.Supervised,
    AutonomyLevel.Autonomous,
  ],
  [AutonomyLevel.Supervised]: [
    AutonomyLevel.AcceptEdits,
    AutonomyLevel.WorkspaceWrite,
    AutonomyLevel.Supervised,
  ],
  [AutonomyLevel.Autonomous]: [AutonomyLevel.Autonomous],
}

export function autonomyAllows(
  required: AutonomyLevel | undefined,
  current: AutonomyLevel | undefined,
): boolean {
  if (!required || !current) return true
  return (SATISFIES[required] ?? []).includes(current)
}
