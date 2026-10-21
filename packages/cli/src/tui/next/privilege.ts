import { colors } from '../theme.js'

export type PrivilegeLevel = 'readonly' | 'accept-edits' | 'workspace-write' | 'supervised' | 'autonomous'

export const PRIVILEGE_CYCLE: readonly PrivilegeLevel[] = ['supervised', 'accept-edits', 'workspace-write', 'readonly']

export function privilegeCycle(configuredDefault?: string | null): PrivilegeLevel[] {
  const cycle = [...PRIVILEGE_CYCLE]
  if (configuredDefault && isPrivilegeLevel(configuredDefault) && !cycle.includes(configuredDefault)) cycle.push(configuredDefault)
  return cycle
}

export function nextPrivilege(current: string, configuredDefault?: string | null): PrivilegeLevel {
  const cycle = privilegeCycle(configuredDefault)
  const index = cycle.indexOf(current as PrivilegeLevel)
  return index < 0 ? cycle[0]! : cycle[(index + 1) % cycle.length]!
}

export const PRIVILEGE_DESCRIPTIONS: Readonly<Record<PrivilegeLevel, string>> = {
  readonly: 'read-only: inspect and answer, no changes',
  supervised: 'ask: approve every change and command',
  'accept-edits': 'auto-edit: file edits run without asking, commands ask',
  'workspace-write': 'workspace: file edits run inside the workspace only, commands ask',
  autonomous: 'full: no approvals (use with care)',
}

export function privilegeColor(level: string): string {
  switch (level) {
    case 'readonly': return colors.info
    case 'accept-edits': return colors.success
    case 'workspace-write': return colors.warning
    case 'autonomous': return colors.error
    default: return colors.text
  }
}

export function isPrivilegeLevel(value: string): value is PrivilegeLevel {
  return value in PRIVILEGE_DESCRIPTIONS
}
