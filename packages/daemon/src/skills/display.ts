import type { SkillMetadata } from '@sepilotd/core'

export const MAX_PROMPT_SKILL_DESCRIPTION_CHARS = 240
export const MAX_AUTOMATIC_ROUTER_SKILLS = 32

export function formatSkillDisplayLabel(
  metadata: Pick<SkillMetadata, 'id' | 'name'>,
): string {
  return metadata.name === metadata.id
    ? metadata.id
    : `${metadata.name} (id: ${metadata.id})`
}

export function formatSkillDescriptionForPrompt(description: string): string {
  const compact = description.trim().replace(/\s+/gu, ' ')
  if (compact.length <= MAX_PROMPT_SKILL_DESCRIPTION_CHARS) return compact
  return `${compact.slice(0, MAX_PROMPT_SKILL_DESCRIPTION_CHARS - 1)}…`
}

export function isSkillEligibleForAutomaticRouting(
  skill: Pick<SkillMetadata, 'autoDiscovered' | 'enabled'>,
): boolean {
  return skill.enabled !== false && skill.autoDiscovered !== true
}
