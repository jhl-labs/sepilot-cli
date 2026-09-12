import type { SkillMetadata } from '@sepilotd/core'

export const MAX_PROMPT_SKILL_DESCRIPTION_CHARS = 500
export const MAX_AUTOMATIC_ROUTER_SKILLS = 32

/**
 * Human-readable skill label that keeps the immutable lookup id explicit.
 * Names are presentation text and may change or collide; agent calls must use
 * `metadata.id`.
 */
export function formatSkillDisplayLabel(
  metadata: Pick<SkillMetadata, 'id' | 'name'>,
): string {
  return metadata.name === metadata.id
    ? metadata.id
    : `${metadata.name} (id: ${metadata.id})`
}

/**
 * Skill descriptions are catalog data, not an instruction channel. Keep them
 * one-line and bounded before placing them in a system/router prompt so a
 * project SKILL.md cannot turn a metadata field into a second prompt.
 */
export function formatSkillDescriptionForPrompt(description: string): string {
  const compact = description.trim().replace(/\s+/gu, ' ')
  if (compact.length <= MAX_PROMPT_SKILL_DESCRIPTION_CHARS) return compact
  return `${compact.slice(0, MAX_PROMPT_SKILL_DESCRIPTION_CHARS - 1)}…`
}

/** Project/home compatibility skills require an explicit user selection. */
export function isSkillEligibleForAutomaticRouting(
  skill: Pick<SkillMetadata, 'autoDiscovered' | 'enabled'>,
): boolean {
  return skill.enabled !== false && skill.autoDiscovered !== true
}
