export const SCHEDULED_AGENT_PROFILE_VERSION = 1 as const
export const MAX_SCHEDULED_AGENT_SKILLS = 8
export const MAX_SCHEDULED_AGENT_SKILL_ID_CHARS = 256

export interface ScheduledAgentSkillRef {
  name: string
}

export interface ScheduledAgentProfileV1 {
  version: typeof SCHEDULED_AGENT_PROFILE_VERSION
  skillRefs: ScheduledAgentSkillRef[]
}

export class ScheduledAgentProfileError extends Error {
  readonly code = 'INVALID_SCHEDULED_AGENT_PROFILE'

  constructor(message: string) {
    super(message)
    this.name = 'ScheduledAgentProfileError'
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Normalize the durable skill identity list without guessing from prompt text.
 * Duplicate ids are collapsed in first-seen order; malformed or over-broad
 * selections fail closed instead of being silently truncated.
 */
export function normalizeScheduledAgentSkillRefs(
  refs: readonly ScheduledAgentSkillRef[],
): ScheduledAgentSkillRef[] {
  const names: string[] = []
  for (const ref of refs) {
    if (!isRecord(ref) || typeof ref.name !== 'string') {
      throw new ScheduledAgentProfileError('Each scheduled skill reference must contain a string name.')
    }
    const name = ref.name.trim()
    if (!name) {
      throw new ScheduledAgentProfileError('Scheduled skill names must not be empty.')
    }
    if (name.length > MAX_SCHEDULED_AGENT_SKILL_ID_CHARS) {
      throw new ScheduledAgentProfileError(
        `Scheduled skill names must be at most ${MAX_SCHEDULED_AGENT_SKILL_ID_CHARS} characters.`,
      )
    }
    if (!names.includes(name)) names.push(name)
  }
  if (names.length > MAX_SCHEDULED_AGENT_SKILLS) {
    throw new ScheduledAgentProfileError(
      `At most ${MAX_SCHEDULED_AGENT_SKILLS} unique skills may be selected for a scheduled task.`,
    )
  }
  return names.map((name) => ({ name }))
}

/** Parse the reserved scheduler metadata namespace. Unknown versions fail closed. */
export function scheduledAgentSkillRefsFromMetadata(
  metadata: Record<string, unknown> | null | undefined,
): ScheduledAgentSkillRef[] {
  if (!metadata || metadata.scheduledAgent === undefined) return []
  const profile = metadata.scheduledAgent
  if (!isRecord(profile)) {
    throw new ScheduledAgentProfileError('scheduledAgent metadata must be an object.')
  }
  if (profile.version !== SCHEDULED_AGENT_PROFILE_VERSION) {
    throw new ScheduledAgentProfileError(
      `Unsupported scheduledAgent metadata version: ${String(profile.version)}.`,
    )
  }
  if (!Array.isArray(profile.skillRefs)) {
    throw new ScheduledAgentProfileError('scheduledAgent.skillRefs must be an array.')
  }
  return normalizeScheduledAgentSkillRefs(profile.skillRefs as ScheduledAgentSkillRef[])
}

/**
 * Replace only the scheduler-owned profile while preserving unrelated
 * notification, delivery, and app metadata. An empty list clears the profile.
 */
export function withScheduledAgentSkillRefs(
  metadata: Record<string, unknown> | null | undefined,
  refs: readonly ScheduledAgentSkillRef[],
): Record<string, unknown> | null {
  const normalized = normalizeScheduledAgentSkillRefs(refs)
  const next = { ...(metadata ?? {}) }
  if (normalized.length === 0) {
    delete next.scheduledAgent
  } else {
    next.scheduledAgent = {
      version: SCHEDULED_AGENT_PROFILE_VERSION,
      skillRefs: normalized,
    } satisfies ScheduledAgentProfileV1
  }
  return Object.keys(next).length > 0 ? next : null
}
