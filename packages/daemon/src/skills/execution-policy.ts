import type { ActiveSkillExecutionPolicy, SkillMetadata } from '@sepilotd/core'
import { builtinSkills } from './builtin.js'

interface ResolvedSkillDefinition {
  metadata: SkillMetadata
  content: string
}

const canonicalBuiltinSkills = new Map(
  builtinSkills.map((skill) => [skill.metadata.id, skill] as const),
)

const trustedBuiltinExecutionProfiles = new Map(
  builtinSkills.flatMap((skill) => skill.metadata.execution
    ? [[skill.metadata.id, {
        policy: skill.metadata.execution,
        toolNames: skill.metadata.tools,
      }] as const]
    : []),
)

export type TrustedSkillSpecialistRoute = 'researcher' | 'reviewer'

const trustedSkillRouteByTag = new Map<string, TrustedSkillSpecialistRoute>([
  ['research', 'researcher'],
  ['review', 'reviewer'],
])

const canonicalBuiltinFingerprints = new Map(
  builtinSkills.map((skill) => [
    skill.metadata.id,
    {
      metadata: skillMetadataFingerprint(skill.metadata),
      content: normalizeSkillContent(skill.content),
    },
  ] as const),
)

/**
 * Match a resolved skill against the artifact shipped in this runtime bundle.
 *
 * An ID, author string, or `builtin` tag is not an origin proof: each can be
 * copied by a workspace SKILL.md. Auto-discovered definitions are therefore
 * always rejected, and managed definitions must match the canonical metadata
 * and prompt body. `enabled` is intentionally excluded because it is a local
 * user preference preserved when built-ins are re-seeded.
 */
export function isCanonicalBuiltinSkill(
  skill: ResolvedSkillDefinition,
  expectedSkillId: string = skill.metadata.id,
): boolean {
  if (
    skill.metadata.id !== expectedSkillId
    || skill.metadata.autoDiscovered === true
    || !canonicalBuiltinSkills.has(expectedSkillId)
  ) {
    return false
  }

  const fingerprint = canonicalBuiltinFingerprints.get(expectedSkillId)
  return Boolean(
    fingerprint
    && fingerprint.metadata === skillMetadataFingerprint(skill.metadata)
    && fingerprint.content === normalizeSkillContent(skill.content),
  )
}

/** Record every canonical built-in selected for the current turn. */
export function addCanonicalBuiltinSelectedSkillId(
  target: Set<string> | undefined,
  skill: ResolvedSkillDefinition,
): void {
  if (target && isCanonicalBuiltinSkill(skill)) {
    target.add(skill.metadata.id)
  }
}

/** Record only canonical built-ins that carry trusted runtime execution rules. */
export function addCanonicalBuiltinExecutionSkillId(
  target: Set<string> | undefined,
  skill: ResolvedSkillDefinition,
): void {
  if (
    target
    && trustedBuiltinExecutionProfiles.has(skill.metadata.id)
    && isCanonicalBuiltinSkill(skill)
  ) {
    target.add(skill.metadata.id)
  }
}

/**
 * Resolve only canonical policies shipped in this trusted runtime bundle.
 * Disk and marketplace skill metadata is intentionally not consulted: an
 * untrusted SKILL.md may narrow its own prompt/tool allowlist, but it cannot
 * inject control-flow rules into the generic agent engine.
 */
export function resolveActiveSkillExecutionPolicies(
  executionSkillIds: readonly string[] | undefined,
): ActiveSkillExecutionPolicy[] {
  return (executionSkillIds ?? []).flatMap((skillId) => {
    const profile = trustedBuiltinExecutionProfiles.get(skillId)
    if (!profile) return []
    return [{
      skillId,
      policy: {
        ...profile.policy,
        stages: profile.policy.stages.map((stage) => ({
          ...stage,
          tools: [...stage.tools],
          ...(stage.requires ? { requires: [...stage.requires] } : {}),
        })),
        ...(profile.policy.argumentBindings
          ? {
              argumentBindings: profile.policy.argumentBindings.map((binding) => ({
                ...binding,
                targets: binding.targets.map((target) => ({ ...target })),
              })),
            }
          : {}),
      },
    }]
  })
}

export function resolveActiveSkillToolNames(
  executionSkillIds: readonly string[] | undefined,
): string[] {
  return [...new Set((executionSkillIds ?? []).flatMap((skillId) =>
    trustedBuiltinExecutionProfiles.get(skillId)?.toolNames ?? []
  ))]
}

/**
 * Resolve a specialist hint only when every canonical selected skill has one
 * routing tag and all of those tags agree. A skill with no specialist route is
 * part of a mixed workflow, not evidence that it can be silently ignored.
 * The hint comes from canonical built-in metadata, never from an untrusted
 * workspace or marketplace skill.
 */
export function resolveActiveSkillSpecialistRoute(
  selectedSkillIds: readonly string[] | undefined,
): TrustedSkillSpecialistRoute | undefined {
  if (!selectedSkillIds?.length) return undefined
  let agreedRoute: TrustedSkillSpecialistRoute | undefined
  for (const skillId of selectedSkillIds) {
    const skill = canonicalBuiltinSkills.get(skillId)
    if (!skill) return undefined
    const routes = [...new Set((skill.metadata.tags ?? []).flatMap((tag) => {
      const route = trustedSkillRouteByTag.get(tag)
      return route ? [route] : []
    }))]
    if (routes.length !== 1) return undefined
    const route = routes[0]!
    if (agreedRoute && agreedRoute !== route) return undefined
    agreedRoute = route
  }
  return agreedRoute
}

export function resolveSkillExecutionContext(
  selectedSkillIdsInput: ReadonlySet<string> | readonly string[],
  loadedSkillToolNames: ReadonlySet<string> | readonly string[] = [],
): {
  selectedSkillIds?: string[]
  executionSkillIds?: string[]
  skillToolNames?: string[]
  skillExecutionPolicies?: ActiveSkillExecutionPolicy[]
} {
  const selectedSkillIds = [...selectedSkillIdsInput].filter((skillId) =>
    canonicalBuiltinSkills.has(skillId)
  )
  const executionSkillIds = selectedSkillIds.filter((skillId) =>
    trustedBuiltinExecutionProfiles.has(skillId)
  )
  // Tool declarations and trusted execution policies have different trust
  // boundaries. A loaded skill may select only tools that already survived
  // request-boundary registry filtering, even when it is not a canonical
  // built-in allowed to install control-flow rules. Preserve those declared
  // tools into graph node profiles so a specialist route cannot erase the
  // capability the selected skill was loaded to provide.
  const skillToolNames = [...new Set([
    ...selectedSkillIds.flatMap((skillId) =>
      canonicalBuiltinSkills.get(skillId)?.metadata.tools ?? []
    ),
    ...loadedSkillToolNames,
  ])]
  const skillExecutionPolicies = resolveActiveSkillExecutionPolicies(executionSkillIds)
  return {
    ...(selectedSkillIds.length > 0 ? { selectedSkillIds } : {}),
    ...(executionSkillIds.length > 0 ? { executionSkillIds } : {}),
    ...(skillToolNames.length > 0 ? { skillToolNames } : {}),
    ...(skillExecutionPolicies.length > 0 ? { skillExecutionPolicies } : {}),
  }
}

function skillMetadataFingerprint(metadata: SkillMetadata): string {
  const {
    autoDiscovered: _autoDiscovered,
    enabled: _enabled,
    ...identity
  } = metadata
  return stableSerialize(identity)
}

function normalizeSkillContent(content: string): string {
  return content.replace(/\r\n/g, '\n').trim()
}

function stableSerialize(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableSerialize(item)).join(',')}]`
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, entry]) => entry !== undefined)
      .sort(([left], [right]) => left.localeCompare(right))
    return `{${entries.map(([key, entry]) =>
      `${JSON.stringify(key)}:${stableSerialize(entry)}`
    ).join(',')}}`
  }
  return JSON.stringify(value)
}
