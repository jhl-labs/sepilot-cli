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

export function addCanonicalBuiltinSelectedSkillId(
  target: Set<string> | undefined,
  skill: ResolvedSkillDefinition,
): void {
  if (target && isCanonicalBuiltinSkill(skill)) {
    target.add(skill.metadata.id)
  }
}

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
