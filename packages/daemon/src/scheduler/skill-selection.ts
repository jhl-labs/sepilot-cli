import {
  normalizeScheduledAgentSkillRefs,
  scheduledAgentSkillRefsFromMetadata,
  type ScheduledAgentSkillRef,
} from '@sepilotd/api-client'
import type { AutonomyLevel, ISkillRegistry } from '@sepilotd/core'
import { resolveSkillExecutionContext } from '../skills/execution-policy.js'
import type { ToolRegistry } from '../tools/registry.js'
import { resolveSkillRefsContent } from '../server/routes/chat-skills.js'

export interface ScheduledSkillResolverDeps {
  skillRegistry?: ISkillRegistry
  toolRegistry: ToolRegistry
  autonomy: AutonomyLevel
}

export interface ResolvedScheduledSkillContext {
  refs: ScheduledAgentSkillRef[]
  systemPrompt?: string
  selectedSkillIds?: string[]
  executionSkillIds?: string[]
  skillToolNames?: string[]
  skillExecutionPolicies?: ReturnType<typeof resolveSkillExecutionContext>['skillExecutionPolicies']
}

/**
 * Rehydrate the same structured skill selection used by interactive chat.
 * A persisted selection is an execution contract: missing registries,
 * disabled/missing skills, malformed metadata, and policy incompatibilities
 * all fail the run instead of silently executing the bare instruction.
 */
export async function resolveScheduledSkillContext(
  metadata: Record<string, unknown> | null | undefined,
  deps: ScheduledSkillResolverDeps,
): Promise<ResolvedScheduledSkillContext> {
  const refs = scheduledAgentSkillRefsFromMetadata(metadata)
  return resolveScheduledSkillRefsContext(refs, deps)
}

export async function resolveScheduledSkillRefsContext(
  inputRefs: readonly ScheduledAgentSkillRef[],
  deps: ScheduledSkillResolverDeps,
): Promise<ResolvedScheduledSkillContext> {
  const refs = normalizeScheduledAgentSkillRefs(inputRefs)
  if (refs.length === 0) return { refs }
  if (!deps.skillRegistry) {
    throw new Error('SCHEDULED_SKILL_REGISTRY_UNAVAILABLE: selected skills cannot be resolved')
  }

  const declaredToolNames = new Set<string>()
  const loadedSkillIds = new Set<string>()
  const systemPrompt = await resolveSkillRefsContent(
    refs,
    deps.skillRegistry,
    deps.toolRegistry,
    undefined,
    deps.autonomy,
    declaredToolNames,
    undefined,
    loadedSkillIds,
  )
  return {
    refs,
    systemPrompt,
    ...resolveSkillExecutionContext(loadedSkillIds, declaredToolNames),
  }
}

export async function validateScheduledSkillRefs(
  refs: readonly ScheduledAgentSkillRef[],
  deps: ScheduledSkillResolverDeps,
): Promise<void> {
  if (refs.length === 0) return
  await resolveScheduledSkillRefsContext(refs, deps)
}
