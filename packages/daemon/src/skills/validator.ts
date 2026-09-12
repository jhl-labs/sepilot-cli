import type { AutonomyLevel, SkillMetadata } from '@sepilotd/core'
import type { ToolRegistry } from '../tools/registry.js'
import type { PolicyEngine } from '../security/policy-engine.js'
import { autonomyAllows } from '../utils/autonomy.js'

export interface SkillValidationResult {
  valid: boolean
  errors: string[]
  warnings: string[]
}

export function validateSkill(
  skill: SkillMetadata,
  content: string,
  toolRegistry: ToolRegistry,
  policyEngine: PolicyEngine,
  currentAutonomy: AutonomyLevel,
): SkillValidationResult {
  const errors: string[] = []
  const warnings: string[] = []

  // 1. Check required fields
  if (!skill.name) errors.push('Skill name is required')
  if (!skill.version) errors.push('Skill version is required')
  if (!skill.description) errors.push('Skill description is required')

  // 2. Check all required tools exist
  if (skill.tools) {
    for (const toolName of skill.tools) {
      if (!toolRegistry.get(toolName)) {
        errors.push(`Required tool not available: ${toolName}`)
      }
    }
  }

  if (skill.execution) {
    const declaredTools = new Set(skill.tools ?? [])
    const stageIds = new Set<string>()
    const stageIndex = new Map<string, number>()
    const stageByTool = new Map<string, string>()
    if (skill.execution.stages.length === 0) {
      errors.push('Execution policy requires at least one stage')
    }
    for (const [index, stage] of skill.execution.stages.entries()) {
      if (!stage.id.trim()) errors.push('Execution stage id is required')
      if (stageIds.has(stage.id)) errors.push(`Duplicate execution stage id: ${stage.id}`)
      stageIds.add(stage.id)
      stageIndex.set(stage.id, index)
      if (stage.tools.length === 0) errors.push(`Execution stage ${stage.id} requires a tool`)
      for (const toolName of stage.tools) {
        const previousStage = stageByTool.get(toolName)
        if (previousStage && previousStage !== stage.id) {
          errors.push(
            `Execution tool ${toolName} cannot be assigned to multiple stages: ${previousStage}, ${stage.id}`,
          )
        } else {
          stageByTool.set(toolName, stage.id)
        }
        if (!declaredTools.has(toolName)) {
          errors.push(`Execution stage ${stage.id} uses undeclared tool: ${toolName}`)
        }
      }
      if (
        stage.maxCallsPerTurn !== undefined
        && (!Number.isInteger(stage.maxCallsPerTurn) || stage.maxCallsPerTurn < 1 || stage.maxCallsPerTurn > 10)
      ) {
        errors.push(`Execution stage ${stage.id} maxCallsPerTurn must be an integer within 1..10`)
      }
      if (
        stage.satisfyOn !== undefined
        && stage.satisfyOn !== 'success'
        && stage.satisfyOn !== 'executed-outcome'
      ) {
        errors.push(
          `Execution stage ${stage.id} satisfyOn must be success or executed-outcome`,
        )
      }
    }
    for (const [index, stage] of skill.execution.stages.entries()) {
      for (const dependency of stage.requires ?? []) {
        const dependencyIndex = stageIndex.get(dependency)
        if (dependencyIndex === undefined) {
          errors.push(`Execution stage ${stage.id} requires unknown stage: ${dependency}`)
        } else if (dependencyIndex >= index) {
          errors.push(`Execution stage ${stage.id} must require an earlier stage: ${dependency}`)
        }
      }
    }
    const bindingIds = new Set<string>()
    for (const binding of skill.execution.argumentBindings ?? []) {
      if (!binding.id.trim()) errors.push('Execution argument binding id is required')
      if (bindingIds.has(binding.id)) {
        errors.push(`Duplicate execution argument binding id: ${binding.id}`)
      }
      bindingIds.add(binding.id)
      if (binding.targets.length < 2) {
        errors.push(`Execution argument binding ${binding.id} requires at least two targets`)
      }
      const boundStages = new Set<string>()
      for (const target of binding.targets) {
        if (!stageIds.has(target.stage)) {
          errors.push(
            `Execution argument binding ${binding.id} targets unknown stage: ${target.stage}`,
          )
        }
        if (!target.argument.trim()) {
          errors.push(`Execution argument binding ${binding.id} requires an argument name`)
        }
        if (boundStages.has(target.stage)) {
          errors.push(
            `Execution argument binding ${binding.id} targets stage more than once: ${target.stage}`,
          )
        }
        boundStages.add(target.stage)
      }
    }
    if (
      skill.execution.maxCompletionRetries !== undefined
      && (
        !Number.isInteger(skill.execution.maxCompletionRetries)
        || skill.execution.maxCompletionRetries < 0
        || skill.execution.maxCompletionRetries > 3
      )
    ) {
      errors.push('Execution maxCompletionRetries must be an integer within 0..3')
    }
  }

  // 3. Check autonomy level
  if (skill.autonomy_required) {
    const requiredLevel = skill.autonomy_required
    if (!autonomyAllows(requiredLevel, currentAutonomy)) {
      errors.push(`Skill requires ${requiredLevel} autonomy, but current level is ${currentAutonomy}`)
    }
  }

  // 4. Check for dangerous patterns in skill content
  const dangerousPatterns = [
    /rm\s+-rf\s+\//,
    /eval\s*\(/,
    /curl.*\|\s*bash/,
    /wget.*\|\s*bash/,
    /sudo\s+/,
    /chmod\s+777/,
  ]
  for (const pattern of dangerousPatterns) {
    if (pattern.test(content)) {
      warnings.push(`Potentially dangerous pattern found: ${pattern.source}`)
    }
  }

  // 5. Check tool policy for each declared tool
  if (skill.tools) {
    for (const toolName of skill.tools) {
      const result = policyEngine.check({ tool: toolName, input: {} }, currentAutonomy)
      if (!result.allowed && !result.requiresApproval) {
        warnings.push(`Tool ${toolName} is blocked by policy`)
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  }
}
