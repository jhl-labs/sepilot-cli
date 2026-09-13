import type { AutonomyLevel } from '@sepilotd/core'
import { isPolicyReadOnlyTool, type PolicyEngine } from '../security/policy-engine.js'

/**
 * Static approval posture of a registered tool for one run, evaluated from
 * the policy rule mode, the run autonomy, and the primary agent — never from
 * a concrete input. It tells the model up front which tools will pause for
 * approval and which are unavailable, so it can plan around them instead of
 * discovering each block one call at a time.
 *
 * `auto` is an optimistic label: input-dependent gates (deny_patterns,
 * deny_paths, workspace boundary, allow_patterns) still apply at call time.
 */
export type ToolApprovalPosture = 'auto' | 'ask' | 'blocked'

export interface ToolApprovalPostureEntry {
  posture: ToolApprovalPosture
  /** Short human-readable cause for `ask` / `blocked`. */
  reason?: string
}

export type ToolApprovalPostureMap = Map<string, ToolApprovalPostureEntry>

export interface DescribeToolApprovalPostureOptions {
  /** CI/automation auto-approval: `ask` becomes `auto`. */
  autoApprove?: boolean
}

const PROMPTING_AUTONOMIES: ReadonlySet<string> = new Set([
  'supervised',
  'accept-edits',
  'workspace-write',
])

export function describeToolApprovalPosture(
  tools: ReadonlyArray<{ name: string }>,
  policy: Pick<PolicyEngine, 'ruleModeFor'>,
  autonomy: AutonomyLevel,
  primaryAgentId?: string,
  options: DescribeToolApprovalPostureOptions = {},
): ToolApprovalPostureMap {
  const result: ToolApprovalPostureMap = new Map()
  const planAgent = primaryAgentId?.trim().toLowerCase() === 'plan'
  const promptingAutonomy = PROMPTING_AUTONOMIES.has(autonomy)

  for (const tool of tools) {
    const name = tool.name
    if (result.has(name)) continue
    const readOnly = isPolicyReadOnlyTool(name)
    const rule = policy.ruleModeFor(name)

    if (autonomy === 'readonly' && !readOnly) {
      result.set(name, { posture: 'blocked', reason: 'read-only autonomy' })
      continue
    }
    if (planAgent && !readOnly) {
      result.set(name, { posture: 'blocked', reason: 'plan mode' })
      continue
    }
    if (rule.mode === 'blocked') {
      result.set(name, { posture: 'blocked', reason: 'blocked by policy' })
      continue
    }
    if (readOnly) {
      result.set(name, { posture: 'auto' })
      continue
    }
    if (!rule.hasRule && rule.unmatchedPolicy === 'deny') {
      result.set(name, { posture: 'blocked', reason: 'no policy rule' })
      continue
    }
    if (rule.workspaceWriteFastPath
      && (autonomy === 'accept-edits' || autonomy === 'workspace-write')) {
      result.set(name, { posture: 'auto' })
      continue
    }
    const ruleRequestsPrompt = rule.mode === 'ask'
      || (rule.mode === 'supervised'
        && (autonomy !== 'autonomous' || rule.honorsSupervisedUnderAutonomous))
    const autonomyPrompts = promptingAutonomy && !rule.userMemoryWrite
    if (ruleRequestsPrompt || autonomyPrompts) {
      if (options.autoApprove) {
        result.set(name, { posture: 'auto' })
      } else {
        result.set(name, {
          posture: 'ask',
          reason: rule.mode === 'ask' ? 'ask rule' : 'supervised',
        })
      }
      continue
    }
    result.set(name, { posture: 'auto' })
  }
  return result
}

export const TOOL_APPROVAL_ASK_SUFFIX = ' [requires approval]'

/** Description suffix for a tool definition in the outbound catalog. */
export function toolApprovalDescriptionSuffix(
  entry: ToolApprovalPostureEntry | undefined,
): string {
  if (!entry) return ''
  if (entry.posture === 'ask') return TOOL_APPROVAL_ASK_SUFFIX
  if (entry.posture === 'blocked') return ` [unavailable: ${entry.reason ?? 'blocked'}]`
  return ''
}

/** Names grouped by posture, in registry order. */
export function partitionToolApprovalPosture(
  posture: ReadonlyMap<string, ToolApprovalPostureEntry>,
): { ask: string[]; blocked: string[] } {
  const ask: string[] = []
  const blocked: string[] = []
  for (const [name, entry] of posture) {
    if (entry.posture === 'ask') ask.push(name)
    else if (entry.posture === 'blocked') blocked.push(name)
  }
  return { ask, blocked }
}
