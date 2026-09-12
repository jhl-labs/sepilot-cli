import type { AutonomyLevel } from '../agent/types.js'

export type SkillSourceType = 'marketplace' | 'git' | 'url'

export interface SkillSourceRecord {
  type: SkillSourceType
  ref: string
}

export type SkillRiskTier = 'low' | 'medium' | 'high'

export type SkillVerificationMode =
  | 'builtin'
  | 'digest'
  | 'signature'
  | 'manual'
  | 'unverified'

export interface SkillSignatureRecord {
  algorithm: 'ed25519'
  keyId?: string
  value: string
  verified?: boolean
}

export interface SkillScanRecord {
  // 'unknown' = no content scanner ran (the install pipeline does not ship a
  // real static/secret scanner yet, so it must not claim 'pass').
  result: 'pass' | 'warn' | 'fail' | 'unknown'
  checkedAt: string
  errors?: string[]
  warnings?: string[]
}

export interface SkillProvenanceRecord {
  source: SkillSourceRecord
  digest: string
  installedAt?: string
  verified: boolean
  verification: SkillVerificationMode
  publisher?: string
  sourceRef?: string
  signature?: SkillSignatureRecord
  scan?: SkillScanRecord
}

export interface SkillPermissionManifest {
  tools?: string[]
  network?: string[]
  files?: string[]
}

/**
 * Declarative, model-independent execution stage for a skill turn.
 *
 * Stages are ordered as declared. A later stage may be skipped when optional,
 * but once a later stage succeeds the runtime will not execute an earlier one
 * in the same user turn. This lets a skill express bounded workflows without
 * teaching the generic agent engine about a particular skill or tool family.
 */
export interface SkillExecutionStage {
  id: string
  tools: string[]
  /**
   * Optional hard cardinality boundary for a stage whose product semantics
   * genuinely require one or a small fixed number of successful outcomes.
   * Omit it for evidence-gathering stages whose useful call count depends on
   * the request; global iteration, batch, duplicate, and stuck-loop controls
   * still bound the turn.
   */
  maxCallsPerTurn?: number
  requires?: string[]
  requiredForCompletion?: boolean
  /**
   * By default, only a successful result advances the stage. Use
   * `executed-outcome` when one bounded external attempt must advance the
   * workflow even if the provider reports an error. Pre-execution policy or
   * approval rejection never satisfies this mode.
   */
  satisfyOn?: 'success' | 'executed-outcome'
}

export interface SkillExecutionArgumentTarget {
  stage: string
  argument: string
}

/**
 * Keeps selected tool arguments consistent across otherwise generic stages.
 * Missing values may be accepted for cursor-based tools (for example, a tool
 * that means "the current item" when no index is supplied), but two explicit
 * values in the same binding must always agree.
 */
export interface SkillExecutionArgumentBinding {
  id: string
  targets: SkillExecutionArgumentTarget[]
  allowMissing?: boolean
}

export interface SkillExecutionPolicy {
  stages: SkillExecutionStage[]
  argumentBindings?: SkillExecutionArgumentBinding[]
  maxCompletionRetries?: number
}

/** Resolved policy attached only after the corresponding skill was loaded. */
export interface ActiveSkillExecutionPolicy {
  skillId: string
  policy: SkillExecutionPolicy
}

export interface SkillMetadata {
  id: string
  name: string
  version: string
  description: string
  author?: string
  tags?: string[]
  tools: string[]
  autonomy_required?: AutonomyLevel
  created?: string
  source?: SkillSourceRecord
  provenance?: SkillProvenanceRecord
  risk_tier?: SkillRiskTier
  permissions?: SkillPermissionManifest
  /** Optional deterministic runtime constraints for this skill's tool workflow. */
  execution?: SkillExecutionPolicy
  /**
   * True when the skill was auto-discovered from a project (cwd) or home
   * compatibility root (e.g. `.claude/skills`) rather than explicitly installed
   * into the managed skills dir or seeded as a built-in. Auto-discovered skills
   * are untrusted supply-chain content: their instructions must be surfaced to
   * the model with an untrusted-provenance banner, never as trusted directives.
   */
  autoDiscovered?: boolean
  /**
   * Whether this skill is active. Absent = enabled (true). When false the
   * skill is hidden from the agent's catalog (system prompt) and the `skill`
   * tool, but is still listed by management surfaces. Lets built-in skills
   * ship opt-in (e.g. `container-sandbox`, which needs Docker; `env-setup`,
   * which configures a corporate-proxy machine; `kubectl`, which talks to
   * a live cluster).
   */
  enabled?: boolean
}
