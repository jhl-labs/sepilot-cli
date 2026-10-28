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

export interface SkillExecutionStage {
  id: string
  tools: string[]
  maxCallsPerTurn?: number
  requires?: string[]
  requiredForCompletion?: boolean
  satisfyOn?: 'success' | 'executed-outcome'
}

export interface SkillExecutionArgumentTarget {
  stage: string
  argument: string
}

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

export interface ActiveSkillExecutionPolicy {
  skillId: string
  policy: SkillExecutionPolicy
}

export type SkillDiscoveryScope = 'home' | 'project'

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
  execution?: SkillExecutionPolicy
  autoDiscovered?: boolean
  discoveryScope?: SkillDiscoveryScope
  enabled?: boolean
}
