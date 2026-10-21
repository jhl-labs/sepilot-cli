export type ToolPolicyMode = 'autonomous' | 'supervised' | 'ask' | 'blocked'

export interface PolicyRule {
  mode: ToolPolicyMode
  deny_patterns?: string[]
  deny_paths?: string[]
  deny_urls?: string[]
  deny_executables?: string[]
  allow_patterns?: string[]
  allow_paths?: string[]
  allow_urls?: string[]
  allow_hosts?: string[]
  max_timeout_ms?: number
  max_output_bytes?: number
}

export interface ResourceAccessPolicy {
  version: 1
  enabled: boolean
  tools: { allow: string[]; deny: string[] }
  urls: { allow: string[]; deny: string[] }
  paths: { allow: string[]; deny: string[] }
  internalTools?: string[]
}

export interface PolicyConfig {
  access?: ResourceAccessPolicy
  version: number
  defaults: {
    mode: ToolPolicyMode
    unmatched_policy: 'allow' | 'deny'
    autonomous_honors_supervised_rules?: boolean
    max_timeout_ms: number
    max_output_bytes: number
  }
  tools: Record<string, PolicyRule>
  elevated?: Record<string, Partial<PolicyRule>>
}

export interface PolicyCheckResult {
  allowed: boolean
  reason?: string
  code?: 'workspace_confinement_required'
  requiresApproval?: boolean
  sandbox?: boolean
}
