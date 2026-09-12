/**
 * `ask` always requires human approval regardless of autonomy. `supervised`
 * prompts under supervised-family autonomy but runs without prompting under
 * autonomous autonomy (unless defaults.autonomous_honors_supervised_rules).
 */
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

export interface PolicyConfig {
  version: number
  defaults: {
    mode: ToolPolicyMode
    unmatched_policy: 'allow' | 'deny'
    /** Restore legacy behaviour: autonomous treats `supervised` rules as approval-required. Default false. */
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
  requiresApproval?: boolean
  sandbox?: boolean
}
