export const AUTO_APPROVE_ENV_VAR = 'SEPILOTD_UNSAFE_AUTO_APPROVE'
export const AUTO_APPROVE_ENV_VALUE = '1'

export interface AutoApproveInputs {
  env: Record<string, string | undefined>
  cliFlag: boolean
}

/**
 * CI auto-approve requires BOTH an env var AND an explicit CLI flag.
 * Either alone does not activate — this guards against accidental triggering
 * in terminals that inherit unexpected environment variables.
 *
 * When active, the policy engine should downgrade 'ask' decisions to 'allow'
 * for this run only, and must still honour explicit 'deny' rules. Every
 * auto-approved tool call must be marked in the audit log so operators can
 * trace what ran under this mode.
 */
export function isAutoApproveActive(inputs: AutoApproveInputs): boolean {
  if (!inputs.cliFlag) return false
  return inputs.env[AUTO_APPROVE_ENV_VAR] === AUTO_APPROVE_ENV_VALUE
}
