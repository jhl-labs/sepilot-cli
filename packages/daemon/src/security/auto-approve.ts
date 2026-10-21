export const AUTO_APPROVE_ENV_VAR = 'SEPILOTD_UNSAFE_AUTO_APPROVE'
export const AUTO_APPROVE_ENV_VALUE = '1'

export interface AutoApproveInputs {
  env: Record<string, string | undefined>
  cliFlag: boolean
}

export function isAutoApproveActive(inputs: AutoApproveInputs): boolean {
  if (!inputs.cliFlag) return false
  return inputs.env[AUTO_APPROVE_ENV_VAR] === AUTO_APPROVE_ENV_VALUE
}
