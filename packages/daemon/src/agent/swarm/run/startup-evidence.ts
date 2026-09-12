import type {
  SwarmAgentHandle,
  SwarmAgentName,
  SwarmAgentRuntimeName,
  SwarmAgentRecoveryHint,
  SwarmAgentStartupEvidence,
  SwarmAgentStartupFailureKind,
  SwarmAgentStartupState,
} from '@sepilotd/core'

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

const SENSITIVE_NAME_RE = /(?:api[_-]?key|token|secret|password|passwd|authorization|credential|client[_-]?secret|private[_-]?key)/i
const SENSITIVE_ASSIGNMENT_RE = /((?:api[_-]?key|token|secret|password|passwd|credential|client[_-]?secret|private[_-]?key)\s*[:=]\s*)([^\s'",<>&]+)/gi
const AUTHORIZATION_ASSIGNMENT_RE = /((?:authorization|auth)\s*[:=]\s*)(bearer\s+)?[^\s'",<>&]+/gi
const BEARER_TOKEN_RE = /(bearer\s+)[A-Za-z0-9._~+/=-]+/gi

function redactUrl(value: string): string {
  if (!value.includes('://')) return value
  try {
    const parsed = new URL(value)
    if (parsed.username) parsed.username = 'redacted'
    if (parsed.password) parsed.password = 'redacted'
    for (const key of [...parsed.searchParams.keys()]) {
      if (SENSITIVE_NAME_RE.test(key)) parsed.searchParams.set(key, 'redacted')
    }
    return parsed.toString()
  } catch {
    return value
  }
}

export function redactStartupEvidenceText(value: string): string {
  return redactUrl(value)
    .replace(AUTHORIZATION_ASSIGNMENT_RE, (_match, prefix: string, bearerPrefix?: string) =>
      `${prefix}${bearerPrefix ?? ''}redacted`)
    .replace(SENSITIVE_ASSIGNMENT_RE, '$1redacted')
    .replace(BEARER_TOKEN_RE, '$1redacted')
}

export function formatStartupCommand(parts: readonly string[]): string {
  let redactNext = false
  return parts.map((part) => {
    if (redactNext) {
      redactNext = false
      return 'redacted'
    }
    const assignment = part.match(/^([^=\s]+)=(.*)$/)
    if (assignment && !part.includes('://') && SENSITIVE_NAME_RE.test(assignment[1])) {
      return `${assignment[1]}=redacted`
    }
    const flag = part.match(/^(--?[^=\s]+)(?:=(.*))?$/)
    if (flag && SENSITIVE_NAME_RE.test(flag[1])) {
      if (flag[2] !== undefined) return `${flag[1]}=redacted`
      redactNext = true
      return flag[1]
    }
    return redactStartupEvidenceText(part)
  }).join(' ')
}

const TRUST_PROMPT_PATTERNS = [
  /\bdo you trust\b/i,
  /\btrust (?:this|the) (?:workspace|folder|directory|repo|repository|project)\b/i,
  /\b(?:workspace|folder|directory|repo|repository|project) trust\b/i,
  /\buntrusted (?:workspace|folder|directory|repo|repository|project|files?)\b/i,
  /\bnot trusted\b/i,
  /(?:작업공간|워크스페이스|폴더|디렉터리|저장소|프로젝트).{0,40}(?:신뢰|trust)/i,
]

const TOOL_PERMISSION_PROMPT_PATTERNS = [
  /\b(?:allow|approve|confirm|authorize)\b.{0,80}\b(?:command|tool|shell|bash|terminal|exec|execute|run|edit|write|file|network)\b/i,
  /\b(?:command|tool|shell|bash|terminal|exec|execute|run|edit|write|file|network)\b.{0,80}\b(?:requires|needs|waiting for)\b.{0,40}\b(?:permission|approval|confirmation|authorization)\b/i,
  /\b(?:do you want to|would you like to)\b.{0,80}\b(?:proceed|continue|allow|approve|run|execute|edit|write)\b/i,
  /\b(?:approve|deny|reject|yes|no|y\/n)\b.{0,80}\b(?:command|tool|permission|approval|shell|bash|exec|edit|write)\b/i,
  /\bpermission required\b/i,
  /\bapproval required\b/i,
  /\bwaiting for approval\b/i,
  /(?:허용|승인|확인).{0,80}(?:명령|도구|권한|실행|파일|네트워크)/i,
  /(?:명령|도구|권한|실행|파일|네트워크).{0,80}(?:허용|승인|확인)/i,
  /허용할까요\??/i,
  /승인하시겠습니까\??/i,
]

export interface SwarmStartupOutputClassification {
  lifecycleState: Extract<SwarmAgentStartupState, 'trust_required' | 'tool_permission_required'>
  trustPromptDetected: boolean
  toolPermissionPromptDetected: boolean
}

export function classifySwarmStartupOutput(output: string): SwarmStartupOutputClassification | null {
  const text = output.replace(/\s+/g, ' ').trim()
  if (!text) return null
  const trustPromptDetected = TRUST_PROMPT_PATTERNS.some((pattern) => pattern.test(text))
  const toolPermissionPromptDetected = TOOL_PERMISSION_PROMPT_PATTERNS.some((pattern) => pattern.test(text))
  if (!trustPromptDetected && !toolPermissionPromptDetected) return null
  return {
    lifecycleState: trustPromptDetected ? 'trust_required' : 'tool_permission_required',
    trustPromptDetected,
    toolPermissionPromptDetected,
  }
}

function baseStartupEvidence(handle: SwarmAgentHandle, now: number): SwarmAgentStartupEvidence {
  const base = handle.startupEvidence
  const startedAt = base?.startedAt ?? handle.spawnedAt ?? now
  return {
    handle: handle.handle,
    agent: handle.agent,
    runtime: base?.runtime ?? handle.runtime ?? 'unknown',
    lifecycleState: base?.lifecycleState ?? 'ready_for_prompt',
    cwd: base?.cwd ?? handle.cwd,
    paneCommand: base?.paneCommand,
    startedAt,
    readyAt: base?.readyAt,
    promptSentAt: base?.promptSentAt,
    promptAccepted: base?.promptAccepted,
    trustPromptDetected: base?.trustPromptDetected ?? false,
    toolPermissionPromptDetected: base?.toolPermissionPromptDetected ?? false,
    transportHealthy: base?.transportHealthy ?? true,
    elapsedMs: Math.max(0, now - startedAt),
    failureKind: base?.failureKind,
    failureMessage: base?.failureMessage,
    lastOutputPreview: base?.lastOutputPreview,
    recoveryHint: base?.recoveryHint,
  }
}

function previewStartupOutput(output: string): string {
  return redactStartupEvidenceText(output.replace(/\s+/g, ' ').trim()).slice(-500)
}

export function createSwarmRecoveryHint(
  evidence: SwarmAgentStartupEvidence,
): SwarmAgentRecoveryHint | undefined {
  if (evidence.lifecycleState === 'trust_required') {
    return {
      scenario: 'trust_prompt_unresolved',
      steps: ['inspect_worker_output', 'review_trust_prompt', 'escalate_to_human'],
      maxAttempts: 1,
      escalationPolicy: 'alert_human',
      reason: 'worker is waiting for a workspace trust decision',
    }
  }
  if (evidence.lifecycleState === 'tool_permission_required') {
    return {
      scenario: 'tool_permission_unresolved',
      steps: ['inspect_worker_output', 'review_tool_permission', 'escalate_to_human'],
      maxAttempts: 1,
      escalationPolicy: 'alert_human',
      reason: 'worker is waiting for a tool permission decision',
    }
  }
  if (evidence.lifecycleState !== 'failed') return undefined
  if (evidence.failureKind === 'timeout' && evidence.promptSentAt && !evidence.promptAccepted) {
    return {
      scenario: 'prompt_misdelivery',
      steps: ['inspect_worker_output', 'resend_prompt', 'restart_worker', 'escalate_to_human'],
      maxAttempts: 1,
      escalationPolicy: 'alert_human',
      reason: 'prompt was sent but no accepted response state was observed before timeout',
    }
  }
  if (evidence.failureKind === 'configuration' || evidence.failureKind === 'process_spawn') {
    return {
      scenario: 'configuration_failure',
      steps: ['fix_configuration', 'restart_worker', 'escalate_to_human'],
      maxAttempts: 1,
      escalationPolicy: 'alert_human',
      reason: 'worker launch failed before a healthy runtime was available',
    }
  }
  if (evidence.failureKind === 'transport') {
    return {
      scenario: 'transport_failure',
      steps: ['check_transport', 'restart_worker', 'escalate_to_human'],
      maxAttempts: 1,
      escalationPolicy: 'alert_human',
      reason: 'worker transport failed or disconnected',
    }
  }
  if (evidence.failureKind === 'protocol') {
    return {
      scenario: 'protocol_failure',
      steps: ['inspect_protocol_logs', 'restart_worker', 'escalate_to_human'],
      maxAttempts: 1,
      escalationPolicy: 'abort',
      reason: 'worker protocol initialization or session handshake failed',
    }
  }
  if (evidence.failureKind === 'timeout') {
    return {
      scenario: 'provider_failure',
      steps: ['inspect_worker_output', 'restart_worker', 'escalate_to_human'],
      maxAttempts: 1,
      escalationPolicy: 'alert_human',
      reason: 'worker did not become ready before timeout',
    }
  }
  return {
    scenario: 'unknown_failure',
    steps: ['inspect_worker_output', 'restart_worker', 'escalate_to_human'],
    maxAttempts: 1,
    escalationPolicy: 'alert_human',
    reason: 'worker failed without a more specific recovery classification',
  }
}

function withRecoveryHint(evidence: SwarmAgentStartupEvidence): SwarmAgentStartupEvidence {
  const recoveryHint = createSwarmRecoveryHint(evidence)
  return recoveryHint ? { ...evidence, recoveryHint } : { ...evidence, recoveryHint: undefined }
}

function evidenceChanged(
  before: SwarmAgentStartupEvidence | undefined,
  after: SwarmAgentStartupEvidence,
): boolean {
  if (!before) return true
  return before.lifecycleState !== after.lifecycleState
    || before.promptSentAt !== after.promptSentAt
    || before.promptAccepted !== after.promptAccepted
    || before.trustPromptDetected !== after.trustPromptDetected
    || before.toolPermissionPromptDetected !== after.toolPermissionPromptDetected
    || before.transportHealthy !== after.transportHealthy
    || before.failureKind !== after.failureKind
    || before.failureMessage !== after.failureMessage
    || before.lastOutputPreview !== after.lastOutputPreview
    || JSON.stringify(before.recoveryHint) !== JSON.stringify(after.recoveryHint)
}

export function createSwarmPromptSentEvidence(
  handle: SwarmAgentHandle,
  now = Date.now(),
): SwarmAgentStartupEvidence {
  const base = baseStartupEvidence(handle, now)
  return withRecoveryHint({
    ...base,
    lifecycleState: 'running',
    promptSentAt: now,
    promptAccepted: false,
    transportHealthy: true,
    elapsedMs: Math.max(0, now - base.startedAt),
    failureKind: undefined,
    failureMessage: undefined,
    lastOutputPreview: undefined,
  })
}

export function createSwarmPromptAcceptedEvidence(
  handle: SwarmAgentHandle,
  output = '',
  now = Date.now(),
): SwarmAgentStartupEvidence | null {
  const base = baseStartupEvidence(handle, now)
  if (!base.promptSentAt || base.promptAccepted) return null
  const preview = previewStartupOutput(output)
  const evidence: SwarmAgentStartupEvidence = withRecoveryHint({
    ...base,
    lifecycleState: 'running',
    promptAccepted: true,
    transportHealthy: true,
    elapsedMs: Math.max(0, now - base.startedAt),
    failureKind: undefined,
    failureMessage: undefined,
    lastOutputPreview: preview || base.lastOutputPreview,
  })
  return evidenceChanged(handle.startupEvidence, evidence) ? evidence : null
}

export function createSwarmPromptAcceptanceTimeoutEvidence(
  handle: SwarmAgentHandle,
  output: string,
  now = Date.now(),
): SwarmAgentStartupEvidence | null {
  const base = baseStartupEvidence(handle, now)
  if (!base.promptSentAt || base.promptAccepted) return null
  const elapsedSincePrompt = Math.max(0, now - base.promptSentAt)
  const message = `prompt acceptance timeout after ${elapsedSincePrompt} ms`
  const preview = previewStartupOutput(output)
  const evidence: SwarmAgentStartupEvidence = withRecoveryHint({
    ...base,
    lifecycleState: 'failed',
    promptAccepted: false,
    transportHealthy: true,
    elapsedMs: Math.max(0, now - base.startedAt),
    failureKind: 'timeout',
    failureMessage: message,
    lastOutputPreview: preview || message,
  })
  return evidenceChanged(handle.startupEvidence, evidence) ? evidence : null
}

export function createSwarmStartupOutputEvidence(
  handle: SwarmAgentHandle,
  output: string,
  now = Date.now(),
): SwarmAgentStartupEvidence | null {
  const classification = classifySwarmStartupOutput(output)
  if (!classification) return null
  const base = baseStartupEvidence(handle, now)
  const evidence: SwarmAgentStartupEvidence = withRecoveryHint({
    ...base,
    lifecycleState: classification.lifecycleState,
    promptAccepted: base.promptAccepted || Boolean(base.promptSentAt),
    trustPromptDetected: base.trustPromptDetected || classification.trustPromptDetected,
    toolPermissionPromptDetected: base.toolPermissionPromptDetected
      || classification.toolPermissionPromptDetected,
    transportHealthy: true,
    elapsedMs: Math.max(0, now - base.startedAt),
    lastOutputPreview: previewStartupOutput(output),
    failureKind: undefined,
    failureMessage: undefined,
  })
  return evidenceChanged(handle.startupEvidence, evidence) ? evidence : null
}

export function classifySwarmStartupFailure(error: unknown): SwarmAgentStartupFailureKind {
  const message = errorMessage(error)
  if (/timeout|timed out|abort/i.test(message)) return 'timeout'
  if (/ENOENT|not found|command not found|spawn/i.test(message)) return 'process_spawn'
  if (/requires|configured|invalid|unsupported|credentials|url/i.test(message)) return 'configuration'
  if (/jsonrpc|protocol|initialize|session/i.test(message)) return 'protocol'
  if (/transport|socket|ECONN|fetch failed|network/i.test(message)) return 'transport'
  return 'unknown'
}

export function createSwarmStartupFailureEvidence(input: {
  handle: string
  agent: SwarmAgentName
  cwd: string
  runtime?: SwarmAgentRuntimeName
  paneCommand?: string
  startedAt?: number
  error: unknown
}): SwarmAgentStartupEvidence {
  const now = Date.now()
  const startedAt = input.startedAt ?? now
  const message = redactStartupEvidenceText(errorMessage(input.error))
  return withRecoveryHint({
    handle: input.handle,
    agent: input.agent,
    runtime: input.runtime ?? 'unknown',
    lifecycleState: 'failed',
    cwd: input.cwd,
    paneCommand: input.paneCommand ? redactStartupEvidenceText(input.paneCommand) : undefined,
    startedAt,
    promptAccepted: false,
    trustPromptDetected: false,
    toolPermissionPromptDetected: false,
    transportHealthy: false,
    elapsedMs: Math.max(0, now - startedAt),
    failureKind: classifySwarmStartupFailure(input.error),
    failureMessage: message,
    lastOutputPreview: message.slice(0, 500),
  })
}
