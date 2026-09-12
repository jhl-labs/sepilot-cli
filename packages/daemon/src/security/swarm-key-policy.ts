import type { AuditEvent } from '@sepilotd/core'

export interface SwarmKeyForwardingInput {
  runId: string
  handle: string
  keys?: string
  keyName?: string
  enter?: boolean
  resize?: { cols: number; rows: number }
  principal?: string
  device?: string
  env?: NodeJS.ProcessEnv
}

export interface SwarmKeyForwardingDecision {
  allowed: boolean
  reason?: string
  auditEvent: AuditEvent
}

const DANGEROUS_TEXT_PATTERNS: Array<{ reason: string; pattern: RegExp }> = [
  {
    reason: 'dangerous rm -rf root pattern',
    pattern: /\brm\s+-rf\s+\/(?:\s|$|\*)/i,
  },
  { reason: 'fork bomb pattern', pattern: /:\(\)\s*\{\s*:?\|:&\s*};:/i },
  { reason: 'mkfs pattern', pattern: /\bmkfs(?:\.[\w-]+)?\b/i },
  { reason: 'raw disk write pattern', pattern: /\bdd\b[\s\S]*\bof=\/dev\//i },
  { reason: 'root chmod pattern', pattern: /\bchmod\s+-R\s+777\s+\//i },
  {
    reason: 'download pipe to shell pattern',
    pattern:
      /\b(?:curl|wget|fetch)\b[\s\S]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|dash|ksh|fish|python3?|perl|ruby|node)\b/i,
  },
]

function denyEnabled(env: NodeJS.ProcessEnv | undefined): boolean {
  return env?.SEPILOTD_SWARM_KEY_DENY === '1'
}

function classifyPayload(input: SwarmKeyForwardingInput): string {
  if (input.resize) return 'resize'
  if (typeof input.keyName === 'string') return 'named_key'
  if (typeof input.keys === 'string') return 'text'
  return 'empty'
}

function dangerousTextReason(keys: string): string | undefined {
  for (const { reason, pattern } of DANGEROUS_TEXT_PATTERNS) {
    if (pattern.test(keys)) return reason
  }
  return undefined
}

export function evaluateSwarmKeyForwarding(
  input: SwarmKeyForwardingInput,
): SwarmKeyForwardingDecision {
  const payloadType = classifyPayload(input)
  const reason =
    payloadType === 'text' && denyEnabled(input.env)
      ? dangerousTextReason(input.keys ?? '')
      : undefined
  const allowed = !reason
  const auditEvent: AuditEvent = {
    timestamp: new Date().toISOString(),
    event: 'swarm_key_forward',
    device: input.device ?? 'daemon',
    session: input.runId,
    runId: input.runId,
    handle: input.handle,
    principal: input.principal ?? 'unknown',
    payloadType,
    decision: allowed ? 'allowed' : 'denied',
  }
  if (typeof input.keys === 'string') {
    auditEvent.keysLength = input.keys.length
    auditEvent.enter = Boolean(input.enter)
  }
  if (typeof input.keyName === 'string') {
    auditEvent.keyName = input.keyName
  }
  if (input.resize) {
    auditEvent.resize = { ...input.resize }
  }
  if (reason) {
    auditEvent.reason = reason
  }
  return { allowed, reason, auditEvent }
}
