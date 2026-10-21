import {
  describeStopReason,
  escalationTarget,
  type RunStopLocale,
  type RunStopNextAction,
  type RunStopReason,
} from '@sepilotd/api-client'

export type RunStopHintMode = 'shell' | 'cli'

export function runStopActionHint(
  kind: RunStopNextAction,
  reason: RunStopReason,
  mode: RunStopHintMode = 'shell',
): string {
  const requestId = reason.detail?.requestId
  switch (kind) {
    case 'resume':
      return mode === 'shell' ? '/resume' : 'sepilot chat --resume'
    case 'switch_autonomy':
      return mode === 'shell' ? '/autonomy <level>' : 'sepilot config set agent.autonomy <level>'
    case 'approve_pending':
      if (requestId) return `sepilot approve ${requestId}`
      return mode === 'shell' ? '/approvals' : 'sepilot approvals'
    case 'raise_budget':
      return mode === 'shell'
        ? '/resume (or re-run with a larger maxIterations)'
        : 're-run with a larger maxIterations'
    case 'retry':
      return mode === 'shell' ? '/edit-last' : 'run the same prompt again'
    case 'escalate': {
      const target = escalationTarget(reason) ?? 'deep-coder'
      const bot = reason.detail?.escalateBot
      if (mode === 'shell') {
        return bot ? `/bot ${bot} then /edit-last` : `/mode ${target} then /edit-last`
      }
      return bot
        ? `sepilot ask --bot ${bot} "<same prompt>"`
        : `sepilot ask --mode ${target} "<same prompt>"`
    }
    default:
      return kind
  }
}

export function formatRunStopCard(
  reason: RunStopReason,
  options: { locale?: RunStopLocale; hintMode?: RunStopHintMode } = {},
): string {
  const copy = describeStopReason(reason, options.locale ?? 'en')
  const lines = [`[run stopped] ${copy.title}`, copy.body]
  for (const action of copy.actions) {
    lines.push(`  ${action.label}: ${runStopActionHint(action.kind, reason, options.hintMode)}`)
  }
  return lines.join('\n')
}
