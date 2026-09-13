import {
  describeStopReason,
  type RunStopLocale,
  type RunStopNextAction,
  type RunStopReason,
} from '@sepilotd/api-client'

export type RunStopHintMode = 'shell' | 'cli'

/**
 * Command hint for a next action. `shell` spells chat-shell slash commands;
 * `cli` spells top-level `sepilot` verbs for piped/one-shot runs where no
 * slash prompt exists.
 */
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
    default:
      return kind
  }
}

/**
 * Compact stop card for terminal surfaces: title, one-line body, and the
 * next actions as command hints. Plain text (no ANSI) so it can be stored
 * as a system message and printed by any renderer.
 */
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
