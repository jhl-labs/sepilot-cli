/** Structured daemon outcomes take precedence over success-looking prose. */
export function isUnsuccessfulAgentResult(result: {
  content?: string
  stopReason?: unknown
} | null | undefined): boolean {
  if (!result) return true
  if (result.stopReason && typeof result.stopReason === 'object'
    && 'kind' in result.stopReason && result.stopReason.kind !== 'completed') return true
  // Compatibility with older daemons that did not send structured stop reasons.
  return (result.content ?? '').split(/\r?\n/).some(line => line.trimStart().startsWith('INCOMPLETE:'))
}
