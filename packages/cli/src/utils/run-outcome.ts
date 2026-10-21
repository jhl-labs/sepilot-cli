export function isUnsuccessfulAgentResult(result: {
  content?: string
  stopReason?: unknown
} | null | undefined): boolean {
  if (!result) return true
  if (result.stopReason && typeof result.stopReason === 'object'
    && 'kind' in result.stopReason && result.stopReason.kind !== 'completed') return true
  return (result.content ?? '').split(/\r?\n/).some(line => line.trimStart().startsWith('INCOMPLETE:'))
}
