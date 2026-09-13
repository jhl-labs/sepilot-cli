import { createHash } from 'node:crypto'

/** Bounded, serializable run evidence; never reset on a phase transition. */
export interface WorkProgress {
  revision: number
  seen: string[]
}

export interface ProgressObservation {
  tool: string
  input?: Record<string, unknown>
  output?: string
  outputFingerprint?: string
  status: string
  executionObserved?: boolean
  blocked?: boolean
  securityEffect?: string
}

function stable(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stable).join(',')}]`
  if (value && typeof value === 'object') {
    return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
      .map(([key, item]) => `${JSON.stringify(key)}:${stable(item)}`).join(',')}}`
  }
  return JSON.stringify(value) ?? 'null'
}

/** New failures are useful evidence too; retries, prose and todos are not. */
export function observeWorkProgress(
  current: WorkProgress | undefined,
  observation: ProgressObservation,
): WorkProgress {
  const result = current ?? { revision: 0, seen: [] }
  if (observation.blocked || observation.executionObserved !== true
    || observation.securityEffect === 'internal-state') return result
  const input = { ...observation.input }
  // An annotation or a longer deadline does not make the same result new.
  delete input.actionPurpose
  delete input.timeoutMs
  const output = observation.outputFingerprint
    ?? createHash('sha256').update(observation.output ?? '').digest('hex')
  const fingerprint = createHash('sha256').update(stable({
    tool: observation.tool, input, output, status: observation.status,
  })).digest('hex')
  if (result.seen.includes(fingerprint)) return result
  return {
    revision: result.revision + 1,
    // A memory bound, not a task deadline; global iteration/recovery budgets
    // still bound pathological streams of unique but unhelpful observations.
    seen: [...result.seen, fingerprint].slice(-4096),
  }
}
