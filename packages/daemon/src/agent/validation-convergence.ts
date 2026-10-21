export const MAX_POST_VALIDATION_READ_ONLY_TURNS = 2

export interface ValidationConvergenceEntry {
  status: 'success' | 'error'
  executionObserved: boolean
  securityEffect: string
  actionPurpose?: unknown
  reliableSuccess?: boolean
}

export interface ValidationConvergenceState {
  mutationPending: boolean
  validationArmed: boolean
  readOnlyTurns: number
}

export interface ValidationConvergenceTransition {
  state: ValidationConvergenceState
  forceFinal: boolean
}

export function initialValidationConvergenceState(): ValidationConvergenceState {
  return { mutationPending: false, validationArmed: false, readOnlyTurns: 0 }
}

export function advanceValidationConvergence(
  previous: ValidationConvergenceState,
  entries: readonly ValidationConvergenceEntry[],
): ValidationConvergenceTransition {
  const observed = entries.filter((entry) => entry.executionObserved)
  if (observed.length === 0) return { state: previous, forceFinal: false }

  const successfulMutation = observed.some((entry) => (
    entry.status === 'success'
    && entry.actionPurpose !== 'validate'
    && entry.actionPurpose !== 'observe'
    && entry.securityEffect !== 'observe'
  ))
  const validationEntries = observed.filter((entry) => entry.actionPurpose === 'validate')
  const successfulValidation = validationEntries.some((entry) => entry.status === 'success' && entry.reliableSuccess !== false)
  const failedValidation = validationEntries.some((entry) => entry.status !== 'success' || entry.reliableSuccess === false)
  const onlySuccessfulObservations = entries.every((entry) => (
    entry.executionObserved
    && entry.status === 'success'
    && (entry.securityEffect === 'observe' || entry.actionPurpose === 'validate')
  ))
  const cleanBatch = entries.every((entry) => (
    entry.executionObserved
    && entry.status === 'success'
  ))

  if (successfulMutation) {
    return {
      state: {
        mutationPending: true,
        validationArmed: false,
        readOnlyTurns: 0,
      },
      forceFinal: false,
    }
  }

  if (failedValidation || !cleanBatch) {
    return {
      state: {
        ...previous,
        validationArmed: false,
        readOnlyTurns: 0,
      },
      forceFinal: false,
    }
  }

  if (successfulValidation && !previous.validationArmed) {
    return previous.mutationPending
      ? {
          state: { ...previous, validationArmed: true, readOnlyTurns: 0 },
          forceFinal: false,
        }
      : { state: previous, forceFinal: false }
  }

  if (!previous.validationArmed) return { state: previous, forceFinal: false }
  if (!onlySuccessfulObservations) {
    return {
      state: { ...previous, validationArmed: false, readOnlyTurns: 0 },
      forceFinal: false,
    }
  }

  const readOnlyTurns = previous.readOnlyTurns + 1
  const forceFinal = readOnlyTurns >= MAX_POST_VALIDATION_READ_ONLY_TURNS
  return {
    state: {
      ...previous,
      validationArmed: !forceFinal,
      readOnlyTurns,
    },
    forceFinal,
  }
}
