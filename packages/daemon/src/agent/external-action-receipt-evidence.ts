export interface ExternalActionReceiptEvidenceInput {
  status: unknown
  output?: string
  securityEffect?: unknown
  executionObserved?: boolean
}

/**
 * A successful external-write result is authoritative only for the executor
 * boundary: the registered tool ran and returned a non-empty success receipt.
 * It does not prove that a downstream provider later delivered or processed
 * the action. Callers must preserve that distinction in user-facing reports.
 */
export function isExecutorConfirmedExternalActionReceipt(
  input: ExternalActionReceiptEvidenceInput,
): boolean {
  return input.status === 'success'
    && input.executionObserved === true
    && input.securityEffect === 'external-write'
    && Boolean(input.output?.trim())
}
