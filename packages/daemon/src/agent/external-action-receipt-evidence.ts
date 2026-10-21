export interface ExternalActionReceiptEvidenceInput {
  status: unknown
  output?: string
  securityEffect?: unknown
  executionObserved?: boolean
}

export function isExecutorConfirmedExternalActionReceipt(
  input: ExternalActionReceiptEvidenceInput,
): boolean {
  return input.status === 'success'
    && input.executionObserved === true
    && input.securityEffect === 'external-write'
    && Boolean(input.output?.trim())
}
