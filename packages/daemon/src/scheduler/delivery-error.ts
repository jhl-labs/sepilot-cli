import {
  createTraceRedactionContext,
  redactSensitiveText,
} from '../observability/trace-redaction.js'

const MAX_SCHEDULER_DELIVERY_ERROR_LENGTH = 500

export function safeSchedulerDeliveryError(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error)
  return redactSensitiveText(raw, createTraceRedactionContext(), {
    maxStringLength: MAX_SCHEDULER_DELIVERY_ERROR_LENGTH,
  })
}
