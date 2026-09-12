import {
  createTraceRedactionContext,
  redactSensitiveText,
} from '../observability/trace-redaction.js'

const MAX_SCHEDULER_DELIVERY_ERROR_LENGTH = 500

/**
 * Channel failures cross both a user-visible run-result boundary and a
 * durable outbox boundary. Treat connector/remote response text as untrusted:
 * logs are redacted independently, but SQLite and API-visible evidence must
 * never receive the raw exception first.
 */
export function safeSchedulerDeliveryError(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error)
  return redactSensitiveText(raw, createTraceRedactionContext(), {
    maxStringLength: MAX_SCHEDULER_DELIVERY_ERROR_LENGTH,
  })
}
