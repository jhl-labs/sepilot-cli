export interface RoutedEvent {
  id: string
  source: string
  type: string
  subject: string | null
  occurredAt: number
  dedupeKey: string
  payload: Record<string, unknown>
}

export type RoutedEventInput = Omit<RoutedEvent, 'id'> & { id?: string }

export type EventSummary = Pick<RoutedEvent, 'id' | 'source' | 'type' | 'subject' | 'occurredAt'>

export interface RenderedMessage {
  ruleId: string
  title: string
  text: string
  event: EventSummary
}

export type DeliveryOutcome =
  | { ok: true }
  | { ok: false; retryable: boolean; error: string }

export function summarizeEvent(event: RoutedEvent): EventSummary {
  return {
    id: event.id,
    source: event.source,
    type: event.type,
    subject: event.subject,
    occurredAt: event.occurredAt,
  }
}
