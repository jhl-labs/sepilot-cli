import type { ToolDefinitionRuntime, ToolExecutionContext, ToolResult } from '../tools/registry.js'
import type { EventDestination, EventRoutingConfig } from './config-schema.js'
import type { DeliveryInput, EventDeliveryOutbox } from './delivery-outbox.js'

export interface EventRoutingAgentToolDeps {
  getConfig: () => EventRoutingConfig
  outbox: Pick<EventDeliveryOutbox, 'enqueueDetailed'>
  now?: () => number
  scheduleDrain?: () => void
}

export const EVENT_DESTINATIONS_TOOL = 'event.destinations'
export const EVENT_DELIVER_TOOL = 'event.deliver'
export const AGENT_TOOL_RULE_ID = 'agent-tool'

const DESTINATION_ID_PATTERN = '^[a-z0-9][a-z0-9-]{0,63}$'
const DESTINATION_ID_REGEX = new RegExp(DESTINATION_ID_PATTERN)
const MAX_TEXT_LENGTH = 8_000
const MAX_TITLE_LENGTH = 200
const MAX_DEDUP_KEY_LENGTH = 240
const DEFAULT_TITLE = 'sepilot'

class InvalidInput extends Error {}

function failure(output: string, start: number, code: string): ToolResult {
  return { output, status: 'error', durationMs: Date.now() - start, code }
}

function success(payload: unknown, start: number): ToolResult {
  return { output: JSON.stringify(payload), status: 'success', durationMs: Date.now() - start }
}

function destinationView(destination: EventDestination) {
  return {
    id: destination.id,
    kind: destination.kind,
    ...(destination.kind === 'messenger-webhook' ? { preset: destination.preset } : {}),
    enabled: destination.enabled,
  }
}

export function createEventDestinationsTool(deps: EventRoutingAgentToolDeps): ToolDefinitionRuntime {
  return {
    name: EVENT_DESTINATIONS_TOOL,
    description:
      'List configured event-routing destinations by id, kind and enabled state, plus whether event routing is enabled. ' +
      'Use the id with event.deliver. Never returns destination URLs or credentials.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    async execute(): Promise<ToolResult> {
      const start = Date.now()
      const config = deps.getConfig()
      return success({ destinations: config.destinations.map(destinationView), enabled: config.enabled }, start)
    },
  }
}

interface DeliverRequest {
  destination: string
  text: string
  title?: string
  dedupKey?: string
}

function optionalText(value: unknown, name: string, max: number): string | undefined {
  if (value === undefined || value === null) return undefined
  if (typeof value !== 'string') throw new InvalidInput(`${name} must be a string.`)
  const text = value.trim()
  if (text.length > max) throw new InvalidInput(`${name} must be at most ${max} characters.`)
  return text || undefined
}

function parseDeliverRequest(input: Record<string, unknown>): DeliverRequest {
  const destination = input.destination
  if (typeof destination !== 'string' || !DESTINATION_ID_REGEX.test(destination)) {
    throw new InvalidInput('destination must be a destination id from event.destinations, not a URL.')
  }
  const text = optionalText(input.text, 'text', MAX_TEXT_LENGTH)
  if (!text) throw new InvalidInput('text is required.')
  return {
    destination,
    text,
    title: optionalText(input.title, 'title', MAX_TITLE_LENGTH),
    dedupKey: optionalText(input.dedupKey, 'dedupKey', MAX_DEDUP_KEY_LENGTH),
  }
}

function dedupeScope(context: ToolExecutionContext | undefined): string {
  return context?.sessionId ?? context?.channelContext?.chatKey ?? 'global'
}

export function createEventDeliverTool(deps: EventRoutingAgentToolDeps): ToolDefinitionRuntime {
  const now = deps.now ?? Date.now
  return {
    name: EVENT_DELIVER_TOOL,
    description:
      'Queue a message for durable delivery to a configured event-routing destination (by id from event.destinations). ' +
      'Set confirmSend=true only when the user explicitly requested this delivery or authorized the enclosing unattended ' +
      'schedule; approval is still required and enforced separately. ' +
      'Use a stable dedupKey so retries of the same message are not delivered twice; never include credentials or secret values.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        destination: {
          type: 'string',
          pattern: DESTINATION_ID_PATTERN,
          description: 'Destination id from event.destinations. URLs are rejected.',
        },
        text: { type: 'string', minLength: 1, maxLength: MAX_TEXT_LENGTH },
        title: { type: 'string', maxLength: MAX_TITLE_LENGTH },
        dedupKey: {
          type: 'string',
          maxLength: MAX_DEDUP_KEY_LENGTH,
          description: 'Stable non-secret identity used to deduplicate retries of the same delivery.',
        },
        confirmSend: {
          type: 'boolean',
          description: 'True only after explicit user authorization for this delivery.',
        },
      },
      required: ['destination', 'text', 'confirmSend'],
      additionalProperties: false,
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      if (input.confirmSend !== true) {
        return failure(
          'Event delivery requires confirmSend=true after explicit user authorization.',
          start,
          'EVENT_DELIVER_CONFIRMATION_REQUIRED_USER',
        )
      }
      let request: DeliverRequest
      try {
        request = parseDeliverRequest(input)
      } catch (error) {
        if (!(error instanceof InvalidInput)) throw error
        return failure(error.message, start, 'INVALID_INPUT_PERMANENT')
      }

      const config = deps.getConfig()
      if (!config.enabled) {
        return failure('Event routing is disabled; nothing was queued.', start, 'EVENT_ROUTING_DISABLED')
      }
      const destination = config.destinations.find((candidate) => candidate.id === request.destination)
      if (!destination) {
        return failure('Unknown destination. Call event.destinations for valid ids.', start, 'EVENT_DESTINATION_UNKNOWN')
      }
      if (!destination.enabled) {
        return failure(`Destination '${request.destination}' is disabled.`, start, 'EVENT_DESTINATION_DISABLED')
      }

      const executionId = context?.executionId ?? `anonymous-${start}`
      const eventId = `agent:${executionId}`
      const occurredAt = now()
      const delivery: DeliveryInput = {
        eventId,
        ruleId: AGENT_TOOL_RULE_ID,
        destinationId: destination.id,
        title: request.title ?? DEFAULT_TITLE,
        text: request.text,
        event: { id: eventId, source: 'agent', type: 'deliver', subject: request.title ?? null, occurredAt },
        idempotencyKey: [AGENT_TOOL_RULE_ID, dedupeScope(context), request.dedupKey ?? executionId, destination.id]
          .map(encodeURIComponent)
          .join(':'),
      }
      const { record, duplicate } = deps.outbox.enqueueDetailed(delivery, occurredAt)
      if (!duplicate) {
        deps.scheduleDrain?.()
        return success({ deliveryId: record.id, destination: destination.id, status: 'queued' }, start)
      }
      return success(
        { deliveryId: record.id, destination: destination.id, status: 'already-queued', existingStatus: record.status },
        start,
      )
    },
  }
}
