import { createHash, randomUUID } from 'node:crypto'
import {
  externalNotificationRelayConfigured,
  publishStoredNotification,
} from '../notifications/publish.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

const MAX_TITLE_LENGTH = 200
const MAX_TEXT_LENGTH = 8_000

function failure(output: string, start: number, code: string): ToolResult {
  return { output, status: 'error', durationMs: Date.now() - start, code }
}

function requiredText(value: unknown, name: string, max: number): string {
  const text = typeof value === 'string' ? value.trim() : ''
  if (!text) throw new Error(`${name} is required.`)
  if (text.length > max) throw new Error(`${name} must be at most ${max} characters.`)
  return text
}

function notificationId(scope: string, dedupKey: unknown): string {
  if (typeof dedupKey !== 'string' || !dedupKey.trim()) {
    return `agent-notification:${randomUUID()}`
  }
  if (dedupKey.length > 240) throw new Error('dedupKey must be at most 240 characters.')
  const digest = createHash('sha256')
    .update(`${scope}\0${dedupKey.trim()}`)
    .digest('hex')
    .slice(0, 32)
  return `agent-notification:${digest}`
}

function notificationKind(value: unknown): 'alert' | 'incident' | 'report' | 'custom' {
  if (value === 'alert' || value === 'incident' || value === 'report' || value === 'custom') {
    return value
  }
  throw new Error('kind must be alert, incident, report, or custom.')
}

function notificationPriority(value: unknown): 'low' | 'normal' | 'high' | 'critical' {
  if (value === 'low' || value === 'normal' || value === 'high' || value === 'critical') {
    return value
  }
  throw new Error('priority must be low, normal, high, or critical.')
}

/**
 * Publish an explicit agent notification through the same persisted,
 * audience-gated notification path used by scheduler events. The tool never
 * receives relay credentials or destination ids; those remain runtime-only.
 */
export function createNotificationPublishTool(): ToolDefinitionRuntime {
  return {
    name: 'notification.publish',
    description:
      'Publish a user-authorized notification to Sepilot and the configured external notification relay. ' +
      'Use for actionable alerts, incident/recovery notices, or requested reports. Set confirmSend=true only ' +
      'when the user explicitly requested this notification or authorized the enclosing unattended schedule. ' +
      'Use a stable dedupKey for retries of the same event; never include credentials or secret values.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        kind: {
          type: 'string',
          enum: ['alert', 'incident', 'report', 'custom'],
          description: 'Notification domain type.',
        },
        priority: {
          type: 'string',
          enum: ['low', 'normal', 'high', 'critical'],
          description: 'Delivery priority; reserve high/critical for action-worthy events.',
        },
        title: { type: 'string', minLength: 1, maxLength: MAX_TITLE_LENGTH },
        text: { type: 'string', minLength: 1, maxLength: MAX_TEXT_LENGTH },
        dedupKey: {
          type: 'string',
          maxLength: 240,
          description: 'Stable non-secret event identity used to deduplicate retries.',
        },
        confirmSend: {
          type: 'boolean',
          description: 'True only after explicit user authorization for this notification workflow.',
        },
      },
      required: ['kind', 'priority', 'title', 'text', 'confirmSend'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      if (input.confirmSend !== true) {
        return failure(
          'Notification delivery requires confirmSend=true after explicit user authorization.',
          start,
          'NOTIFICATION_CONFIRMATION_REQUIRED_USER',
        )
      }

      try {
        const kind = notificationKind(input.kind)
        const priority = notificationPriority(input.priority)
        const scope = context?.sessionId
          ?? context?.channelContext?.chatKey
          ?? 'global'
        const item = publishStoredNotification({
          id: notificationId(scope, input.dedupKey),
          title: requiredText(input.title, 'title', MAX_TITLE_LENGTH),
          body: requiredText(input.text, 'text', MAX_TEXT_LENGTH),
          topic: `agent-notification:${kind}:${priority}`,
          audience: null,
        })
        return {
          output: JSON.stringify({
            id: item.id,
            stored: true,
            externalRelayConfigured: externalNotificationRelayConfigured(),
            delivery: externalNotificationRelayConfigured() ? 'queued_durable' : 'local_only',
          }),
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (error) {
        return failure(
          error instanceof Error ? error.message : String(error),
          start,
          'INVALID_INPUT_PERMANENT',
        )
      }
    },
  }
}
