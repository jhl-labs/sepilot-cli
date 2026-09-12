import type {
  AssistantRuntimeStatusOptions,
  buildAssistantRuntimeStatus,
} from '../server/routes/system.js'
import type {
  ToolDefinitionRuntime,
  ToolExecutionContext,
  ToolResult,
} from './registry.js'

type AssistantRuntimeStatusSnapshot = Awaited<ReturnType<typeof buildAssistantRuntimeStatus>>

export interface AssistantStatusToolDeps {
  snapshot(options: AssistantRuntimeStatusOptions): Promise<AssistantRuntimeStatusSnapshot>
}

const DEFAULT_SECTIONS = [
  'integrations',
  'channels',
  'scheduler',
  'notifications',
  'capabilities',
] as const

const VALID_SECTIONS = new Set([
  ...DEFAULT_SECTIONS,
  'interactions',
  'all',
])

function requestedSections(input: Record<string, unknown>): Set<string> {
  if (!Array.isArray(input.sections) || input.sections.length === 0) {
    return new Set(DEFAULT_SECTIONS)
  }
  const sections = input.sections
    .filter((section): section is string => (
      typeof section === 'string' && VALID_SECTIONS.has(section)
    ))
  return sections.length > 0 ? new Set(sections) : new Set(DEFAULT_SECTIONS)
}

function wants(sections: ReadonlySet<string>, section: string): boolean {
  return sections.has('all') || sections.has(section)
}

function requestedChannelTypes(input: Record<string, unknown>): string[] {
  if (!Array.isArray(input.channelTypes)) return []
  return [...new Set(input.channelTypes
    .filter((value): value is string => typeof value === 'string')
    .map((value) => value.trim().toLowerCase())
    .filter((value) => /^[a-z0-9][a-z0-9._-]{0,63}$/.test(value)))]
    .slice(0, 20)
}

function assistantStatusObservationCovers(
  observed: Record<string, unknown>,
  requested: Record<string, unknown>,
): boolean {
  const observedSections = requestedSections(observed)
  const requestedSectionSet = requestedSections(requested)
  if ([...requestedSectionSet].some((section) => !wants(observedSections, section))) {
    return false
  }
  if (requested.includeRecentRuns === true && observed.includeRecentRuns !== true) {
    return false
  }
  const observedChannels = requestedChannelTypes(observed)
  const requestedChannels = requestedChannelTypes(requested)
  if (requestedChannels.length === 0) return observedChannels.length === 0
  const observedSet = new Set(observedChannels)
  return requestedChannels.every((channel) => observedSet.has(channel))
}

function projectChannels(
  snapshot: AssistantRuntimeStatusSnapshot,
  requestedTypes: readonly string[],
) {
  if (requestedTypes.length === 0) return snapshot.operations.channels
  const byType = new Map(snapshot.operations.channels.map((channel) => [channel.type, channel]))
  return requestedTypes.map((type) => byType.get(type) ?? {
    type,
    configured: false,
    configuredCount: 0,
    enabled: false,
    enabledCount: 0,
    activeCount: 0,
    status: 'not_configured' as const,
  })
}

function projectNotifyRelay(snapshot: AssistantRuntimeStatusSnapshot) {
  const relay = snapshot.integrations.notifyRelay
  const latest = relay.latestDelivery
  return {
    configured: relay.configured,
    deliveryMode: relay.deliveryMode,
    latestDelivery: latest ? {
      status: latest.status,
      httpStatus: latest.httpStatus,
      attemptedAt: latest.attemptedAt,
      completedAt: latest.completedAt,
      providerDelivery: latest.providerDelivery ? {
        lookupStatus: latest.providerDelivery.lookupStatus,
        status: latest.providerDelivery.status,
        httpStatus: latest.providerDelivery.httpStatus,
        attempt: latest.providerDelivery.attempt,
        checkedAt: latest.providerDelivery.checkedAt,
        completedAt: latest.providerDelivery.completedAt,
      } : null,
    } : null,
    outbox: relay.outbox,
    providerDelivery: relay.providerDelivery,
  }
}

export function createAssistantStatusTool(
  deps: AssistantStatusToolDeps,
): ToolDefinitionRuntime {
  return {
    name: 'assistant.status',
    description:
      'Read the daemon-owned, non-secret operational readiness snapshot for personal-assistant integrations and delivery. Use for Notify Relay configuration/outbox/provider status, configured channel types such as Mattermost and their live connection state, scheduler job health/counts, notification inventory, pending interaction counts, and required skill/tool readiness. This is the canonical alternative to searching source/config files or calling the daemon HTTP API through terminal/web tools. It never sends a notification, changes a schedule, or returns credentials, destinations, message bodies, job instructions, or channel allowlists.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'assistant-status' },
    observationCoverage: {
      covers: (observed, requested) => assistantStatusObservationCovers(observed, requested),
    },
    inputSchema: {
      type: 'object',
      properties: {
        sections: {
          type: 'array',
          maxItems: 7,
          uniqueItems: true,
          items: {
            type: 'string',
            enum: [
              'all',
              'integrations',
              'channels',
              'scheduler',
              'notifications',
              'capabilities',
              'interactions',
            ],
          },
          description:
            'Optional sections. Defaults to integrations, channels, scheduler, notifications, and capabilities.',
        },
        channelTypes: {
          type: 'array',
          maxItems: 20,
          uniqueItems: true,
          items: { type: 'string', maxLength: 64 },
          description:
            'Optional exact channel types to report, for example ["mattermost"]. Missing requested types are returned explicitly as not_configured.',
        },
        includeRecentRuns: {
          type: 'boolean',
          default: false,
          description:
            'Include bounded recent scheduler run correlations. Defaults to false because job counts and delivery-outbox health are the operational summary; use schedule_runs for one job history.',
        },
      },
      additionalProperties: false,
    },
    async execute(
      input: Record<string, unknown>,
      context?: ToolExecutionContext,
    ): Promise<ToolResult> {
      const startedAt = Date.now()
      try {
        const sections = requestedSections(input)
        const snapshot = await deps.snapshot({
          surface: context?.channelContext?.channel ?? null,
          includeInteractions: wants(sections, 'interactions'),
          includeJpadPublications: false,
        })
        const relay = projectNotifyRelay(snapshot)
        const output = {
          generatedAt: snapshot.generatedAt,
          ...(wants(sections, 'integrations') ? {
            integrations: {
              notifyRelay: relay,
              jpad: snapshot.integrations.jpad,
              aiSearch: snapshot.integrations.aiSearch,
            },
          } : {}),
          ...(wants(sections, 'channels') ? {
            channels: projectChannels(snapshot, requestedChannelTypes(input)),
          } : {}),
          ...(wants(sections, 'scheduler') ? {
            scheduler: {
              runtime: snapshot.integrations.scheduler,
              jobs: snapshot.operations.schedulerJobs,
              deliveryOutbox: snapshot.operations.schedulerDeliveryOutbox,
              ...(input.includeRecentRuns === true
                ? { recentRuns: snapshot.operations.schedulerRuns }
                : {}),
            },
          } : {}),
          ...(wants(sections, 'notifications') ? {
            notifications: snapshot.operations.notifications,
          } : {}),
          ...(wants(sections, 'capabilities') ? {
            capabilities: {
              skills: snapshot.skills,
              tools: snapshot.tools,
            },
          } : {}),
          ...(wants(sections, 'interactions') ? {
            interactions: snapshot.operations.interactions ?? null,
          } : {}),
        }
        return {
          // Operational snapshots are machine evidence, not presentation.
          // Keep them on one bounded line so the agent's line-oriented
          // observation compactor cannot drop a requested middle section.
          output: JSON.stringify(output),
          status: 'success',
          durationMs: Date.now() - startedAt,
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return {
          output: `Assistant runtime status is unavailable: ${message}`,
          status: 'error',
          code: 'ASSISTANT_STATUS_UNAVAILABLE_TRANSIENT',
          durationMs: Date.now() - startedAt,
        }
      }
    },
  }
}
