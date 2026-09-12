import {
  ServiceSupervisor,
  type ServiceContainerPort,
  type ServiceContainerVolume,
  type ServiceHealthCheck,
  type RestartPolicyMode,
  type ServiceBackend,
  type ServiceSpec,
} from '../service-supervisor/supervisor.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

function ok(start: number, payload: unknown): ToolResult {
  return {
    output: JSON.stringify(payload, null, 2),
    status: 'success',
    durationMs: Date.now() - start,
  }
}

function fail(start: number, error: unknown): ToolResult {
  return {
    output: error instanceof Error ? error.message : String(error),
    status: 'error',
    durationMs: Date.now() - start,
  }
}

function stringValue(input: Record<string, unknown>, key: string): string {
  const value = input[key]
  return typeof value === 'string' ? value.trim() : ''
}

function numberValue(input: Record<string, unknown>, key: string): number | undefined {
  const value = input[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function booleanValue(input: Record<string, unknown>, key: string): boolean {
  return input[key] === true
}

function stringArrayValue(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined
  const items = value.filter((item): item is string => typeof item === 'string')
  return items.length > 0 ? items : undefined
}

function parseContainerPorts(value: unknown): ServiceContainerPort[] | undefined {
  if (!Array.isArray(value)) return undefined
  const ports = value.flatMap((item): ServiceContainerPort[] => {
    if (!item || typeof item !== 'object') return []
    const input = item as Record<string, unknown>
    const containerPort = numberValue(input, 'containerPort')
    if (containerPort === undefined) return []
    const hostPort = numberValue(input, 'hostPort')
    return [{
      containerPort,
      ...(hostPort !== undefined ? { hostPort } : {}),
      ...(stringValue(input, 'protocol') === 'udp' ? { protocol: 'udp' as const } : {}),
    }]
  })
  return ports.length > 0 ? ports : undefined
}

function parseContainerVolumes(value: unknown): ServiceContainerVolume[] | undefined {
  if (!Array.isArray(value)) return undefined
  const volumes = value.flatMap((item): ServiceContainerVolume[] => {
    if (!item || typeof item !== 'object') return []
    const input = item as Record<string, unknown>
    const source = stringValue(input, 'source')
    const target = stringValue(input, 'target')
    if (!source || !target) return []
    return [{
      source,
      target,
      ...(booleanValue(input, 'readonly') ? { readonly: true } : {}),
    }]
  })
  return volumes.length > 0 ? volumes : undefined
}

function parseStartSpec(input: Record<string, unknown>): ServiceSpec {
  const restartInput = input.restart && typeof input.restart === 'object'
    ? input.restart as Record<string, unknown>
    : undefined
  const healthInput = input.health && typeof input.health === 'object'
    ? input.health as Record<string, unknown>
    : undefined
  return {
    id: stringValue(input, 'id') || undefined,
    name: stringValue(input, 'name') || undefined,
    backend: stringValue(input, 'backend') as ServiceBackend || undefined,
    executable: stringValue(input, 'executable') || undefined,
    args: stringArrayValue(input.args),
    cwd: stringValue(input, 'cwd') || undefined,
    env: input.env && typeof input.env === 'object'
      ? Object.fromEntries(
        Object.entries(input.env).filter((entry): entry is [string, string] => typeof entry[1] === 'string'),
      )
      : undefined,
    image: stringValue(input, 'image') || undefined,
    containerRuntime: stringValue(input, 'containerRuntime') || undefined,
    containerName: stringValue(input, 'containerName') || undefined,
    command: stringArrayValue(input.command),
    ports: parseContainerPorts(input.ports),
    volumes: parseContainerVolumes(input.volumes),
    restart: restartInput
      ? {
          mode: stringValue(restartInput, 'mode') as RestartPolicyMode,
          maxRestarts: numberValue(restartInput, 'maxRestarts'),
          backoffMs: numberValue(restartInput, 'backoffMs'),
        }
      : undefined,
    health: parseHealthCheck(healthInput),
  }
}

function parseHealthCheck(input: Record<string, unknown> | undefined): ServiceHealthCheck | undefined {
  if (!input) return undefined
  const type = stringValue(input, 'type')
  const base = {
    intervalMs: numberValue(input, 'intervalMs'),
    timeoutMs: numberValue(input, 'timeoutMs'),
    graceMs: numberValue(input, 'graceMs'),
  }
  if (type === 'process') {
    return { type, ...base }
  }
  if (type === 'http') {
    return {
      type,
      url: stringValue(input, 'url'),
      expectedStatus: numberValue(input, 'expectedStatus'),
      ...base,
    }
  }
  if (type === 'tcp') {
    return {
      type,
      host: stringValue(input, 'host'),
      port: numberValue(input, 'port') ?? 0,
      ...base,
    }
  }
  return undefined
}

export function createServiceSupervisorTools(
  supervisor = new ServiceSupervisor(),
): ToolDefinitionRuntime[] {
  return [
    {
      name: 'service.start',
      description: 'Start a durable local process or container service and persist its supervision record and logs across daemon restarts. Use this instead of process.start when the user explicitly needs host/LAN exposure (for example a real 0.0.0.0 listener), restart survival, or a named long-lived service. The returned stable id is used by service.status, service.logs, service.stop, and later service.list rediscovery.',
      resumeSafety: 'replay-risky',
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Stable service id. Defaults to a generated UUID.' },
          name: { type: 'string', description: 'Human-readable service name.' },
          backend: { type: 'string', enum: ['process', 'container'], description: 'Service backend. Defaults to process.' },
          executable: { type: 'string', description: 'Executable to run for backend=process.' },
          args: { type: 'array', items: { type: 'string' }, description: 'Command arguments.' },
          cwd: { type: 'string', description: 'Working directory for the service.' },
          env: {
            type: 'object',
            description: 'Environment variable overrides. Values are persisted but redacted from tool output.',
            additionalProperties: { type: 'string' },
          },
          image: { type: 'string', description: 'Container image for backend=container.' },
          containerRuntime: { type: 'string', description: 'Container CLI runtime, such as docker or podman.' },
          containerName: { type: 'string', description: 'Optional runtime container name. Defaults to sepilotd-{id}.' },
          command: {
            type: 'array',
            items: { type: 'string' },
            description: 'Container command arguments appended after the image.',
          },
          ports: {
            type: 'array',
            description: 'Container port publishes.',
            items: {
              type: 'object',
              properties: {
                containerPort: { type: 'number' },
                hostPort: { type: 'number' },
                protocol: { type: 'string', enum: ['tcp', 'udp'] },
              },
              required: ['containerPort'],
            },
          },
          volumes: {
            type: 'array',
            description: 'Container volume mounts.',
            items: {
              type: 'object',
              properties: {
                source: { type: 'string' },
                target: { type: 'string' },
                readonly: { type: 'boolean' },
              },
              required: ['source', 'target'],
            },
          },
          restart: {
            type: 'object',
            properties: {
              mode: { type: 'string', enum: ['never', 'on-failure', 'always'] },
              maxRestarts: { type: 'number' },
              backoffMs: { type: 'number' },
            },
          },
          health: {
            type: 'object',
            description: 'Optional service health check. Failed checks are restart-policy failures.',
            properties: {
              type: { type: 'string', enum: ['process', 'http', 'tcp'] },
              url: { type: 'string', description: 'HTTP health URL for type=http.' },
              expectedStatus: { type: 'number', description: 'Exact expected HTTP status. Defaults to any status < 500.' },
              host: { type: 'string', description: 'TCP health host for type=tcp.' },
              port: { type: 'number', description: 'TCP health port for type=tcp.' },
              intervalMs: { type: 'number' },
              timeoutMs: { type: 'number' },
              graceMs: { type: 'number' },
            },
          },
        },
      },
      async execute(input): Promise<ToolResult> {
        const start = Date.now()
        try {
          return ok(start, await supervisor.start(parseStartSpec(input)))
        } catch (error) {
          return fail(start, error)
        }
      },
    },
    {
      name: 'service.list',
      description: 'List durable local services managed by the service supervisor. Use this to rediscover a service id in later or resumed sessions.',
      resumeSafety: 'replay-safe',
      scheduling: { mode: 'parallel-safe', resource: 'services' },
      inputSchema: { type: 'object', properties: {} },
      async execute(): Promise<ToolResult> {
        const start = Date.now()
        try {
          return ok(start, await supervisor.list())
        } catch (error) {
          return fail(start, error)
        }
      },
    },
    {
      name: 'service.status',
      description: 'Return one durable local service status after reconciling persisted backend state.',
      resumeSafety: 'replay-safe',
      scheduling: {
        mode: 'parallel-safe',
        resource: 'services',
        key: (input) => typeof input.id === 'string' ? input.id : null,
      },
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Service id.' },
        },
        required: ['id'],
      },
      async execute(input): Promise<ToolResult> {
        const start = Date.now()
        try {
          return ok(start, await supervisor.status(stringValue(input, 'id')))
        } catch (error) {
          return fail(start, error)
        }
      },
    },
    {
      name: 'service.logs',
      description: 'Read or briefly follow persisted stdout and stderr logs for a durable local service. Carry the returned byte offsets into the next call to avoid repeating old logs.',
      resumeSafety: 'replay-safe',
      scheduling: {
        mode: 'parallel-safe',
        resource: 'services',
        key: (input) => typeof input.id === 'string' ? input.id : null,
      },
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Service id.' },
          stdoutOffset: { type: 'number', description: 'Previously consumed stdout byte offset.' },
          stderrOffset: { type: 'number', description: 'Previously consumed stderr byte offset.' },
          limitBytes: { type: 'number', description: 'Maximum bytes to read per stream. Capped at 256KiB.' },
          tailBytes: { type: 'number', description: 'Read the last N bytes from each stream when no offset is supplied. Capped at 256KiB.' },
          followMs: { type: 'number', description: 'Poll for new log bytes for up to this many milliseconds when no output is immediately available. Capped at 30s.' },
          pollIntervalMs: { type: 'number', description: 'Polling interval for followMs. Defaults to 250ms.' },
        },
        required: ['id'],
      },
      async execute(input): Promise<ToolResult> {
        const start = Date.now()
        try {
          return ok(start, await supervisor.logs({
            id: stringValue(input, 'id'),
            stdoutOffset: numberValue(input, 'stdoutOffset'),
            stderrOffset: numberValue(input, 'stderrOffset'),
            limitBytes: numberValue(input, 'limitBytes'),
            tailBytes: numberValue(input, 'tailBytes'),
            followMs: numberValue(input, 'followMs'),
            pollIntervalMs: numberValue(input, 'pollIntervalMs'),
          }))
        } catch (error) {
          return fail(start, error)
        }
      },
    },
    {
      name: 'service.healthcheck',
      description: 'Run one service health check immediately and return the reconciled service status.',
      resumeSafety: 'replay-risky',
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Service id.' },
        },
        required: ['id'],
      },
      async execute(input): Promise<ToolResult> {
        const start = Date.now()
        try {
          return ok(start, await supervisor.healthcheck(stringValue(input, 'id')))
        } catch (error) {
          return fail(start, error)
        }
      },
    },
    {
      name: 'service.stop',
      description: 'Stop a durable local service.',
      resumeSafety: 'replay-risky',
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Service id.' },
          signal: { type: 'string', description: 'Signal such as SIGTERM, SIGINT, or SIGKILL.' },
          timeoutMs: { type: 'number', description: 'How long to wait for exit before returning stopping status.' },
        },
        required: ['id'],
      },
      async execute(input): Promise<ToolResult> {
        const start = Date.now()
        try {
          return ok(start, await supervisor.stop({
            id: stringValue(input, 'id'),
            signal: stringValue(input, 'signal') as NodeJS.Signals || undefined,
            timeoutMs: numberValue(input, 'timeoutMs'),
          }))
        } catch (error) {
          return fail(start, error)
        }
      },
    },
    {
      name: 'service.restart',
      description: 'Restart a durable local service using its persisted spec.',
      resumeSafety: 'replay-risky',
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Service id.' },
        },
        required: ['id'],
      },
      async execute(input): Promise<ToolResult> {
        const start = Date.now()
        try {
          return ok(start, await supervisor.restart(stringValue(input, 'id')))
        } catch (error) {
          return fail(start, error)
        }
      },
    },
    {
      name: 'service.remove',
      description: 'Remove a durable local service record. Running services require force.',
      resumeSafety: 'replay-risky',
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Service id.' },
          force: { type: 'boolean', description: 'Stop the service if it is still running.' },
          deleteLogs: { type: 'boolean', description: 'Delete service log files after removing the record.' },
        },
        required: ['id'],
      },
      async execute(input): Promise<ToolResult> {
        const start = Date.now()
        try {
          return ok(start, await supervisor.remove({
            id: stringValue(input, 'id'),
            force: booleanValue(input, 'force'),
            deleteLogs: booleanValue(input, 'deleteLogs'),
          }))
        } catch (error) {
          return fail(start, error)
        }
      },
    },
  ]
}
