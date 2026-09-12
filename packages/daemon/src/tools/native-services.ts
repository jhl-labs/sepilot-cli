import {
  NativeServiceManager,
  type NativeServiceInstallInput,
  type NativeServiceProvider,
  type NativeServiceRestart,
} from '../service-supervisor/native-service.js'
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

function booleanValue(input: Record<string, unknown>, key: string): boolean | undefined {
  const value = input[key]
  return typeof value === 'boolean' ? value : undefined
}

function numberValue(input: Record<string, unknown>, key: string): number | undefined {
  const value = input[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function stringArrayValue(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined
  const items = value.filter((item): item is string => typeof item === 'string')
  return items.length > 0 ? items : undefined
}

function stringEnvValue(value: unknown): Record<string, string> | undefined {
  if (!value || typeof value !== 'object') return undefined
  const entries = Object.entries(value)
    .filter((entry): entry is [string, string] => typeof entry[1] === 'string')
  return entries.length > 0 ? Object.fromEntries(entries) : undefined
}

function parseInstallInput(input: Record<string, unknown>): NativeServiceInstallInput {
  return {
    id: stringValue(input, 'id'),
    name: stringValue(input, 'name') || undefined,
    description: stringValue(input, 'description') || undefined,
    provider: stringValue(input, 'provider') as NativeServiceProvider || undefined,
    executable: stringValue(input, 'executable'),
    args: stringArrayValue(input.args),
    cwd: stringValue(input, 'cwd') || undefined,
    env: stringEnvValue(input.env),
    restart: stringValue(input, 'restart') as NativeServiceRestart || undefined,
    enable: booleanValue(input, 'enable'),
    start: booleanValue(input, 'start'),
    reload: booleanValue(input, 'reload'),
  }
}

export function createNativeServiceTools(
  manager = new NativeServiceManager(),
): ToolDefinitionRuntime[] {
  return [
    {
      name: 'service.install',
      description: 'Install a native OS service from an executable spec. Supports Linux systemd --user, macOS launchd user LaunchAgents, and Windows Service registration for service-capable executables.',
      resumeSafety: 'replay-risky',
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Stable native service id.' },
          name: { type: 'string', description: 'Human-readable service name.' },
          description: { type: 'string', description: 'Native service description.' },
          provider: { type: 'string', enum: ['systemd-user', 'launchd-user', 'windows-service'], description: 'Native service provider. Defaults to the host OS provider.' },
          executable: { type: 'string', description: 'Executable path to run.' },
          args: { type: 'array', items: { type: 'string' }, description: 'Command arguments.' },
          cwd: { type: 'string', description: 'Working directory.' },
          env: {
            type: 'object',
            description: 'Environment variables to persist in the native service unit. Values are redacted from tool output.',
            additionalProperties: { type: 'string' },
          },
          restart: { type: 'string', enum: ['no', 'on-failure', 'always'], description: 'Native service restart policy.' },
          enable: { type: 'boolean', description: 'Enable the native service after install.' },
          start: { type: 'boolean', description: 'Start the native service after install.' },
          reload: { type: 'boolean', description: 'Run daemon-reload after file changes. Defaults to true.' },
        },
        required: ['id', 'executable'],
      },
      async execute(input): Promise<ToolResult> {
        const start = Date.now()
        try {
          return ok(start, await manager.install(parseInstallInput(input)))
        } catch (error) {
          return fail(start, error)
        }
      },
    },
    {
      name: 'service.native.status',
      description: 'Return native OS user service installation, enablement, and active state.',
      resumeSafety: 'replay-safe',
      scheduling: {
        mode: 'parallel-safe',
        resource: 'native-services',
        key: (input) => typeof input.id === 'string' ? input.id : null,
      },
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Native service id.' },
        },
        required: ['id'],
      },
      async execute(input): Promise<ToolResult> {
        const start = Date.now()
        try {
          return ok(start, await manager.status(stringValue(input, 'id')))
        } catch (error) {
          return fail(start, error)
        }
      },
    },
    {
      name: 'service.native.logs',
      description: 'Read stdout and stderr logs for a native OS user service installed through service.install.',
      resumeSafety: 'replay-safe',
      scheduling: {
        mode: 'parallel-safe',
        resource: 'native-services',
        key: (input) => typeof input.id === 'string' ? input.id : null,
      },
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Native service id.' },
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
          return ok(start, await manager.logs({
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
      name: 'service.enable',
      description: 'Enable a previously installed native OS user service.',
      resumeSafety: 'replay-risky',
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Native service id.' },
          start: { type: 'boolean', description: 'Start the service after enabling.' },
          reload: { type: 'boolean', description: 'Run daemon-reload before enabling. Defaults to true.' },
        },
        required: ['id'],
      },
      async execute(input): Promise<ToolResult> {
        const start = Date.now()
        try {
          return ok(start, await manager.enable({
            id: stringValue(input, 'id'),
            start: booleanValue(input, 'start'),
            reload: booleanValue(input, 'reload'),
          }))
        } catch (error) {
          return fail(start, error)
        }
      },
    },
    {
      name: 'service.disable',
      description: 'Disable a previously installed native OS user service.',
      resumeSafety: 'replay-risky',
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Native service id.' },
          stop: { type: 'boolean', description: 'Stop the service before disabling.' },
          reload: { type: 'boolean', description: 'Run daemon-reload after disabling. Defaults to true.' },
        },
        required: ['id'],
      },
      async execute(input): Promise<ToolResult> {
        const start = Date.now()
        try {
          return ok(start, await manager.disable({
            id: stringValue(input, 'id'),
            stop: booleanValue(input, 'stop'),
            reload: booleanValue(input, 'reload'),
          }))
        } catch (error) {
          return fail(start, error)
        }
      },
    },
    {
      name: 'service.uninstall',
      description: 'Uninstall a native OS user service and remove its unit file.',
      resumeSafety: 'replay-risky',
      inputSchema: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Native service id.' },
          stop: { type: 'boolean', description: 'Stop the service before uninstalling. Defaults to true.' },
          reload: { type: 'boolean', description: 'Run daemon-reload after uninstalling. Defaults to true.' },
        },
        required: ['id'],
      },
      async execute(input): Promise<ToolResult> {
        const start = Date.now()
        try {
          return ok(start, await manager.uninstall({
            id: stringValue(input, 'id'),
            stop: booleanValue(input, 'stop'),
            reload: booleanValue(input, 'reload'),
          }))
        } catch (error) {
          return fail(start, error)
        }
      },
    },
  ]
}
