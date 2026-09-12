import type { ImageGenQueue, Job } from '../media/image-gen/queue.js'
import { readOutput } from '../media/image-gen/files.js'
import type { Provider } from '../media/image-gen/adapter.js'
import { getImageGenQueue, imageGenProviders } from '../media/image-gen/runtime.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

interface ImageGenToolDeps {
  providers?: () => Map<string, Provider>
  queue?: () => ImageGenQueue
}

function nowResult(start: number, status: ToolResult['status'], output: unknown): ToolResult {
  return {
    status,
    output: typeof output === 'string' ? output : JSON.stringify(output, null, 2),
    durationMs: Date.now() - start,
  }
}

function providers(deps?: ImageGenToolDeps): Map<string, Provider> {
  return deps?.providers?.() ?? imageGenProviders
}

function queue(deps?: ImageGenToolDeps): ImageGenQueue {
  return deps?.queue?.() ?? getImageGenQueue()
}

function stringInput(input: Record<string, unknown>, key: string): string | undefined {
  const value = input[key]
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function objectInput(
  input: Record<string, unknown>,
  key: string,
): Record<string, unknown> | undefined {
  const value = input[key]
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

function numberInput(input: Record<string, unknown>, key: string): number | undefined {
  const value = input[key]
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return undefined
}

function terminal(status: Job['status']): boolean {
  return status === 'succeeded' || status === 'failed' || status === 'cancelled'
}

async function waitForJob(
  q: ImageGenQueue,
  id: string,
  waitMs: number,
  signal?: AbortSignal,
): Promise<Job> {
  const deadline = Date.now() + waitMs
  while (Date.now() < deadline) {
    if (signal?.aborted) throw new Error('image generation cancelled')
    const job = q.get(id)
    if (!job) throw new Error(`image job not found: ${id}`)
    if (terminal(job.status)) return job
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  const job = q.get(id)
  if (!job) throw new Error(`image job not found: ${id}`)
  return job
}

function jobPayload(job: Job): Record<string, unknown> {
  return {
    ...job,
    fileRoutes: job.outputs.map((output) => `/image-gen/files/${encodeURIComponent(output.id)}`),
  }
}

export function createImageGenProvidersTool(deps?: ImageGenToolDeps): ToolDefinitionRuntime {
  return {
    name: 'image_gen.providers',
    description: 'List sepilotd image generation providers available to the daemon.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'image-gen' },
    inputSchema: { type: 'object', properties: {} },
    async execute(): Promise<ToolResult> {
      const start = Date.now()
      return nowResult(start, 'success', {
        providers: Array.from(providers(deps).values()).map((provider) => provider.info),
      })
    },
  }
}

export function createImageGenCreateTool(deps?: ImageGenToolDeps): ToolDefinitionRuntime {
  return {
    name: 'image_gen.create',
    description:
      'Create an image generation job. Use waitMs to poll until done when the user expects an image immediately.',
    resumeSafety: 'replay-risky',
    scheduling: { mode: 'sequential', resource: 'image-gen' },
    inputSchema: {
      type: 'object',
      properties: {
        providerId: { type: 'string' },
        prompt: { type: 'string' },
        params: { type: 'object' },
        waitMs: { type: 'number' },
      },
      required: ['providerId', 'prompt'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      try {
        const providerId = stringInput(input, 'providerId')
        const prompt = stringInput(input, 'prompt')
        if (!providerId || !prompt) {
          return nowResult(start, 'error', 'providerId and prompt are required')
        }
        const q = queue(deps)
        let job = q.enqueue({
          providerId,
          prompt,
          params: objectInput(input, 'params'),
        })
        const waitMs = Math.max(0, Math.min(numberInput(input, 'waitMs') ?? 0, 30 * 60_000))
        if (waitMs > 0) {
          job = await waitForJob(q, job.id, waitMs, context?.signal)
        }
        return nowResult(start, 'success', jobPayload(job))
      } catch (error) {
        return nowResult(start, 'error', error instanceof Error ? error.message : String(error))
      }
    },
  }
}

export function createImageGenJobTool(deps?: ImageGenToolDeps): ToolDefinitionRuntime {
  return {
    name: 'image_gen.job',
    description: 'Get the latest status and output file routes for an image generation job.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'image-gen' },
    inputSchema: {
      type: 'object',
      properties: { id: { type: 'string' } },
      required: ['id'],
    },
    async execute(input): Promise<ToolResult> {
      const start = Date.now()
      const id = stringInput(input, 'id')
      if (!id) return nowResult(start, 'error', 'id is required')
      const job = queue(deps).get(id)
      if (!job) return nowResult(start, 'error', `image job not found: ${id}`)
      return nowResult(start, 'success', jobPayload(job))
    },
  }
}

export function createImageGenFileTool(): ToolDefinitionRuntime {
  return {
    name: 'image_gen.file',
    description:
      'Read metadata for a generated image or video output and verify the bytes are available.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'image-gen' },
    inputSchema: {
      type: 'object',
      properties: { fileId: { type: 'string' } },
      required: ['fileId'],
    },
    async execute(input): Promise<ToolResult> {
      const start = Date.now()
      try {
        const fileId = stringInput(input, 'fileId')
        if (!fileId) return nowResult(start, 'error', 'fileId is required')
        const output = readOutput(fileId)
        if (!output) return nowResult(start, 'error', `image file not found: ${fileId}`)
        return nowResult(start, 'success', {
          fileId,
          mime: output.mime,
          bytes: output.bytes.length,
          route: `/image-gen/files/${encodeURIComponent(fileId)}`,
        })
      } catch (error) {
        return nowResult(start, 'error', error instanceof Error ? error.message : String(error))
      }
    },
  }
}

export function createImageGenCancelTool(deps?: ImageGenToolDeps): ToolDefinitionRuntime {
  return {
    name: 'image_gen.cancel',
    description: 'Cancel a queued or running image generation job.',
    resumeSafety: 'replay-risky',
    scheduling: { mode: 'sequential', resource: 'image-gen' },
    inputSchema: {
      type: 'object',
      properties: { id: { type: 'string' } },
      required: ['id'],
    },
    async execute(input): Promise<ToolResult> {
      const start = Date.now()
      const id = stringInput(input, 'id')
      if (!id) return nowResult(start, 'error', 'id is required')
      queue(deps).cancel(id)
      return nowResult(start, 'success', { ok: true, id })
    },
  }
}
