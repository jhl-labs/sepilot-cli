import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import { z } from 'zod'
import { bindCapability } from '../capabilities/bind.js'
import {
  applyAndPersistRuntimeUpdate,
} from '../runtime/config-runtime.js'
import { getRuntimeDataDir } from './utils.js'

const themeSchema = z.enum(['light', 'dark', 'system'])
const localeSchema = z.enum(['ko', 'en'])

const generalSettingsSchema = z.object({
  theme: themeSchema,
  locale: localeSchema,
  workingDir: z.string().min(1).nullable(),
})

const generalSettingsUpdateSchema = generalSettingsSchema.partial()

const llmSettingsSchema = z.object({
  providerId: z.string().min(1).nullable(),
  modelId: z.string().min(1).nullable(),
})

type GeneralSettings = z.infer<typeof generalSettingsSchema>
type GeneralSettingsUpdate = z.infer<typeof generalSettingsUpdateSchema>
type LlmSettings = z.infer<typeof llmSettingsSchema>

const DEFAULT_GENERAL_SETTINGS: GeneralSettings = {
  theme: 'system',
  locale: 'ko',
  workingDir: null,
}

function generalSettingsPath(app: FastifyInstance): string {
  return join(getRuntimeDataDir(app.runtime ?? {}), 'desktop', 'settings.json')
}

async function readGeneralSettings(app: FastifyInstance): Promise<GeneralSettings> {
  try {
    const raw = JSON.parse(
      await readFile(generalSettingsPath(app), 'utf-8'),
    ) as unknown
    const parsed = generalSettingsSchema.safeParse(raw)
    return parsed.success ? parsed.data : DEFAULT_GENERAL_SETTINGS
  } catch {
    return DEFAULT_GENERAL_SETTINGS
  }
}

async function writeGeneralSettings(
  app: FastifyInstance,
  next: GeneralSettingsUpdate,
): Promise<GeneralSettings> {
  const merged = {
    ...(await readGeneralSettings(app)),
    ...next,
  } satisfies GeneralSettings
  const filePath = generalSettingsPath(app)
  const tempPath = `${filePath}.tmp`
  await mkdir(dirname(filePath), { recursive: true })
  await writeFile(tempPath, JSON.stringify(merged, null, 2), {
    encoding: 'utf-8',
    mode: 0o600,
  })
  await rename(tempPath, filePath)
  return merged
}

function currentLlmSettings(app: FastifyInstance): LlmSettings {
  const runtime = app.runtime
  return {
    providerId: runtime?.config.agent.defaultProvider ?? null,
    modelId: runtime?.config.agent.defaultModel ?? null,
  }
}

export async function registerSettingsCapabilityRoutes(
  app: FastifyInstance,
): Promise<void> {
  await bindCapability(
    app,
    {
      name: 'settings',
      version: '1',
      methods: [
        { method: 'GET', path: '/settings/general' },
        { method: 'PUT', path: '/settings/general' },
        { method: 'GET', path: '/settings/llm' },
        { method: 'PUT', path: '/settings/llm' },
      ],
    },
    async (a) => {
      a.get('/settings/general', async () => readGeneralSettings(app))

      a.put('/settings/general', async (req, reply) => {
        const parsed = generalSettingsUpdateSchema.safeParse(req.body)
        if (!parsed.success) {
          return reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
        }

        await writeGeneralSettings(app, parsed.data)
        return reply.status(204).send()
      })

      a.get('/settings/llm', async (_req, reply) => {
        if (!app.runtime) {
          return reply.status(503).send({
            code: 'SERVICE_UNAVAILABLE',
            message: 'Runtime not initialized',
            retriable: true,
          })
        }

        return currentLlmSettings(app)
      })

      a.put('/settings/llm', async (req, reply) => {
        const runtime = app.runtime
        if (!runtime) {
          return reply.status(503).send({
            code: 'SERVICE_UNAVAILABLE',
            message: 'Runtime not initialized',
            retriable: true,
          })
        }

        const parsed = llmSettingsSchema.safeParse(req.body)
        if (!parsed.success) {
          return reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
        }

        const { providerId, modelId } = parsed.data
        if (!providerId) {
          return reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: 'providerId is required',
            retriable: false,
          })
        }

        const provider = runtime.providerRegistry.get(providerId)
        if (!provider) {
          return reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: `Unknown provider: ${providerId}`,
            retriable: false,
          })
        }

        const resolvedModelId = modelId ?? provider.models[0]?.id ?? null
        if (!resolvedModelId) {
          return reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: `Provider ${providerId} does not expose any models`,
            retriable: false,
          })
        }
        if (!provider.models.some((candidate) => candidate.id === resolvedModelId)) {
          return reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: `Unknown model for provider ${providerId}: ${resolvedModelId}`,
            retriable: false,
          })
        }

        await runtime.configMutationService.apply(
          'settings.llm.update',
          async () => {
            runtime.config.agent.defaultProvider = providerId
            runtime.config.agent.defaultModel = resolvedModelId
            await applyAndPersistRuntimeUpdate(
              runtime,
              new Set(['agent.defaultProvider', 'agent.defaultModel']),
              {
                'agent.defaultProvider': providerId,
                'agent.defaultModel': resolvedModelId,
              },
            )
          },
        )

        return reply.status(204).send()
      })
    },
  )
}
