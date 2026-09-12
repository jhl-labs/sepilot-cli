import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { bindCapability } from '../capabilities/bind.js'

export interface ProviderModel {
  id: string
  label: string
  contextWindow?: number
  vision?: boolean
  maxOutputTokens?: number
}

export interface ProviderInfo {
  id: string
  label: string
  models: ProviderModel[]
}

export interface ProviderCatalog {
  list(): Promise<ProviderInfo[]>
  test(input: {
    providerId: string
    modelId?: string
  }): Promise<{ ok: boolean; latencyMs?: number; reason?: string }>
}

/**
 * 기본 카탈로그 — daemon의 기존 providers 모듈이 직접 provide하기 전까지
 * 비어 있는 목록을 반환한다. desktop은 `/llm/providers`가 200 빈 배열이면
 * "등록된 공급자 없음" 상태를 표시한다.
 */
export const emptyProviderCatalog: ProviderCatalog = {
  list: async () => [],
  test: async () => ({ ok: false, reason: 'no provider registered' }),
}

export async function registerLlmProvidersCapabilityRoutes(
  app: FastifyInstance,
  catalog: ProviderCatalog = emptyProviderCatalog,
): Promise<void> {
  await bindCapability(
    app,
    {
      name: 'llm',
      version: '1',
      methods: [
        { method: 'GET', path: '/llm/providers' },
        { method: 'POST', path: '/llm/test-connection' },
      ],
    },
    async (a) => {
      a.get('/llm/providers', async () => {
        if (catalog !== emptyProviderCatalog) {
          return catalog.list()
        }

        const runtime = app.runtime
        return (runtime?.providerRegistry.list() ?? []).map((provider) => ({
          id: provider.id,
          label: provider.name,
          models: provider.models.map((model) => ({
            id: model.id,
            label: model.id,
            contextWindow: model.contextWindow,
            maxOutputTokens: model.maxOutputTokens,
            vision: model.capabilities.vision === true,
          })),
        }))
      })

      const TestBody = z.object({
        providerId: z.string().min(1),
        modelId: z.string().min(1).optional(),
      })
      a.post('/llm/test-connection', async (req, reply) => {
        const parsed = TestBody.safeParse(req.body)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        return catalog.test(parsed.data)
      })
    },
  )
}
