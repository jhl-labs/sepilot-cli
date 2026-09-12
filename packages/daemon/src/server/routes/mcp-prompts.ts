import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import type { PromptDescriptor } from '../../mcp/client.js'

interface Params { name: string; prompt: string }
type PromptClient = {
  isConnected?: () => boolean
  discoverPrompts?: () => Promise<PromptDescriptor[]>
}
type PromptRouteRuntime = {
  mcpPromptsRegistry?: {
    list: (server: string) => PromptDescriptor[] | null
    set?: (server: string, prompts: PromptDescriptor[]) => void
  }
  mcpManager?: {
    listServers?: () => Array<{ name: string }>
    getClient?: (server: string) => PromptClient | undefined
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function resolvePromptList(
  runtime: PromptRouteRuntime,
  name: string,
): Promise<PromptDescriptor[] | null> {
  const registry = runtime.mcpPromptsRegistry
  const initial = registry?.list(name)
  if (initial != null) return initial

  const serverExists =
    runtime.mcpManager?.listServers?.().some((server) => server.name === name)
    || Boolean(runtime.mcpManager?.getClient?.(name))
  if (!serverExists) return null

  const deadline = Date.now() + 1000
  while (Date.now() <= deadline) {
    const list = registry?.list(name)
    if (list != null) return list

    const client = runtime.mcpManager?.getClient?.(name)
    if (client?.isConnected?.()) {
      const prompts = await client.discoverPrompts?.()
      if (prompts) {
        registry?.set?.(name, prompts)
        return prompts
      }
    }

    await sleep(50)
  }

  return null
}

export async function mcpPromptsRoutes(app: FastifyInstance) {
  const runtime = app.runtime

  app.get<{ Params: Pick<Params, 'name'> }>('/mcp/servers/:name/prompts', async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const list = await resolvePromptList(runtime, params.name)
    if (list == null) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Server not found' } })
    return { data: list }
  })

  app.post<{ Params: Params; Body: { arguments?: Record<string, string> } }>(
    '/mcp/servers/:name/prompts/:prompt',
    async (request, reply) => {
      if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
      const params = request.params
      const body = request.body
      const name = params.name
      const prompt = params.prompt
      const list = await resolvePromptList(runtime, name)
      if (list == null) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Server not found' } })
      if (!list.some((p) => p.name === prompt)) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Prompt not found' } })
      }
      const client = runtime.mcpManager?.getClient(name)
      if (!client?.isConnected()) {
        return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Server disconnected' } })
      }
      try {
        const res = await client.getPrompt(prompt, body?.arguments ?? {})
        return { data: res }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'upstream failed'
        return reply.status(502).send({ error: { code: 'UPSTREAM_ERROR', message } })
      }
    },
  )
}
