import type { FastifyReply } from 'fastify'

export interface ConnectedMcpClient {
  isConnected: () => boolean
}

export interface McpRouteRuntime<TClient extends ConnectedMcpClient> {
  mcpManager?: {
    listServers?: () => Array<{ name: string }>
    getClient?: (server: string) => TClient | undefined | null
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export async function resolveConnectedMcpClient<TClient extends ConnectedMcpClient>(
  runtime: McpRouteRuntime<TClient>,
  name: string,
): Promise<TClient | undefined | null> {
  const manager = runtime.mcpManager
  const initial = manager?.getClient?.(name)
  if (initial?.isConnected()) return initial

  const knownServer = manager?.listServers?.().some((server) => server.name === name)
  if (!knownServer) return initial

  const deadline = Date.now() + 1000
  while (Date.now() <= deadline) {
    const client = manager?.getClient?.(name)
    if (client?.isConnected()) return client
    await sleep(50)
  }

  return manager?.getClient?.(name)
}

export async function getConnectedMcpClient<TClient extends ConnectedMcpClient>(
  runtime: McpRouteRuntime<TClient> | undefined,
  reply: FastifyReply,
  name: string,
): Promise<TClient | null> {
  if (!runtime) {
    reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    return null
  }
  const client = await resolveConnectedMcpClient(runtime, name)
  if (!client) {
    reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Server not found' } })
    return null
  }
  if (!client.isConnected()) {
    reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Server disconnected' } })
    return null
  }
  return client
}

export function sendMcpUpstreamError(reply: FastifyReply, err: unknown) {
  const message = err instanceof Error ? err.message : 'upstream failed'
  return reply.status(502).send({ error: { code: 'UPSTREAM_ERROR', message } })
}
