import type { RuntimeServices } from './runtime.js'
import type { RequestAuthContext } from './auth.js'
import type { DaemonRateLimiter } from './rate-limiter.js'
import type { ConnectionRegistry } from './runtime/connection-registry.js'
import type { ChatKnowledgeProviderRegistry } from './chat-knowledge.js'

export interface DaemonShutdownController {
  shutdown(reason: string): Promise<void>
}

declare module 'fastify' {
  interface FastifyInstance {
    runtime?: RuntimeServices
    authToken?: string | null
    authTokenRequired?: boolean
    daemonRateLimiter?: DaemonRateLimiter
    shutdownController?: DaemonShutdownController
    connectionRegistry?: ConnectionRegistry
    chatKnowledgeProviders?: ChatKnowledgeProviderRegistry
  }

  interface FastifyRequest {
    requestId?: string
    rawBody?: string
    authContext?: RequestAuthContext
  }
}
