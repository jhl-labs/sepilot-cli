import { DaemonWsClient as SharedDaemonWsClient } from '@sepilotd/api-client/node'
import { memoryScopeFor } from './memory-scope.js'
import { loadDaemonToken, resolveDaemonBaseUrl } from './token.js'

export class DaemonWsClient extends SharedDaemonWsClient {
  constructor(baseUrl?: string) {
    const resolvedBaseUrl = resolveDaemonBaseUrl(baseUrl)
    const token = loadDaemonToken()
    super({
      baseUrl: resolvedBaseUrl,
      token,
      surface: 'cli',
      memoryScope: resolvedBaseUrl ? memoryScopeFor(resolvedBaseUrl, token) : undefined,
    })
  }
}
