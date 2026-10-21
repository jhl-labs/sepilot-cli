import { homedir } from 'node:os'
import type { FastifyInstance } from 'fastify'
import '../../server/fastify-types.js'
import {
  createInjectLlmTurnRunner,
  projectWorkspaceDirectory,
  realpathOrSelf,
  untrustedTurnToolNames,
  untrustedWorkspaceRootProblem,
} from '../../server/runtime/event-routing-llm.js'
import { loadProjectContext } from '../../server/routes/project-context.js'
import { getRuntimeDataDir } from '../../server/routes/utils.js'
import { isEgressAllowlistActive } from '../../utils/egress-policy.js'

export { startEventRouting, stopEventRouting } from '../../server/runtime/event-routing-startup.js'

export function installLlmRunner(app: FastifyInstance): void {
  const runtime = app.runtime
  if (!runtime) return
  const untrustedWorkspaceScope = {
    home: realpathOrSelf(homedir()),
    dataDir: realpathOrSelf(getRuntimeDataDir(runtime)),
  }
  const egressAllowlistActiveAtStartup = isEgressAllowlistActive(runtime.config?.security?.egressAllowlist)
  runtime.eventRoutingLlmRunner = createInjectLlmTurnRunner({
    inject: (opts) => app.inject(opts),
    authToken: () => (app as unknown as { authToken?: string | null }).authToken,
    cancelRun: (sessionId) => app.runtime?.activeRuns?.cancel(sessionId) ?? Promise.resolve(false),
    setSessionTitle: async (sessionId, title) => { await app.runtime?.sessions?.updateMeta?.(sessionId, { title }) },
    toolCeiling: (scope) => {
      const registry = app.runtime?.toolRegistry
      return registry ? untrustedTurnToolNames(registry, scope) : []
    },
    resolveProjectWorkspace: async (projectId) =>
      projectWorkspaceDirectory((await loadProjectContext(app.runtime ?? {}, projectId))?.workingDirectory),
    workspaceRootProblem: (root) => untrustedWorkspaceRootProblem(root, untrustedWorkspaceScope),
    egressAllowlistActive: () => egressAllowlistActiveAtStartup
      && isEgressAllowlistActive(app.runtime?.config?.security?.egressAllowlist),
  })
}
