import type { AgentExecutionCapability } from '@sepilotd/core'
import type { AgentGraph } from './engine.js'
import type { Deps } from './nodes.js'
import { validateGraph } from './validator.js'

export type GraphBuilder = (deps: Deps) => AgentGraph

// A structural-only stub used to build a graph for validation at registration
// time (real deps are not available yet). Node builders only capture deps in
// closures — they do not dereference provider/tools while assembling the graph —
// so an all-undefined deps proxy is enough to inspect the edge/node topology.
function graphValidationDeps(): Deps {
  return new Proxy({}, { get: () => undefined }) as unknown as Deps
}
export type GraphAgentSource = 'builtin' | 'yaml' | 'plugin' | 'capability' | 'user'

export interface GraphAgentLimits {
  maxIterations?: number
}

export interface GraphAgentCapabilities {
  /**
   * This graph can create or update durable user-requested artifacts such as
   * reports, markdown files, generated pages, or repository files.
   */
  artifactWrite?: boolean
  /** Graph consumes structured currentUserContent, preserving its tool boundary for attachments. */
  nativeMultimodalInput?: boolean
  /**
   * This graph is intended to stay read-only unless a caller explicitly wraps it
   * in a writer-capable orchestration flow.
   */
  readOnly?: boolean
  /**
   * This graph is a deterministic, narrow capability path. Auto mode may execute
   * it without hydrating long-term memory/document context first because the
   * graph's own tool call fully answers the request.
   */
  direct?: boolean
  /**
   * Attach the durable acceptance-criteria contract and completion gate to
   * this graph. Operational focused loops may opt out because one successful
   * structured tool observation plus a final summary is already their durable
   * evidence boundary; forcing prose criterion verdicts makes them repeat
   * completed commands. Defaults to true.
   */
  durableRunContract?: boolean
  /**
   * Offer this graph to semantic routing (the explicit `auto` classifier and
   * the in-run `agent.transfer` catalog). Builtin graphs default to false so
   * legacy and narrow specialist graphs stay explicit-only; user, YAML and
   * plugin graphs default to true.
   */
  autoRoutable?: boolean
  /**
   * Semantic execution capabilities this graph can fulfill end-to-end. Tool
   * allowlists may include narrower supporting observations (for example a
   * browser graph can inspect an already-running managed process) without
   * advertising the full `process` lifecycle capability here.
   */
  executionCapabilities?: AgentExecutionCapability[]
}

export interface GraphAgentInfo {
  id: string
  name: string
  description: string
  builder: GraphBuilder
  source?: GraphAgentSource
  limits?: GraphAgentLimits
  capabilities?: GraphAgentCapabilities
}

export class GraphAgentRegistry {
  private agents = new Map<string, GraphAgentInfo>()

  register(info: GraphAgentInfo): void {
    this.assertValidStructure(info)
    this.agents.set(info.id, info)
  }

  // Reject a graph whose structure is broken (missing start node, edge target
  // that does not exist) at registration instead of surfacing it as a confusing
  // runtime routing error on the first run. Builders that cannot be assembled
  // with stub deps are skipped rather than blocking registration.
  private assertValidStructure(info: GraphAgentInfo): void {
    let graph: AgentGraph
    try {
      graph = info.builder(graphValidationDeps())
    } catch {
      return
    }
    // A builder that cannot assemble a real graph with stub deps (e.g. test
    // stubs returning null/undefined) yields nothing to validate — skip it
    // rather than crashing on the structural inspection below.
    if (!graph || typeof graph.getNodes !== 'function') {
      return
    }
    const result = validateGraph(graph)
    if (result.errors.length > 0) {
      throw new Error(
        `graph '${info.id}' failed structural validation: ${result.errors.join('; ')}`,
      )
    }
  }

  unregister(id: string): void {
    this.agents.delete(id)
  }

  get(id: string): GraphAgentInfo | undefined {
    return this.agents.get(id)
  }

  list(): GraphAgentInfo[] {
    return Array.from(this.agents.values())
  }
}
