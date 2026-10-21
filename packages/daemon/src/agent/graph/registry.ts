import type { AgentExecutionCapability } from '@sepilotd/core'
import type { AgentGraph } from './engine.js'
import type { Deps } from './nodes.js'
import { validateGraph } from './validator.js'

export type GraphBuilder = (deps: Deps) => AgentGraph

function graphValidationDeps(): Deps {
  return new Proxy({}, { get: () => undefined }) as unknown as Deps
}
export type GraphAgentSource = 'builtin' | 'yaml' | 'plugin' | 'capability' | 'user'

export interface GraphAgentLimits {
  maxIterations?: number
}

export interface GraphAgentCapabilities {
  artifactWrite?: boolean
  nativeMultimodalInput?: boolean
  readOnly?: boolean
  direct?: boolean
  durableRunContract?: boolean
  autoRoutable?: boolean
  executionCapabilities?: AgentExecutionCapability[]
  auxiliaryStages?: 'none'
  executionControl?: 'model-directed'
  escalation?: GraphEscalationTarget
}

export interface GraphEscalationTarget {
  mode: string
  bot?: string
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

  private assertValidStructure(info: GraphAgentInfo): void {
    let graph: AgentGraph
    try {
      graph = info.builder(graphValidationDeps())
    } catch {
      return
    }
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
