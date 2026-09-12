import type { CustomDef } from './loader.js'

export type AgentMode = 'primary' | 'subagent' | 'all'

export interface CustomAgentRecord {
  id: string
  mode: AgentMode
  model?: string
  description?: string
  systemPrompt: string
  source: string
  allowedTools?: string[]
  deniedTools?: string[]
}

export function compileCustomAgents(defs: CustomDef[]): CustomAgentRecord[] {
  return defs.map((d) => ({
    id: d.id,
    mode: toMode(d.data.mode),
    model: typeof d.data.model === 'string' ? d.data.model : undefined,
    description: typeof d.data.description === 'string' ? d.data.description : undefined,
    systemPrompt: d.body.trim(),
    source: d.source,
    allowedTools: Array.isArray(d.data.allowed_tools)
      ? (d.data.allowed_tools as string[])
      : undefined,
    deniedTools: Array.isArray(d.data.denied_tools)
      ? (d.data.denied_tools as string[])
      : undefined,
  }))
}

function toMode(v: unknown): AgentMode {
  return v === 'subagent' || v === 'all' ? v : 'primary'
}
