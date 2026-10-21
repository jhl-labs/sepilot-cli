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
  isolation?: 'worktree'
  maxIterations?: number
  issues?: string[]
  warnings?: string[]
}

const SUPPORTED_AGENT_KEYS = new Set([
  'name',
  'description',
  'mode',
  'model',
  'allowed_tools',
  'denied_tools',
  'tools',
  'disallowedTools',
  'color',
  'isolation',
  'max_iterations',
  'maxIterations',
])

const UNSUPPORTED_POLICY_KEYS = new Set([
  'hooks',
  'permissionMode',
  'permission_mode',
  'permissions',
  'background',
  'mcpServers',
  'mcp_servers',
  'sandbox',
])

export const CUSTOM_AGENT_MAX_ITERATIONS_CAP = 50

export function compileCustomAgents(defs: CustomDef[]): CustomAgentRecord[] {
  return defs.map((d) => {
    const issues = d.error ? [d.error] : []
    const warnings: string[] = []
    for (const key of Object.keys(d.data)) {
      if (SUPPORTED_AGENT_KEYS.has(key)) continue
      if (UNSUPPORTED_POLICY_KEYS.has(key)) issues.push(`Unsupported agent setting: ${key}`)
      else warnings.push(`Ignored unknown agent setting: ${key}`)
    }
    const maxIterations = iterationBudget(d.data.max_iterations, d.data.maxIterations, issues)
    const allowedTools = toolNames(d.data.allowed_tools, d.data.tools, 'tools', issues)
    const deniedTools = toolNames(
      d.data.denied_tools,
      d.data.disallowedTools,
      'disallowedTools',
      issues,
    )
    if (d.data.mode !== undefined && !['primary', 'subagent', 'all'].includes(String(d.data.mode)))
      issues.push('Invalid agent mode')
    if (d.data.isolation !== undefined && d.data.isolation !== 'worktree')
      issues.push('isolation must be worktree')
    return {
      id: d.id,
      mode: toMode(d.data.mode),
      model: typeof d.data.model === 'string' ? d.data.model : undefined,
      description: typeof d.data.description === 'string' ? d.data.description : undefined,
      systemPrompt: d.body.trim(),
      source: d.source,
      allowedTools,
      deniedTools,
      ...(d.data.isolation === 'worktree' ? { isolation: 'worktree' as const } : {}),
      ...(maxIterations !== undefined ? { maxIterations } : {}),
      ...(issues.length ? { issues } : {}),
      ...(warnings.length ? { warnings } : {}),
    }
  })
}

const TOOL_ALIASES: Record<string, string> = {
  Read: 'fs.read',
  Write: 'fs.write',
  Edit: 'fs.edit',
  Glob: 'fs.glob',
  Grep: 'fs.search',
  Bash: 'terminal.run',
  Agent: 'subagent.dispatch',
}

function toolNames(
  native: unknown,
  foreign: unknown,
  field: string,
  issues: string[],
): string[] | undefined {
  if (native !== undefined && foreign !== undefined)
    issues.push(`Conflicting native and imported ${field} settings`)
  const value = native ?? foreign
  if (value === undefined) return undefined
  const list: unknown =
    typeof value === 'string'
      ? value
          .split(',')
          .map((item) => item.trim())
          .filter(Boolean)
      : value
  if (!Array.isArray(list) || !list.every((item) => typeof item === 'string' && item.trim())) {
    issues.push(`${field} must be a comma-separated string or string array`)
    return []
  }
  const names = (list as string[])
    .map((item) => item.trim())
    .map((name) => TOOL_ALIASES[name] ?? name)
  for (const name of names)
    if (!/^[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*$/.test(name))
      issues.push(`Unsupported ${field} rule syntax; use exact Sepilot tool names`)
  return [...new Set(names)]
}

function iterationBudget(native: unknown, imported: unknown, issues: string[]): number | undefined {
  if (native !== undefined && imported !== undefined) {
    issues.push('Conflicting max_iterations and maxIterations settings')
    return undefined
  }
  const value = native ?? imported
  if (value === undefined) return undefined
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) {
    issues.push('max_iterations must be a positive integer')
    return undefined
  }
  return Math.min(value, CUSTOM_AGENT_MAX_ITERATIONS_CAP)
}

export function assertCustomAgentUsable(record: CustomAgentRecord): void {
  if (record.issues?.length)
    throw new Error(
      `Agent "${record.id}" cannot run (${record.source}): ${record.issues.join('; ')}`,
    )
}

function toMode(v: unknown): AgentMode {
  return v === 'subagent' || v === 'all' ? v : 'primary'
}
