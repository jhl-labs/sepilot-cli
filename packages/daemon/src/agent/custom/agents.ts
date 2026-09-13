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
  issues?: string[]
}

export function compileCustomAgents(defs: CustomDef[]): CustomAgentRecord[] {
  return defs.map((d) => {
    const issues = d.error ? [d.error] : []
    const supported = new Set([
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
    ])
    for (const key of Object.keys(d.data))
      if (!supported.has(key)) issues.push(`Unsupported agent setting: ${key}`)
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
      ...(issues.length ? { issues } : {}),
    }
  })
}

// Only exact semantic equivalents are imported. Argument matchers and unknown
// foreign names must never silently weaken a deny rule.
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

export function assertCustomAgentUsable(record: CustomAgentRecord): void {
  if (record.issues?.length)
    throw new Error(
      `Agent "${record.id}" cannot run (${record.source}): ${record.issues.join('; ')}`,
    )
}

function toMode(v: unknown): AgentMode {
  return v === 'subagent' || v === 'all' ? v : 'primary'
}
