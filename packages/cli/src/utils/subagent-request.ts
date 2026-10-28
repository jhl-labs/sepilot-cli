import { readFile } from 'node:fs/promises'
import type { SubagentDispatchInput } from '@sepilotd/api-client'

export interface SubagentRequestOptions {
  system?: string
  systemFile?: string
  category?: string
  agent?: string
  maxIterations?: string
  tools?: string
  model?: string
  provider?: string
  requestKey?: string
  parentSession?: string
  cwd?: string
  isolation?: 'worktree'
  worktreeBase?: 'head' | 'working-tree'
  contextMode?: 'packet' | 'fork'
}

export async function buildSubagentRequest(
  prompt: string,
  options: SubagentRequestOptions,
  reader = (path: string) => readFile(path, 'utf8'),
): Promise<SubagentDispatchInput> {
  if (options.isolation !== undefined && options.isolation !== 'worktree')
    throw new Error('--isolation must be worktree')
  if (
    options.worktreeBase !== undefined &&
    !['head', 'working-tree'].includes(options.worktreeBase)
  )
    throw new Error('--worktree-base must be head or working-tree')
  if (options.contextMode !== undefined && !['packet', 'fork'].includes(options.contextMode))
    throw new Error('--context-mode must be packet or fork')
  if (!prompt.trim()) throw new Error('Subagent prompt is required')
  if (
    options.maxIterations !== undefined &&
    (!/^\d+$/.test(options.maxIterations) ||
      !Number.isSafeInteger(Number(options.maxIterations)) ||
      Number(options.maxIterations) < 1)
  )
    throw new Error('--max-iterations must be a positive integer')
  if (options.worktreeBase && options.isolation !== 'worktree')
    throw new Error('--worktree-base requires --isolation worktree')
  if (options.contextMode === 'fork' && !options.parentSession)
    throw new Error('--context-mode fork requires --parent-session')
  if (options.requestKey && !options.parentSession) throw new Error('--request-key requires --parent-session')
  return {
    ...(options.requestKey ? { requestKey: options.requestKey } : {}),
    isolation: options.isolation,
    ...(options.worktreeBase ? { worktreeBase: options.worktreeBase } : {}),
    ...(options.contextMode ? { contextMode: options.contextMode } : {}),
    cwd: options.cwd ?? process.cwd(),
    prompt,
    system: options.systemFile ? await reader(options.systemFile) : options.system,
    category: options.category,
    agentId: options.agent,
    maxIterations: options.maxIterations === undefined ? undefined : Number(options.maxIterations),
    tools: options.tools
      ?.split(',')
      .map((value) => value.trim())
      .filter(Boolean),
    model: options.model,
    ...(options.provider ? { provider: options.provider } : {}),
    parentSessionId: options.parentSession,
  }
}

export function parseSubagentArgs(args: string[]): {
  prompt: string
  options: SubagentRequestOptions
  background: boolean
} {
  const names: Record<string, keyof SubagentRequestOptions> = {
    system: 'system',
    'system-file': 'systemFile',
    category: 'category',
    agent: 'agent',
    'max-iterations': 'maxIterations',
    tools: 'tools',
    model: 'model',
    provider: 'provider',
    'parent-session': 'parentSession',
    'request-key': 'requestKey',
    cwd: 'cwd',
    isolation: 'isolation',
    'worktree-base': 'worktreeBase',
    'context-mode': 'contextMode',
  }
  const options: Record<string, string> = {}
  const prompt: string[] = []
  let background = false
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index]!
    if (arg === '--') {
      prompt.push(...args.slice(index + 1))
      break
    }
    if (arg === '--background') {
      background = true
      continue
    }
    if (!arg.startsWith('--')) {
      prompt.push(arg)
      continue
    }
    const equals = arg.indexOf('=')
    const key = arg.slice(2, equals < 0 ? undefined : equals)
    const name = names[key]
    if (!name) throw new Error(`Unknown subagent option: --${key}`)
    const value = equals < 0 ? args[++index] : arg.slice(equals + 1)
    if (!value || value.startsWith('--')) throw new Error(`Missing value for --${key}`)
    options[name] = value
  }
  return { prompt: prompt.join(' '), options: options as SubagentRequestOptions, background }
}
