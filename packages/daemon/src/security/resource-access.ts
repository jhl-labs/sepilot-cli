import type { PolicyCheckResult, ResourceAccessPolicy, ToolExecRequest } from '@sepilotd/core'
import { lstatSync, readFileSync, realpathSync } from 'node:fs'
import { dirname, isAbsolute, relative, resolve } from 'node:path'
import { parseApplyPatch } from '../tools/apply-patch-parser.js'
import { resolveToolCwd, resolveToolPath } from '../tools/path-utils.js'

const FILE_TOOLS = new Set(['fs.read', 'fs.write', 'fs.edit', 'fs.append', 'fs.move', 'fs.list', 'apply_patch'])
const GIT_TOOLS = new Set(['git.status', 'git.diff', 'git.log', 'git.transfer'])
const INTERNAL_TOOLS = new Set(['question', 'todowrite'])
const block = (reason: string): PolicyCheckResult => ({ allowed: false, reason: `Access policy: ${reason}` })

export function accessToolMatches(pattern: string, name: string): boolean {
  if (pattern === name) return true
  const p = pattern.includes('.') || pattern === '*' ? pattern : `${pattern}.*`
  return new RegExp('^' + p.split('*').map(s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*') + '$').test(name)
}

export function normalizeResourceAccessPolicy(raw: unknown): ResourceAccessPolicy {
  if (!raw || typeof raw !== 'object') throw new Error('policy.access must be an object')
  const p = raw as ResourceAccessPolicy
  if (p.version !== 1 || typeof p.enabled !== 'boolean') throw new Error('Unsupported policy.access version or enabled value')
  for (const key of ['tools', 'urls', 'paths'] as const) {
    for (const mode of ['allow', 'deny'] as const) {
      if (!Array.isArray(p[key]?.[mode]) || p[key][mode].some(v => typeof v !== 'string' || !v.trim() || /[\u0000-\u001f\u007f]/.test(v))) throw new Error(`Invalid policy.access.${key}.${mode}`)
    }
  }
  if (p.internalTools !== undefined && (!Array.isArray(p.internalTools) || p.internalTools.some(v => typeof v !== 'string' || v.includes('*')))) throw new Error('Invalid policy.access.internalTools')
  for (const v of [...p.urls.allow, ...p.urls.deny]) normalizeAccessUrl(v)
  for (const v of [...p.paths.allow, ...p.paths.deny]) {
    if (!(isAbsolute(v) || v.startsWith('~/')) || v.includes('*') || v.split(/[\\/]/).some(s => s === '.' || s === '..')) throw new Error('Access paths must be absolute or ~/ directory roots')
  }
  return structuredClone(p)
}

export function normalizeAccessUrl(value: string): { origin: string; path: string } {
  if (/[\u0000-\u0020\u007f\\]/.test(value) || /%(?:2f|5c)/i.test(value)) throw new Error('Invalid access URL')
  const u = new URL(value.includes('://') ? value : `https://${value}`)
  if (!['http:', 'https:'].includes(u.protocol) || u.username || u.password || u.hash || value.includes('*')) throw new Error('Invalid access URL')
  const path = decodeURIComponent(u.pathname).replace(/\/+$/, '') || '/'
  if (/[\u0000-\u001f\u007f]/.test(path)) throw new Error('Invalid access URL path')
  return { origin: u.origin, path }
}

function urlWithin(candidate: ReturnType<typeof normalizeAccessUrl>, root: string): boolean {
  const base = normalizeAccessUrl(root)
  return candidate.origin === base.origin && (base.path === '/' || candidate.path === base.path || candidate.path.startsWith(`${base.path}/`))
}

export function checkAccessUrl(policy: ResourceAccessPolicy | undefined, value: string, repository = false): PolicyCheckResult | undefined {
  if (!policy?.enabled) return undefined
  try {
    const normalize = (v: string) => { if (!repository) return v; const u = new URL(v.includes('://') ? v : `https://${v}`); u.pathname = u.pathname.replace(/\.git\/?$/, ''); return u.href }
    const u = normalizeAccessUrl(normalize(value))
    if (policy.urls.deny.some(v => urlWithin(u, normalize(v)))) return block('URL is explicitly denied')
    if (!policy.urls.allow.some(v => urlWithin(u, normalize(v)))) return block('URL is outside the allow list')
  } catch { return block('URL could not be safely validated') }
  return undefined
}

function canonicalPath(value: string): string {
  let path = resolve(value)
  const tail: string[] = []
  for (;;) {
    try { return resolve(realpathSync(path), ...tail.reverse()) } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error
      try { if (lstatSync(path).isSymbolicLink()) throw new Error('Dangling symlink') } catch (e) { if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e }
      const parent = dirname(path)
      if (parent === path) throw new Error('Cannot inspect path')
      tail.push(path.slice(parent.length).replace(/^[/\\]/, '')); path = parent
    }
  }
}
function within(candidate: string, root: string): boolean {
  const r = relative(root, candidate)
  return r === '' || (!r.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`) && r !== '..' && !isAbsolute(r))
}
function pathFailure(policy: ResourceAccessPolicy, value: string, cwd?: string, recursive = false, dataDir?: string): PolicyCheckResult | undefined {
  try {
    if (!value || value.includes('\0')) return block('Path could not be safely validated')
    const lexical = resolve(resolveToolPath(value, cwd))
    const canonical = canonicalPath(lexical)
    const denyRoots = policy.paths.deny.map(v => resolve(resolveToolPath(v)))
    const protectedRoots = dataDir ? [resolve(dataDir)] : []
    if ([lexical, canonical].some(p => p.split(/[\\/]/).some(s => s === '.ssh' || s === '.sepilotd') || /[/\\]\.git[/\\](?:hooks(?:[/\\]|$)|config$)/.test(p))) return block('Runtime state and credential paths are protected')
    for (const root of [...denyRoots, ...protectedRoots]) {
      const realRoot = canonicalPath(root)
      if (within(lexical, root) || within(canonical, realRoot) || recursive && (within(root, lexical) || within(realRoot, canonical))) return block('Path is explicitly denied or intersects a denied subtree')
    }
    if (!policy.paths.allow.some(v => {
      const root = resolve(resolveToolPath(v))
      return within(lexical, root) && within(canonical, canonicalPath(root))
    })) return block('Path is outside the allow list')
  } catch { return block('Path could not be safely validated') }
  return undefined
}

export function checkResourceAccess(policy: ResourceAccessPolicy | undefined, request: ToolExecRequest, dataDir?: string): PolicyCheckResult | undefined {
  if (!policy?.enabled) return undefined
  const { tool, input, cwd } = request
  if (policy.tools.deny.some(p => accessToolMatches(p, tool))) return block(`Tool '${tool}' is explicitly denied`)
  if (INTERNAL_TOOLS.has(tool) || policy.internalTools?.includes(tool)) return undefined
  if (!policy.tools.allow.some(p => accessToolMatches(p, tool))) return block(`Tool '${tool}' is outside the allow list`)
  if (tool === 'git.transfer') {
    const urlDenied = checkAccessUrl(policy, String(input.url ?? ''), true)
    if (urlDenied) return urlDenied
  }
  if (tool === 'webfetch') return checkAccessUrl(policy, String(input.url ?? ''))
  if (!FILE_TOOLS.has(tool) && !GIT_TOOLS.has(tool)) return block(`Tool '${tool}' has no verifiable resource boundary; arbitrary execution, browser, recursive search and external MCP are unavailable`)
  if (input.temporary === true) return block('Temporary writes require an explicitly bounded scratch capability')
  let targets: string[]
  try {
    if (tool === 'apply_patch') targets = parseApplyPatch(String(input.patch ?? '')).files.map(f => f.path)
    else if (tool === 'fs.move') targets = [String(input.from ?? ''), String(input.to ?? '')]
    else if (tool === 'fs.list') targets = [resolveToolCwd(input.cwd, cwd)]
    else if (GIT_TOOLS.has(tool)) {
      const repo = resolveToolCwd(input.repoPath, cwd)
      targets = [repo]
      if (typeof input.path === 'string' && input.path) targets.push(resolveToolPath(input.path, repo))
      if (tool === 'git.transfer' && input.operation === 'clone') {
        const denied = pathFailure(policy, repo, cwd, true, dataDir)
        return denied
      }
      const git = resolve(repo, '.git')
      if (lstatSync(git).isFile()) {
        const match = /^gitdir: (.+)\s*$/.exec(readFileSync(git, 'utf8'))
        if (!match) return block('Git metadata could not be inspected')
        targets.push(resolve(repo, match[1]!))
      } else targets.push(git)
      const common = resolve(targets[targets.length - 1]!, 'commondir')
      try { targets.push(resolve(dirname(common), readFileSync(common, 'utf8').trim())) } catch (e) { if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e }
    } else targets = [String(input.path ?? '')]
  } catch { return block('Resource targets could not be safely inspected') }
  if (!targets.length) return block('No resource targets were supplied')
  for (const target of targets) {
    const denied = pathFailure(policy, target, cwd, GIT_TOOLS.has(tool) || tool === 'fs.move', dataDir)
    if (denied) return denied
  }
  return undefined
}
