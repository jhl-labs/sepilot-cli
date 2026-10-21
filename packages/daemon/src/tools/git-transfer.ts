import { createEgressPolicy } from '../utils/egress-policy.js'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { access, lstat, readFile, readdir } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'
import { resolveToolPath } from './path-utils.js'
import { enforceOutputLimit } from './output-limit.js'

const exec = promisify(execFile)
const SAFE_CONFIG = ['core.hooksPath', process.platform === 'win32' ? 'NUL' : '/dev/null', 'core.fsmonitor', 'false', 'credential.helper', '', 'credential.interactive', 'false', 'http.followRedirects', 'false', 'http.proxy', '', 'protocol.allow', 'never', 'protocol.https.allow', 'always', 'protocol.http.allow', 'always', 'fetch.recurseSubmodules', 'false', 'push.recurseSubmodules', 'no', 'push.gpgSign', 'false', 'gc.auto', '0', 'maintenance.auto', 'false'].flatMap((v, i, a) => i % 2 === 0 ? ['-c', `${v}=${a[i + 1]}`] : [])
const SAFE_ENV = { PATH: process.env.PATH, ...(process.platform === 'win32' ? { SystemRoot: process.env.SystemRoot } : {}), GIT_CONFIG_NOSYSTEM: '1', GIT_CONFIG_GLOBAL: process.platform === 'win32' ? 'NUL' : '/dev/null', GIT_TERMINAL_PROMPT: '0', GIT_OPTIONAL_LOCKS: '0' }

export function boundedGitEnvironment(): NodeJS.ProcessEnv { return { ...SAFE_ENV } }

export async function inspectBoundedGitConfiguration(repo: string): Promise<void> {
  const git = join(repo, '.git')
  if (!(await lstat(git)).isDirectory()) throw new Error('Git transfer requires a regular repository with its own .git directory')
  const pending = [git]
  let inspected = 0
  while (pending.length) {
    const directory = pending.pop()!
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      if (++inspected > 100000) throw new Error('Git metadata is too large to validate within the bounded operation')
      if (entry.isSymbolicLink()) throw new Error('Symlinked Git metadata is unavailable for bounded operations')
      if (entry.isDirectory()) pending.push(join(directory, entry.name))
    }
  }
  const { stdout } = await exec('git', ['-C', repo, 'config', '--local', '--no-includes', '--null', '--list'], { env: SAFE_ENV, timeout: 5000 })
  for (const record of stdout.split('\0').filter(Boolean)) {
    const key = record.split('\n')[0]!.toLowerCase()
    if (/^(?:url\.|http\.|https\.|include\.|includeif\.|credential\.|filter\.|protocol\.|extensions\.|core\.(?:sshcommand|gitproxy|worktree|excludesfile|attributesfile)$|remote\..*\.vcs$)/.test(key)) throw new Error(`Git configuration '${key}' requires review before network transfer`)
  }
  for (const path of ['objects/info/alternates', 'objects/info/http-alternates']) {
    try { if ((await readFile(join(git, path), 'utf8')).trim()) throw new Error('Git alternate object stores are unavailable for bounded transfers') } catch (e) { if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e }
  }
}

export function createGitTransferTool(options: { egressAllowlist?: readonly string[] | null } = {}): ToolDefinitionRuntime {
  const egressPolicy = createEgressPolicy({ allowlist: options.egressAllowlist })
  return {
    name: 'git.transfer',
    description: 'Run a structured HTTP(S) Git clone, fetch or non-force push against an explicit repository URL. Access policy checks the Git tool, URL and local folder together. No shell, named remote, redirect, hooks, credential helper, submodule recursion or checkout filters. Clone creates a repository without checking out files. Worktrees and repositories with symlinked/alternate metadata or executable/network configuration are unsupported. Authentication helpers and SSH are unsupported: private repositories may require a separately reviewed credential capability. Network operations are not replay safe; verify local/remote state after interruptions.',
    security: { effect: 'external-write', rationale: 'Mutates a local repository or publishes commits to an explicit remote', reach: 'network' },
    resumeSafety: 'replay-risky',
    inputSchema: { type: 'object', additionalProperties: false, required: ['operation', 'repoPath', 'url'], properties: {
      operation: { type: 'string', enum: ['clone', 'fetch', 'push'] },
      repoPath: { type: 'string', description: 'Local repository directory, or new clone destination' },
      url: { type: 'string', description: 'Explicit HTTP(S) repository URL without credentials or query parameters' },
      branch: { type: 'string', description: 'Required branch for fetch/push; ordinary branch names only' },
    } },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const fail = (output: string): ToolResult => ({ output, status: 'error', code: 'GIT_TRANSFER_REJECTED_PERMANENT', durationMs: Date.now() - start })
      if (context?.workspaceRoot) return fail('Git transfer requires a separately supported strict workspace/network sandbox')
      if (Object.keys(input).some(k => !['operation', 'repoPath', 'url', 'branch'].includes(k))) return fail('Unsupported Git transfer input')
      const op = input.operation
      if (!['clone', 'fetch', 'push'].includes(String(op)) || typeof input.repoPath !== 'string' || !input.repoPath.trim() || input.repoPath.includes('\0')) return fail('Invalid Git operation or local folder')
      let url: URL
      try {
        url = new URL(String(input.url))
        if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || /[\s\\]/.test(String(input.url))) return fail('Provide a credential-free HTTP(S) repository URL')
      } catch { return fail('Provide an explicit HTTP(S) repository URL') }
      try { egressPolicy.assertAllowed(url.href, 'git.transfer') } catch (e) { return fail(e instanceof Error ? e.message : String(e)) }
      const branch = typeof input.branch === 'string' ? input.branch : ''
      if (op !== 'clone' && (!/^[A-Za-z0-9][A-Za-z0-9_./-]{0,199}$/.test(branch) || branch.includes('..') || branch.includes('//') || branch.endsWith('/') || branch.endsWith('.lock'))) return fail('Provide a valid branch name; options and revision expressions are not accepted')
      const repo = resolve(resolveToolPath(input.repoPath, context?.cwd))
      try {
        let args: string[]
        if (op === 'clone') {
          try { await access(repo); return fail('Clone destination must not already exist') } catch (e) { if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e }
          args = [...SAFE_CONFIG, 'clone', '--no-checkout', '--no-recurse-submodules', '--', url.href, repo]
        } else {
          await inspectBoundedGitConfiguration(repo)
          args = [...SAFE_CONFIG, '-C', repo, ...(op === 'fetch' ? ['fetch', '--no-recurse-submodules', '--', url.href, `refs/heads/${branch}`] : ['push', '--no-verify', '--', url.href, `HEAD:refs/heads/${branch}`])]
        }
        const { stdout, stderr } = await exec('git', args, { cwd: op === 'clone' ? dirname(repo) : repo, env: SAFE_ENV, signal: context?.signal, timeout: 60000, maxBuffer: 1024 * 1024 })
        return { output: enforceOutputLimit(stdout || stderr || '[Git transfer completed]', { toolName: 'git.transfer' }).output, status: 'success', durationMs: Date.now() - start }
      } catch (e) { return fail(e instanceof Error ? e.message : String(e)) }
    },
  }
}
