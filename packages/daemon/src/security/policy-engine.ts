import type {
  AutonomyLevel,
  IToolPolicy,
  ToolExecRequest,
  PolicyCheckResult,
  PolicyConfig,
  PolicyRule,
  ToolPolicyMode,
} from '@sepilotd/core'

/**
 * Tool-rule mode as understood by the daemon policy engine.
 *
 * - `autonomous`: run without prompting.
 * - `supervised`: prompt when the run autonomy prompts (supervised,
 *   accept-edits, workspace-write); under `autonomous` autonomy the tool
 *   runs without prompting once every deny gate has passed.
 * - `ask`: always prompt, regardless of run autonomy. Use it for the few
 *   tools that must have a human in the loop even in headless runs.
 * - `blocked`: never run.
 *
 * `ask` is a daemon-side extension of the core `ToolPolicyMode` union so a
 * policy file can opt individual tools into "always ask" without changing
 * what `supervised` means for interactive runs.
 */
export type ToolRuleMode = ToolPolicyMode | 'ask'

export type PolicyEngineRule = Omit<PolicyRule, 'mode'> & { mode: ToolRuleMode }

export interface PolicyEngineDefaults {
  mode: ToolRuleMode
  unmatched_policy: PolicyConfig['defaults']['unmatched_policy']
  max_timeout_ms: number
  max_output_bytes: number
  /**
   * Escape hatch for operators: when `true`, `autonomous` autonomy treats a
   * `supervised` tool rule as approval-required (which the autonomous
   * executor turns into a hard block), restoring the pre-`ask` behaviour.
   * Default `false`: autonomous means "do not prompt", not "block more".
   */
  autonomous_honors_supervised_rules?: boolean
}

/** `PolicyConfig` widened with the daemon-only `ask` mode and defaults knob.
 * Every core `PolicyConfig` is assignable to it, so existing callers and
 * loaders need no changes. */
export interface PolicyEngineConfig {
  version: number
  defaults: PolicyEngineDefaults
  tools: Record<string, PolicyEngineRule>
  elevated?: Record<string, Partial<PolicyEngineRule>>
}

export const AUTONOMOUS_SUPERVISED_RULE_REASON =
  'Autonomous mode runs supervised-rule tools without prompting'
import { lstatSync, readlinkSync } from 'node:fs'
import { basename, dirname, extname, isAbsolute, parse, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseApplyPatch } from '../tools/apply-patch-parser.js'
import { resolvePagesScaffoldPolicyTargets } from '../tools/pages-scaffold-plan.js'
import { resolveToolCwd, resolveToolPath } from '../tools/path-utils.js'
import { resolveToolSecurityDescriptor } from '../tools/security.js'
import {
  isReadOnlyNetworkTerminalCommand,
  normalizeTerminalArgs,
  normalizeTerminalCommandShape,
} from '../tools/terminal.js'

/**
 * Simple glob matching supporting * (single segment) and ** (any depth).
 * Converts a glob pattern to a RegExp.
 */
function globToRegex(pattern: string): RegExp {
  const escaped = pattern
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/\*\*/g, '\u0000')
    .replace(/\*/g, '[^/]*')
    .replace(/\u0000/g, '.*')
  return new RegExp(`^${escaped}$`)
}

function normalizePolicyPath(value: string): string {
  return value.replace(/\\/g, '/')
}

function matchesAny(value: string, patterns: string[]): boolean {
  return patterns.some((p) => globToRegex(p).test(value))
}

function matchesPathAny(value: string, patterns: string[]): boolean {
  const normalizedValue = normalizePolicyPath(value)
  return patterns.some((pattern) => globToRegex(normalizePolicyPath(pattern)).test(normalizedValue))
}

/**
 * Like {@link matchesAny} but matches the pattern *anywhere inside* the value.
 * Used for wrapper-script content where the dangerous fragment may sit between
 * other shell tokens (e.g. `cd /tmp && rm -rf /` shouldn't slip past a
 * `rm -rf /` deny_pattern just because the script also has a `cd` prefix).
 */
function containsAny(value: string, patterns: string[]): boolean {
  return patterns.some((pattern) => {
    const inner = globToRegex(pattern).source.replace(/^\^/, '').replace(/\$$/, '')
    try {
      return new RegExp(inner).test(value)
    } catch {
      return false
    }
  })
}

// Download-pipe-to-shell detection. The glob deny_patterns `curl * | sh`
// / `wget * | sh` use a path-segment glob (`*` → `[^/]*`), so a URL with
// slashes (the common case) slips past them. This regex catches a
// fetch-then-pipe-to-interpreter chain regardless of slashes in the URL.
const DOWNLOAD_PIPE_TO_SHELL =
  /\b(?:curl|wget|fetch)\b[\s\S]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|dash|ksh|fish|python3?|perl|ruby|node)\b/i

function isDownloadPipeToShell(value: string): boolean {
  return DOWNLOAD_PIPE_TO_SHELL.test(value)
}

function isShellCommandFlag(arg: string): boolean {
  return arg === '-c' || /^-[A-Za-z]*c[A-Za-z]*$/.test(arg)
}

/**
 * If the request is a shell wrapper (`bash -c "<script>"`, `sh -c "<script>"`,
 * etc.), return the script body so the policy can scan it independently of
 * the wrapper invocation. Returns `null` when no `-c` script body is present.
 */
function extractWrappedScript(input: Record<string, unknown>): string | null {
  const normalized = normalizeTerminalCommandShape(input)
  const executable = typeof normalized.executable === 'string' ? normalized.executable : undefined
  const args = normalizeTerminalArgs(normalized.args, executable)
  for (let i = 0; i < args.length; i++) {
    if (isShellCommandFlag(args[i]!) && typeof args[i + 1] === 'string') {
      return args[i + 1]
    }
  }
  return null
}

function normalizeExecutable(rawExecutable: unknown): string {
  if (typeof rawExecutable !== 'string') {
    return ''
  }

  return basename(rawExecutable).trim().toLowerCase()
}

function isInsideOrEqual(path: string, root: string): boolean {
  const relativePath = relative(resolve(root), resolve(path))
  return (
    relativePath === '' ||
    (!!relativePath && !relativePath.startsWith('..') && !isAbsolute(relativePath))
  )
}

type BoundaryPathResult = { path: string } | { error: string }

const MAX_BOUNDARY_SYMLINK_DEPTH = 32

function canonicalBoundaryPath(rawPath: string, symlinkDepth = 0): BoundaryPathResult {
  if (symlinkDepth > MAX_BOUNDARY_SYMLINK_DEPTH) {
    return { error: `Too many symbolic links while resolving ${rawPath}` }
  }

  const absolutePath = resolve(rawPath)
  const root = parse(absolutePath).root
  const segments = relative(root, absolutePath)
    .split(/[\\/]+/)
    .filter(Boolean)
  let current = root

  for (let index = 0; index < segments.length; index++) {
    const segment = segments[index]!
    const candidate = resolve(current, segment)
    let stat
    try {
      stat = lstatSync(candidate)
    } catch (error) {
      const code =
        typeof error === 'object' && error && 'code' in error
          ? String((error as { code?: unknown }).code)
          : ''
      if (code && code !== 'ENOENT' && code !== 'ENOTDIR') {
        return {
          error: `Could not inspect path ${candidate}: ${
            error instanceof Error ? error.message : String(error)
          }`,
        }
      }
      return { path: resolve(current, segment, ...segments.slice(index + 1)) }
    }

    if (!stat.isSymbolicLink()) {
      current = candidate
      continue
    }

    let linkTarget
    try {
      linkTarget = readlinkSync(candidate)
    } catch (error) {
      return {
        error: `Could not inspect symbolic link ${candidate}: ${
          error instanceof Error ? error.message : String(error)
        }`,
      }
    }
    const resolvedTarget = isAbsolute(linkTarget)
      ? linkTarget
      : resolve(dirname(candidate), linkTarget)
    return canonicalBoundaryPath(
      resolve(resolvedTarget, ...segments.slice(index + 1)),
      symlinkDepth + 1,
    )
  }

  return { path: current }
}

/**
 * Canonical, symlink-aware containment check shared by tools whose effective
 * file target is runtime state rather than a model-visible path argument.
 */
export function strictWorkspacePathViolation(
  rawPath: string,
  rawWorkspaceRoot: string,
): string | null {
  const workspaceRoot = canonicalBoundaryPath(rawWorkspaceRoot)
  if ('error' in workspaceRoot) {
    return 'Strict workspace root could not be safely validated'
  }
  if (relative(resolve(rawWorkspaceRoot), workspaceRoot.path) !== '') {
    return 'Strict workspace root changed after it was selected; reselect the workspace before continuing'
  }
  const target = canonicalBoundaryPath(rawPath)
  if ('error' in target) {
    return 'Strict workspace target could not be safely validated'
  }
  return isInsideOrEqual(target.path, workspaceRoot.path)
    ? null
    : 'Strict workspace blocks access: the target resolves outside workspace boundary'
}

const SHELL_COMMAND_SEPARATORS = new Set([';', '&&', '||', '|', '&'])

function tokenizeShellForPolicy(script: string): string[] {
  const tokens: string[] = []
  let current = ''
  let quote: '"' | "'" | null = null
  let escaped = false

  const flush = () => {
    if (current) {
      tokens.push(current)
      current = ''
    }
  }

  for (let index = 0; index < script.length; index++) {
    const char = script[index]!

    if (escaped) {
      current += char
      escaped = false
      continue
    }

    if (char === '\\') {
      escaped = true
      continue
    }

    if (quote) {
      if (char === quote) {
        quote = null
      } else {
        current += char
      }
      continue
    }

    if (char === '"' || char === "'") {
      quote = char
      continue
    }

    if (/\s/.test(char)) {
      flush()
      continue
    }

    if (char === '&' && script[index + 1] === '>') {
      flush()
      let operator = '&>'
      index += 1
      if (script[index + 1] === '>') {
        operator = '&>>'
        index += 1
      }
      tokens.push(operator)
      continue
    }

    if (char === '&' && script[index + 1] === '&') {
      flush()
      tokens.push('&&')
      index += 1
      continue
    }

    if (char === '|' && script[index + 1] === '|') {
      flush()
      tokens.push('||')
      index += 1
      continue
    }

    if (char === ';' || char === '|' || char === '&') {
      flush()
      tokens.push(char)
      continue
    }

    if (char === '>' || char === '<') {
      let operator = char
      if (/^\d+$/.test(current)) {
        operator = `${current}${operator}`
        current = ''
      } else {
        flush()
      }
      if (script[index + 1] === char) {
        operator = `${operator}${char}`
        index += 1
      } else if (char === '>' && script[index + 1] === '|') {
        operator = `${operator}|`
        index += 1
      }
      tokens.push(operator)
      continue
    }

    current += char
  }

  flush()
  return tokens
}

function isOutputRedirectionToken(token: string): boolean {
  return /^(?:\d+)?(?:>>?|>\|)$/.test(token) || /^&>>?$/.test(token)
}

function isAnyRedirectionToken(token: string): boolean {
  return isOutputRedirectionToken(token) || /^(?:\d+)?<<?$/.test(token)
}

function shouldInspectShellPathTarget(raw: string): boolean {
  const target = raw.trim()
  if (!target || target === '-' || target.startsWith('-')) return false
  if (target === '/dev/null' || target === '/dev/stdout' || target === '/dev/stderr') return false
  if (/^&?\d+$/.test(target)) return false
  if (
    target.startsWith('$') ||
    target.startsWith('`') ||
    target.startsWith('<(') ||
    target.startsWith('>(')
  ) {
    return false
  }
  return true
}

function resolveShellPathTarget(raw: string, baseCwd: string): string | null {
  return shouldInspectShellPathTarget(raw) ? resolveToolPath(raw, baseCwd) : null
}

function collectOutputRedirectionTargets(tokens: string[], baseCwd: string): string[] {
  const targets: string[] = []
  for (let index = 0; index < tokens.length; index++) {
    if (!isOutputRedirectionToken(tokens[index]!)) continue
    const target = tokens[index + 1]
    if (!target || SHELL_COMMAND_SEPARATORS.has(target) || isAnyRedirectionToken(target)) continue
    const resolved = resolveShellPathTarget(target, baseCwd)
    if (resolved) targets.push(resolved)
  }
  return targets
}

function stripRedirectionTokens(tokens: string[]): string[] {
  const stripped: string[] = []
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]!
    if (isAnyRedirectionToken(token)) {
      const next = tokens[index + 1]
      if (next && !SHELL_COMMAND_SEPARATORS.has(next) && !isAnyRedirectionToken(next)) {
        index += 1
      }
      continue
    }
    stripped.push(token)
  }
  return stripped
}

function splitShellCommandSegments(tokens: string[]): string[][] {
  const segments: string[][] = []
  let current: string[] = []
  for (const token of tokens) {
    if (SHELL_COMMAND_SEPARATORS.has(token)) {
      if (current.length > 0) segments.push(current)
      current = []
      continue
    }
    current.push(token)
  }
  if (current.length > 0) segments.push(current)
  return segments
}

function commandTokens(tokens: string[]): { command: string; args: string[] } | null {
  let index = 0
  while (tokens[index] && /^[A-Za-z_][A-Za-z0-9_]*=/.test(tokens[index]!)) {
    index += 1
  }
  const command = tokens[index]
  if (!command) return null
  return {
    command: normalizeExecutable(command),
    args: tokens.slice(index + 1),
  }
}

function positionalArgs(
  args: string[],
  options: { optionValueFlags?: Set<string> } = {},
): string[] {
  const positions: string[] = []
  let literal = false
  for (let index = 0; index < args.length; index++) {
    const arg = args[index]!
    if (!literal && arg === '--') {
      literal = true
      continue
    }
    if (!literal && options.optionValueFlags?.has(arg)) {
      index += 1
      continue
    }
    if (!literal && arg.startsWith('-')) {
      continue
    }
    positions.push(arg)
  }
  return positions
}

function collectMutatingCommandTargets(tokens: string[], baseCwd: string): string[] {
  const cleaned = stripRedirectionTokens(tokens)
  const parsed = commandTokens(cleaned)
  if (!parsed) return []

  const { command, args } = parsed
  const targets: string[] = []
  const addTarget = (raw: string | undefined) => {
    if (!raw) return
    const resolved = resolveShellPathTarget(raw, baseCwd)
    if (resolved) targets.push(resolved)
  }

  if (command === 'tee') {
    for (const target of positionalArgs(args)) addTarget(target)
    return targets
  }

  if (command === 'touch' || command === 'mkdir' || command === 'rm' || command === 'rmdir') {
    for (const target of positionalArgs(args)) addTarget(target)
    return targets
  }

  if (command === 'cp' || command === 'mv') {
    const targetDirectoryIndex = args.findIndex(
      (arg) => arg === '-t' || arg === '--target-directory',
    )
    if (targetDirectoryIndex >= 0) {
      addTarget(args[targetDirectoryIndex + 1])
      return targets
    }
    const inlineTargetDirectory = args.find((arg) => arg.startsWith('--target-directory='))
    if (inlineTargetDirectory) {
      addTarget(inlineTargetDirectory.slice('--target-directory='.length))
      return targets
    }
    const positions = positionalArgs(args, {
      optionValueFlags: new Set(['-t', '--target-directory']),
    })
    addTarget(positions.at(-1))
    return targets
  }

  if (command === 'dd') {
    for (const arg of args) {
      if (arg.startsWith('of=')) addTarget(arg.slice(3))
    }
    return targets
  }

  targets.push(...collectInterpreterSnippetTargets(command, args, baseCwd))
  return targets
}

const INLINE_EVAL_FLAGS = new Set(['-c', '-e', '--eval'])

function extractInlineFlagValue(args: string[], flags: Set<string>): string | null {
  for (let index = 0; index < args.length; index++) {
    const arg = args[index]!
    if (flags.has(arg)) {
      return typeof args[index + 1] === 'string' ? args[index + 1]! : null
    }
    for (const flag of flags) {
      if (
        arg.startsWith(flag) &&
        arg.length > flag.length &&
        flag.startsWith('-') &&
        flag.length === 2
      ) {
        return arg.slice(flag.length)
      }
    }
  }
  return null
}

function extractDenoEvalScript(args: string[]): string | null {
  const evalIndex = args.findIndex((arg) => arg === 'eval')
  return evalIndex >= 0 && typeof args[evalIndex + 1] === 'string' ? args[evalIndex + 1]! : null
}

function decodePolicyStringLiteral(value: string): string {
  return value.replace(/\\([\\'"`nrt])/g, (_match, escaped: string) => {
    switch (escaped) {
      case 'n':
        return '\n'
      case 'r':
        return '\r'
      case 't':
        return '\t'
      default:
        return escaped
    }
  })
}

function isWriteMode(mode: string | undefined): boolean {
  return !!mode && /[wax+]/i.test(mode)
}

function collectPythonInlineWriteTargets(script: string, baseCwd: string): string[] {
  const targets: string[] = []
  const addTarget = (raw: string | undefined) => {
    if (!raw) return
    const resolved = resolveShellPathTarget(decodePolicyStringLiteral(raw), baseCwd)
    if (resolved) targets.push(resolved)
  }

  const stringPattern = String.raw`(?:"((?:\\.|[^"\\])*)"|'((?:\\.|[^'\\])*)')`
  const openPattern = new RegExp(
    String.raw`(?:^|[^\w.])(?:open|io\.open)\s*\(\s*${stringPattern}\s*(?:,\s*${stringPattern})?`,
    'g',
  )
  let match: RegExpExecArray | null
  while ((match = openPattern.exec(script)) !== null) {
    const path = match[1] ?? match[2]
    const mode = match[3] ?? match[4]
    if (isWriteMode(mode)) addTarget(path)
  }

  const pathlibWritePattern = new RegExp(
    String.raw`(?:Path|pathlib\.Path)\s*\(\s*${stringPattern}\s*\)\s*\.\s*(?:write_text|write_bytes|touch|mkdir)\s*\(`,
    'g',
  )
  while ((match = pathlibWritePattern.exec(script)) !== null) {
    addTarget(match[1] ?? match[2])
  }

  const pathlibOpenPattern = new RegExp(
    String.raw`(?:Path|pathlib\.Path)\s*\(\s*${stringPattern}\s*\)\s*\.\s*open\s*\(\s*${stringPattern}`,
    'g',
  )
  while ((match = pathlibOpenPattern.exec(script)) !== null) {
    const path = match[1] ?? match[2]
    const mode = match[3] ?? match[4]
    if (isWriteMode(mode)) addTarget(path)
  }

  const osWritePattern = new RegExp(
    String.raw`(?:^|[^\w.])(?:os\.)?(?:makedirs|mkdir|remove|unlink|rmdir)\s*\(\s*${stringPattern}`,
    'g',
  )
  while ((match = osWritePattern.exec(script)) !== null) {
    addTarget(match[1] ?? match[2])
  }

  return targets
}

function collectJavaScriptInlineWriteTargets(script: string, baseCwd: string): string[] {
  const targets: string[] = []
  const addTarget = (raw: string | undefined) => {
    if (!raw || raw.includes('${')) return
    const resolved = resolveShellPathTarget(decodePolicyStringLiteral(raw), baseCwd)
    if (resolved) targets.push(resolved)
  }

  const stringPattern =
    '(?:"((?:\\\\.|[^"\\\\])*)"|\'((?:\\\\.|[^\'\\\\])*)\'|`((?:\\\\.|[^`\\\\])*)`)'
  const firstPathPattern = new RegExp(
    String.raw`(?:writeFileSync|writeFile|appendFileSync|appendFile|createWriteStream|rmSync|unlinkSync|mkdirSync|rm|unlink|mkdir|Deno\.writeTextFileSync|Deno\.writeTextFile|Deno\.writeFileSync|Deno\.writeFile|Bun\.write)\s*\(\s*${stringPattern}`,
    'g',
  )
  let match: RegExpExecArray | null
  while ((match = firstPathPattern.exec(script)) !== null) {
    addTarget(match[1] ?? match[2] ?? match[3])
  }

  const secondPathPattern = new RegExp(
    String.raw`(?:cpSync|copyFileSync|renameSync|cp|copyFile|rename)\s*\(\s*${stringPattern}\s*,\s*${stringPattern}`,
    'g',
  )
  while ((match = secondPathPattern.exec(script)) !== null) {
    addTarget(match[4] ?? match[5] ?? match[6])
  }

  const openPattern = new RegExp(
    String.raw`(?:openSync|open)\s*\(\s*${stringPattern}\s*,\s*${stringPattern}`,
    'g',
  )
  while ((match = openPattern.exec(script)) !== null) {
    const path = match[1] ?? match[2] ?? match[3]
    const mode = match[4] ?? match[5] ?? match[6]
    if (isWriteMode(mode)) addTarget(path)
  }

  return targets
}

function collectInterpreterSnippetTargets(
  command: string,
  args: string[],
  baseCwd: string,
): string[] {
  if (/^python(?:\d+(?:\.\d+)?)?$/.test(command) || command === 'py') {
    const script = extractInlineFlagValue(args, new Set(['-c']))
    return script ? collectPythonInlineWriteTargets(script, baseCwd) : []
  }

  if (command === 'node' || command === 'nodejs' || command === 'bun') {
    const script = extractInlineFlagValue(args, INLINE_EVAL_FLAGS)
    return script ? collectJavaScriptInlineWriteTargets(script, baseCwd) : []
  }

  if (command === 'deno') {
    const script = extractDenoEvalScript(args)
    return script ? collectJavaScriptInlineWriteTargets(script, baseCwd) : []
  }

  return []
}

function collectShellWriteTargets(script: string, baseCwd: string): string[] {
  const tokens = tokenizeShellForPolicy(script)
  const targets = collectOutputRedirectionTargets(tokens, baseCwd)
  for (const segment of splitShellCommandSegments(tokens)) {
    targets.push(...collectMutatingCommandTargets(segment, baseCwd))
  }
  return targets
}

function terminalWorkspaceTargets(input: Record<string, unknown>, workspaceRoot: string): string[] {
  const normalized = normalizeTerminalCommandShape(input)
  const terminalCwd = resolveToolCwd(normalized.cwd, workspaceRoot)
  const executable = typeof normalized.executable === 'string' ? normalized.executable : ''
  const args = normalizeTerminalArgs(normalized.args, executable)
  const script = SHELL_WRAPPER_EXECUTABLES.has(normalizeExecutable(executable))
    ? extractWrappedScript(normalized)
    : null
  return [
    terminalCwd,
    ...(script
      ? collectShellWriteTargets(script, terminalCwd)
      : collectMutatingCommandTargets([executable, ...args], terminalCwd)),
  ]
}

function workspaceTargetsForRequest(
  tool: string,
  input: Record<string, unknown>,
  cwd: string | undefined,
): { paths: string[] } | { error: string } {
  if (tool === 'fs.write' || tool === 'fs.append' || tool === 'fs.edit') {
    return typeof input.path === 'string' && input.path.trim()
      ? { paths: [resolveToolPath(input.path, cwd)] }
      : { paths: [] }
  }

  if (tool === 'workspace.prepare') {
    return typeof input.path === 'string' && input.path.trim()
      ? { paths: [resolveToolPath(input.path, cwd)] }
      : { paths: [] }
  }

  // A move mutates both ends: the source disappears and the destination
  // appears, so both are workspace-write targets.
  if (tool === 'fs.move') {
    const paths: string[] = []
    for (const key of ['from', 'to'] as const) {
      const value = input[key]
      if (typeof value === 'string' && value.trim()) paths.push(resolveToolPath(value, cwd))
    }
    return { paths }
  }

  if (tool === 'apply_patch') {
    try {
      const parsed = parseApplyPatch(String(input.patch ?? ''))
      return { paths: parsed.files.map((file) => resolveToolPath(file.path, cwd)) }
    } catch (error) {
      return {
        error: `WorkspaceWrite mode could not inspect apply_patch paths: ${
          error instanceof Error ? error.message : String(error)
        }`,
      }
    }
  }

  if (tool === 'terminal.run') {
    return { paths: terminalWorkspaceTargets(input, cwd ?? process.cwd()) }
  }

  return { paths: [] }
}

const STRICT_WORKSPACE_PATH_TOOLS = new Set([
  'workspace.prepare',
  'fs.read',
  'fs.write',
  'fs.append',
  'fs.edit',
  'fs.move',
  'media.inspect',
  'media.extract_text',
  'media.transcribe',
  'notebook.inspect',
  'code.dependencies',
  'pages.scan',
  'browser.screenshot',
  'browser.click',
  'browser.evaluate',
])

// These Office operations execute against the host PowerPoint process, but
// their public contract is narrower than the legacy office.* bridge: they take
// one .pptx path, canonicalize it before and after realpath, reject symlink
// escapes, and re-check the active presentation on every call.
const STRICT_WORKSPACE_SELF_CONFINED_OFFICE_TOOLS = new Set([
  'office.open_presentation',
  'office.navigate_slide',
  'office.read_slide',
  'office.capture_slide',
])

const STRICT_WORKSPACE_CWD_TOOLS = new Set([
  'fs.list',
  'fs.glob',
  'fs.search',
  'code.symbols',
])

// These tools execute Git only through a runner that must attest active
// workspace-only filesystem isolation. Their repoPath is still checked here
// and is rechecked immediately before execution by the tool itself.
const STRICT_WORKSPACE_SANDBOXED_GIT_TOOLS = new Set([
  'git.status',
  'git.diff',
  'git.log',
])

// terminal.run is permitted to reach the ordinary policy/approval gate only
// because the tool runtime replaces host execution with a fail-closed
// read-only, no-network workspace sandbox whenever workspaceRoot is present.
// The runner re-attests that posture after execution and never falls back to
// the host. Managed background children use the separate lifecycle set below.
const STRICT_WORKSPACE_SANDBOXED_TERMINAL_TOOLS = new Set(['terminal.run'])

// Managed process.start uses the same fail-closed bubblewrap launcher as
// terminal.run when a strict workspace is present. The remaining lifecycle
// tools operate only on daemon-owned session ids and captured buffers; they do
// not open arbitrary host paths. This is a capability boundary, not a command
// allowlist: the child executable remains subject to ordinary policy and
// approval, while the runtime attests the namespace posture.
const STRICT_WORKSPACE_MANAGED_PROCESS_TOOLS = new Set([
  'process.start',
  'process.sessions',
  'process.read',
  'process.follow',
  'process.wait',
  'process.stop',
])

const STRICT_WORKSPACE_UNCONFINED_TOOLS = new Set([
  'media.transcribe',
  'media.speak',
  'image_gen.create',
  'skillhub.install',
  'external_acp.run',
  'device.delegate',
  'a2a.send',
  'pages.status',
])

const STRICT_WORKSPACE_UNCONFINED_PREFIXES = [
  'computer.',
  'mcp.',
  'office.',
  'process.',
  'service.',
  'swarm.',
] as const

const STRICT_WORKSPACE_AUDITED_TOOLS = new Set([
  // Knowledge tools accept record IDs and content, never caller-selected paths.
  // They access only the daemon-owned profile store; write approval still applies.
  'knowledge.search',
  'knowledge.read',
  'knowledge.save',
  'knowledge.edit',
  'knowledge.review',
  'apply_patch',
  'browser.extract',
  'browser.navigate',
  // Companion exposes only HTTP(S) tab observation and fixed input actions.
  // No filesystem paths, uploads, arbitrary CDP, or script execution are accepted.
  // External-write approval and URL policy still apply below.
  'browser.remote_snapshot',
  'browser.remote_action',
  'image_gen.cancel',
  'image_gen.file',
  'image_gen.job',
  'image_gen.providers',
  'gitea.actions.runs.inspect',
  'market.quote',
  'jpad.workspaces',
  'jpad.pages.list',
  'jpad.pages.get',
  'jpad.pages.create',
  'jpad.pages.update',
  'pages.scaffold',
  'question',
  'notification.publish',
  'monitor.evaluate',
  'monitor.report',
  'schedule_create',
  'schedule_cancel',
  'schedule_cancel_all',
  'schedule_get',
  'schedule_list',
  'schedule_pause',
  'schedule_resume',
  'schedule_runs',
  'schedule_run_now',
  'schedule_update',
  'assistant.status',
  'self.info',
  // Trusted, structured host metrics only. system.info never accepts a file
  // read target and returns coarse resource values, so enabling the explicit
  // hostSystemInfo capability does not widen the workspace filesystem root.
  'system.info',
  'skill',
  'skillhub.search',
  'subagent.dispatch',
  'todowrite',
  'usage.report',
  'web.search',
  'webfetch',
])

const STRICT_WORKSPACE_INTERNAL_STATE_PREFIXES = [
  'apps.',
  'doc.',
  'memory.',
] as const

function hasAuditedStrictWorkspacePosture(tool: string): boolean {
  return STRICT_WORKSPACE_PATH_TOOLS.has(tool)
    || STRICT_WORKSPACE_SELF_CONFINED_OFFICE_TOOLS.has(tool)
    || STRICT_WORKSPACE_CWD_TOOLS.has(tool)
    || STRICT_WORKSPACE_SANDBOXED_GIT_TOOLS.has(tool)
    || STRICT_WORKSPACE_SANDBOXED_TERMINAL_TOOLS.has(tool)
    || STRICT_WORKSPACE_MANAGED_PROCESS_TOOLS.has(tool)
    || STRICT_WORKSPACE_AUDITED_TOOLS.has(tool)
    || STRICT_WORKSPACE_INTERNAL_STATE_PREFIXES.some((prefix) => tool.startsWith(prefix))
}

function strictWorkspaceTargetsForRequest(
  request: ToolExecRequest,
  workspaceRoot: string,
): { paths: string[] } | { error: string } {
  const baseCwd = request.cwd ?? workspaceRoot
  const { tool, input } = request

  if (STRICT_WORKSPACE_PATH_TOOLS.has(tool)) {
    return typeof input.path === 'string' && input.path.trim()
      ? { paths: [resolveToolPath(input.path, baseCwd)] }
      : { paths: [] }
  }

  if (STRICT_WORKSPACE_SELF_CONFINED_OFFICE_TOOLS.has(tool)) {
    return typeof input.path === 'string' && input.path.trim()
      ? { paths: [resolveToolPath(input.path, baseCwd)] }
      : { paths: [] }
  }

  if (STRICT_WORKSPACE_CWD_TOOLS.has(tool)) {
    return { paths: [resolveToolCwd(input.cwd, baseCwd)] }
  }

  if (STRICT_WORKSPACE_SANDBOXED_GIT_TOOLS.has(tool)) {
    const repoPath = typeof input.repoPath === 'string' && input.repoPath.trim()
      ? resolveToolPath(input.repoPath, baseCwd)
      : resolveToolPath(baseCwd)
    const paths = [repoPath]
    if (typeof input.path === 'string' && input.path.trim()) {
      paths.push(resolveToolPath(input.path, repoPath))
    }
    return { paths }
  }

  if (STRICT_WORKSPACE_SANDBOXED_TERMINAL_TOOLS.has(tool)) {
    return { paths: terminalWorkspaceTargets(input, baseCwd) }
  }

  if (tool === 'process.start') {
    return { paths: terminalWorkspaceTargets(input, baseCwd) }
  }

  if (tool === 'code.diagnostics' || tool === 'lsp') {
    if (typeof input.uri !== 'string' || !input.uri.trim()) return { paths: [] }
    try {
      const url = new URL(input.uri)
      if (url.protocol !== 'file:') {
        return { error: `Strict workspace only allows file:// URIs for ${tool}` }
      }
      return { paths: [fileURLToPath(url)] }
    } catch (error) {
      return {
        error: `Strict workspace could not inspect ${tool} URI: ${
          error instanceof Error ? error.message : String(error)
        }`,
      }
    }
  }

  if (tool === 'pages.scaffold') {
    try {
      return { paths: resolvePagesScaffoldPolicyTargets(input, baseCwd) }
    } catch (error) {
      return {
        error: `Strict workspace could not inspect pages.scaffold outputs: ${
          error instanceof Error ? error.message : String(error)
        }`,
      }
    }
  }

  if (tool === 'media.speak') {
    if (typeof input.outputPath !== 'string' || !input.outputPath.trim()) {
      return {
        error: 'Strict workspace blocks media.speak without outputPath because its default temporary file is outside the workspace',
      }
    }
    return { paths: [resolveToolPath(input.outputPath, baseCwd)] }
  }

  if (tool === 'apply_patch') {
    try {
      const parsed = parseApplyPatch(String(input.patch ?? ''))
      return { paths: parsed.files.map((file) => resolveToolPath(file.path, baseCwd)) }
    } catch (error) {
      return {
        error: `Strict workspace could not inspect apply_patch paths: ${
          error instanceof Error ? error.message : String(error)
        }`,
      }
    }
  }

  return { paths: [] }
}

/**
 * A strict workspace is a filesystem capability, not an autonomy setting.
 * Human approval can authorize a side effect but cannot widen this root.
 */
function checkStrictWorkspaceBoundary(request: ToolExecRequest): PolicyCheckResult | null {
  if (!request.workspaceRoot) return null

  const canonicalWorkspaceRoot = canonicalBoundaryPath(request.workspaceRoot)
  if ('error' in canonicalWorkspaceRoot) {
    return {
      allowed: false,
      reason: 'Strict workspace root could not be safely validated',
    }
  }
  if (relative(resolve(request.workspaceRoot), canonicalWorkspaceRoot.path) !== '') {
    return {
      allowed: false,
      reason: 'Strict workspace root changed after it was selected; reselect the workspace before continuing',
    }
  }

  // Plugin and MCP tools do not carry an auditable local-filesystem posture.
  // A user approval can authorize a side effect, but cannot attest that an
  // opaque implementation stays under the immutable workspace root.
  if (request.registrationSource === 'plugin' || request.tool.startsWith('mcp.')) {
    return {
      allowed: false,
      reason: `Strict workspace blocks ${request.tool}: unaudited external tools cannot guarantee workspace-only filesystem access`,
    }
  }

  // LSP servers retain their own process-wide project roots and may return or
  // inspect dependency files outside a narrower desktop workspace.
  if (
    request.tool === 'lsp'
    || request.tool === 'code.diagnostics'
    || (
      request.tool === 'code.symbols'
      && typeof request.input.language === 'string'
      && request.input.language.trim()
    )
  ) {
    return {
      allowed: false,
      reason: `Strict workspace blocks ${request.tool}: the language-server process is not workspace-confined`,
    }
  }

  // Text extraction is in-process for text files and PDFs with OCR disabled.
  // OCR-capable paths spawn pdftoppm/tesseract and use host temp directories,
  // which cannot be confined without an OS filesystem sandbox.
  if (request.tool === 'media.extract_text') {
    const rawPath = typeof request.input.path === 'string' ? request.input.path : ''
    const extension = extname(rawPath).toLowerCase()
    const needsHostOcr = extension === '.pdf'
      ? request.input.allowOcr !== false
      : ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff', '.gif'].includes(extension)
    if (needsHostOcr) {
      return {
        allowed: false,
        reason: 'Strict workspace blocks media.extract_text OCR: host subprocesses and temporary files are not workspace-confined',
      }
    }
  }

  // Host command execution cannot be proven workspace-confined by parsing
  // command strings. Keep this fail-closed until a runner can attest that it
  // provides an active workspace-only filesystem sandbox.
  if (
    STRICT_WORKSPACE_UNCONFINED_TOOLS.has(request.tool)
    || (
      !STRICT_WORKSPACE_SELF_CONFINED_OFFICE_TOOLS.has(request.tool)
      && !STRICT_WORKSPACE_MANAGED_PROCESS_TOOLS.has(request.tool)
      && STRICT_WORKSPACE_UNCONFINED_PREFIXES.some((prefix) => request.tool.startsWith(prefix))
    )
  ) {
    return {
      allowed: false,
      reason: `Strict workspace blocks ${request.tool}: local host command execution cannot guarantee workspace-only filesystem access`,
    }
  }

  if (!hasAuditedStrictWorkspacePosture(request.tool)) {
    return {
      allowed: false,
      reason: `Strict workspace blocks ${request.tool}: this built-in tool has no audited workspace confinement posture`,
    }
  }

  const targets = strictWorkspaceTargetsForRequest(request, canonicalWorkspaceRoot.path)
  if ('error' in targets) {
    return { allowed: false, reason: targets.error }
  }

  const cwdTarget = request.cwd ? [request.cwd] : []
  for (const target of [...cwdTarget, ...targets.paths]) {
    const canonicalTarget = canonicalBoundaryPath(target)
    if ('error' in canonicalTarget) {
      return {
        allowed: false,
        reason: `Strict workspace could not safely validate the ${request.tool} target`,
      }
    }
    if (!isInsideOrEqual(canonicalTarget.path, canonicalWorkspaceRoot.path)) {
      return {
        allowed: false,
        reason: `Strict workspace blocks ${request.tool}: the target resolves outside workspace boundary`,
      }
    }
  }
  return null
}

function checkWorkspaceWriteBoundary(request: ToolExecRequest): PolicyCheckResult | null {
  const workspaceRoot = resolveToolPath(request.cwd ?? process.cwd())
  const canonicalWorkspaceRoot = canonicalBoundaryPath(workspaceRoot)
  if ('error' in canonicalWorkspaceRoot) {
    return {
      allowed: false,
      reason: `WorkspaceWrite mode could not inspect workspace root: ${canonicalWorkspaceRoot.error}`,
    }
  }

  const targets = workspaceTargetsForRequest(request.tool, request.input, workspaceRoot)
  if ('error' in targets) {
    return { allowed: false, reason: targets.error }
  }
  for (const target of targets.paths) {
    const canonicalTarget = canonicalBoundaryPath(target)
    if ('error' in canonicalTarget) {
      return {
        allowed: false,
        reason: `WorkspaceWrite mode could not inspect ${request.tool} target: ${canonicalTarget.error}`,
      }
    }
    if (!isInsideOrEqual(canonicalTarget.path, canonicalWorkspaceRoot.path)) {
      return {
        allowed: false,
        reason: `WorkspaceWrite mode blocks ${request.tool}: ${canonicalTarget.path} is outside workspace ${canonicalWorkspaceRoot.path}`,
      }
    }
  }
  return null
}

function uniqueValues(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)))
}

function policyPathVariants(
  rawPath: string,
  cwd: string | undefined,
): { paths: string[] } | { error: string } {
  const absolutePath = resolveToolPath(rawPath, cwd)
  const variants = [rawPath, absolutePath]
  const canonical = canonicalBoundaryPath(absolutePath)
  if ('error' in canonical) {
    return { error: canonical.error }
  }
  variants.push(canonical.path)
  return { paths: uniqueValues(variants) }
}

const TEMPLATE_SUFFIXES = ['.example', '.sample', '.template', '.dist']
const TEMPLATE_EXEMPTABLE_GLOBS = new Set(['**/.env.*', '**/credentials*', '**/secrets*'])

function isTemplateExempt(path: string, matchedGlob: string): boolean {
  if (!TEMPLATE_EXEMPTABLE_GLOBS.has(matchedGlob)) return false
  const file = basename(path).toLowerCase()
  return TEMPLATE_SUFFIXES.some((suffix) =>
    file.endsWith(suffix) || file.includes(`${suffix}.`),
  )
}

function findDeniedPathGlob(path: string, denyPaths: string[]): string | null {
  for (const glob of denyPaths) {
    if (!matchesPathAny(path, [glob])) continue
    if (isTemplateExempt(path, glob)) continue
    return glob
  }
  return null
}

function checkDeniedPathCandidates(
  rawPaths: string[],
  denyPaths: string[],
  cwd: string | undefined,
): PolicyCheckResult | null {
  for (const rawPath of rawPaths) {
    const absolutePath = resolveToolPath(rawPath, cwd)
    const directGlob = findDeniedPathGlob(rawPath, denyPaths)
      ?? findDeniedPathGlob(absolutePath, denyPaths)
    if (directGlob) {
      return {
        allowed: false,
        reason: `Path matches deny_paths glob '${directGlob}'`,
      }
    }
    const variants = policyPathVariants(rawPath, cwd)
    if ('error' in variants) {
      return {
        allowed: false,
        reason: `Path matches could not be safely inspected: ${variants.error}`,
      }
    }
    for (const candidate of variants.paths) {
      const matchedGlob = findDeniedPathGlob(candidate, denyPaths)
      if (matchedGlob) {
        return {
          allowed: false,
          reason: `Path matches deny_paths glob '${matchedGlob}'`,
        }
      }
    }
  }
  return null
}

/**
 * ReadOnly autonomy may launch a temporary observer only when the runtime can
 * enforce the strict-workspace capability: read-only workspace, no host
 * network, bounded resources, daemon ownership, and an automatic positive
 * lifetime. This is intentionally based on execution posture rather than an
 * executable/argument allowlist, so it applies uniformly to samplers, TUIs,
 * test watchers, and future tools without prompt-specific heuristics.
 */
function isStrictReadOnlyManagedProcessStart(request: ToolExecRequest): boolean {
  if (request.tool !== 'process.start' || !request.workspaceRoot) return false
  const env = request.input.env
  if (env && typeof env === 'object' && Object.keys(env).length > 0) return false
  const lifetime = request.input.lifetime
  if (lifetime !== undefined && lifetime !== 'bounded') return false
  const ttlMs = request.input.ttlMs
  return ttlMs === undefined
    || (typeof ttlMs === 'number'
      && Number.isFinite(ttlMs)
      && ttlMs >= 100
      && ttlMs <= 24 * 60 * 60_000)
}

/**
 * A terminal invocation is read-only when the request carries an immutable
 * workspace capability. The terminal runtime treats that capability as a
 * contract: it must execute in the fail-closed strict-workspace runner with a
 * read-only workspace, private temporary storage, bounded resources, hidden
 * host credentials/state, and no network unless the runtime grants a narrower
 * audited read-only network capability.
 *
 * This is deliberately independent of the executable name. Maintaining a
 * growing allowlist of `pwd`, `ls`, language analyzers, and future inspection
 * tools both blocks legitimate observations and mistakes names for security
 * boundaries. Without a strict workspace capability the host execution path
 * remains side-effecting and therefore fails closed in ReadOnly/Plan modes.
 */
function isStrictReadOnlyTerminalRun(request: ToolExecRequest): boolean {
  if (
    request.tool !== 'terminal.run'
    || typeof request.workspaceRoot !== 'string'
    || request.workspaceRoot.trim().length === 0
  ) return false

  const externalNetworkRequested = request.input.network === 'external'
    || Boolean(
      request.input.network
      && typeof request.input.network === 'object'
      && !Array.isArray(request.input.network)
      && (request.input.network as Record<string, unknown>).mode === 'external',
    )
  return !externalNetworkRequested || isReadOnlyNetworkTerminalCommand(request.input)
}

function invocationSecurityEffect(
  request: ToolExecRequest,
): ReturnType<typeof resolveToolSecurityDescriptor>['effect'] {
  if (request.registrationSource === 'plugin' && !request.security) {
    return 'unknown'
  }
  const descriptor = resolveToolSecurityDescriptor(request.tool, request.security)
  if (descriptor.effect !== 'dynamic') return descriptor.effect
  if (isStrictReadOnlyTerminalRun(request)) {
    return 'observe'
  }
  if (request.tool === 'process.start' && isStrictReadOnlyManagedProcessStart(request)) {
    return 'observe'
  }
  return 'process-lifecycle'
}

export function isPolicyReadOnlyRequest(request: ToolExecRequest): boolean {
  const effect = invocationSecurityEffect(request)
  return effect === 'observe' || effect === 'internal-state'
}

/**
 * Canonical policy classification used by per-turn approval guards. Unknown
 * and plugin-provided tools stay side-effecting unless their registered
 * descriptor explicitly classifies them as safe.
 */
export function isPolicyReadOnlyTool(tool: string, security?: ToolExecRequest['security']): boolean {
  const effect = resolveToolSecurityDescriptor(tool, security).effect
  return effect === 'observe' || effect === 'internal-state'
}

/** Destructive memory tools that mutate or delete durable state.
 * These are blocked in read-only and plan modes regardless of the
 * tool-rule policy mode, so an agent can't sneak a memory.forget
 * past a "diagnose only" autonomy. */
const MEMORY_DESTRUCTIVE_TOOLS = new Set([
  'memory.forget',
  'memory.update',
  'memory.maintenance',
  'memory.import',
  'memory.documents.delete',
  'memory.documents.ingest',
  'memory.daily.replace',
  'memory.journal.manage',
  'memory.reminders.cancel',
  'memory.tag.rename',
  'memory.merge',
])

const USER_MEMORY_WRITE_TOOLS = new Set(['memory.remember'])

const WORKSPACE_WRITE_FAST_PATH_TOOLS = new Set(['fs.write', 'fs.append', 'fs.edit', 'fs.move', 'apply_patch'])

const SHELL_WRAPPER_EXECUTABLES = new Set(['bash', 'dash', 'sh', 'zsh', 'fish', 'csh', 'tcsh'])

export class PolicyEngine implements IToolPolicy {
  constructor(private config: PolicyEngineConfig) {}

  /** Read-only snapshot of the active policy. Surfaces use this to render
   * the current rule table to the operator (e.g. cli `/policy`). The
   * returned object is intentionally a structuredClone so callers can't
   * mutate engine state by holding the reference. */
  describe(): PolicyEngineConfig {
    return JSON.parse(JSON.stringify(this.config)) as PolicyEngineConfig
  }

  check(
    request: ToolExecRequest,
    autonomy: AutonomyLevel,
    primaryAgentId?: string,
    autoApprove?: boolean,
  ): PolicyCheckResult {
    const { tool, input } = request
    const isMcpTool = tool.startsWith('mcp.')
    const readOnlySafeInvocation = isPolicyReadOnlyRequest(request)
    // MCP tools arrive with server-defined names the operator has no
    // per-tool rules for; without a fallback rule they would skip every
    // deny_paths / deny_patterns check below. `mcp.*` acts as the shared
    // rule for all of them, overridable per exact tool name.
    const toolRule = this.config.tools[tool] ?? (isMcpTool ? this.config.tools['mcp.*'] : undefined)

    // Normalize before the exact-match so a stored/forwarded agent id like
    // 'Plan', 'plan ', or 'PLAN' can't silently bypass plan-mode write
    // blocking (the surfaces only send 'plan', but a typo/case variant must
    // fail safe — block — not fail open).
    if (primaryAgentId?.trim().toLowerCase() === 'plan' && !readOnlySafeInvocation) {
      return {
        allowed: false,
        reason: `Plan mode blocks ${invocationSecurityEffect(request)} tool '${tool}'`,
      }
    }

    const strictWorkspaceResult = checkStrictWorkspaceBoundary(request)
    if (strictWorkspaceResult) {
      return strictWorkspaceResult
    }

    // 1. ReadOnly mode: block non-read tools
    if (
      autonomy === 'readonly'
      && !readOnlySafeInvocation
    ) {
      // Specialized message for destructive memory tools so the agent's
      // explanation back to the user is precise instead of generic.
      if (MEMORY_DESTRUCTIVE_TOOLS.has(tool)) {
        return {
          allowed: false,
          reason: `ReadOnly autonomy blocks destructive memory tool '${tool}'. Switch autonomy to ask the user before mutating memory.`,
        }
      }
      return { allowed: false, reason: `ReadOnly mode blocks tool '${tool}'` }
    }

    // Build the full command string for pattern matching
    const commandStr = this.buildCommandString(tool, input)

    // 1b. Edit fast-paths are auto-allowed after deny and workspace
    // boundary checks run. deny_patterns / deny_paths still apply below
    // before we return.
    const isAcceptEditsFastPath =
      autonomy === 'accept-edits' && WORKSPACE_WRITE_FAST_PATH_TOOLS.has(tool)
    const isWorkspaceWriteFastPath =
      autonomy === 'workspace-write' && WORKSPACE_WRITE_FAST_PATH_TOOLS.has(tool)

    // 2. deny_patterns check (from tool-specific rule, then defaults-level)
    const denyPatterns = toolRule?.deny_patterns ?? []
    if (denyPatterns.length > 0 && matchesAny(commandStr, denyPatterns)) {
      return { allowed: false, reason: `Command matches deny_pattern` }
    }

    // 2-dl. Download-pipe-to-shell guard for command tools. The glob
    // deny_patterns miss URLs (slashes break the path-segment glob), so a
    // `curl https://x/y | sh` chain is caught here regardless of slashes.
    if (
      denyPatterns.length > 0 &&
      (tool === 'terminal.run' || tool === 'process.start' || tool === 'service.start' || tool === 'service.install' || isMcpTool) &&
      isDownloadPipeToShell(commandStr)
    ) {
      return {
        allowed: false,
        reason: 'Command matches deny_pattern (download piped to interpreter)',
      }
    }

    // 2a. MCP arg hardening: MCP servers name their command/script arguments
    // freely and a shell-running MCP tool would otherwise only get the
    // deny_paths scan, not deny_patterns. Scan every string argument value
    // (substring match, like the shell-wrapper guard) so `rm -rf /` or
    // `curl … | sh` hidden in any arg key is blocked the same way it is for
    // terminal.run. Purely additive — only adds denials.
    if (isMcpTool && denyPatterns.length > 0) {
      for (const value of collectStringValues(input)) {
        if (containsAny(value, denyPatterns)) {
          return { allowed: false, reason: 'MCP tool argument matches deny_pattern' }
        }
      }
    }

    // 2b. Shell wrapper hardening: a `bash -c "<script>"` invocation skips the
    // anchored deny_patterns check above (the wrapper prefix breaks the
    // regex), and supervised mode would otherwise just ask the user to
    // approve. Scan the wrapped script body for the same deny_patterns,
    // matched as substrings, so a hostile wrapper can't smuggle `rm -rf /`
    // past an explicit deny_pattern by hiding it inside `bash -c`.
    if (
      (tool === 'terminal.run' || tool === 'process.start' || tool === 'service.start' || tool === 'service.install') &&
      denyPatterns.length > 0
    ) {
      const exe = normalizeExecutable(input.executable)
      if (exe && SHELL_WRAPPER_EXECUTABLES.has(exe)) {
        const script = extractWrappedScript(input)
        if (script && containsAny(script, denyPatterns)) {
          return {
            allowed: false,
            reason: 'Wrapped shell script matches deny_pattern',
          }
        }
      }
    }

    // 3. deny_paths check
    // For most tools, `input.path` is the affected file. For `apply_patch`,
    // a single call can write many files, so we expand the patch and check
    // every file path against deny_paths — otherwise a hostile patch can
    // bypass the deny list by hiding the protected file as the second
    // entry. Unparseable patches are caught by the workspace-write path.
    const denyPaths = toolRule?.deny_paths ?? []
    if (denyPaths.length > 0) {
      const candidatePaths: string[] = []
      const inputPath = typeof input.path === 'string' ? input.path : ''
      if (inputPath.includes('\0')) {
        return {
          allowed: false,
          reason: 'Path could not be safely inspected: NUL bytes are not allowed',
        }
      }
      if (inputPath) candidatePaths.push(inputPath)
      if (isMcpTool) {
        // MCP servers name their path arguments freely (path, file, target,
        // uri, ...), so every string value is a deny_paths candidate.
        // Non-path strings resolve under cwd and simply never match the
        // protected-path globs. A NUL-bearing value cannot be an OS path;
        // skip it here so binary content is not rejected by path
        // canonicalization. The explicit input.path above remains fail-closed.
        candidatePaths.push(
          ...collectStringValues(input).filter((value) => !value.includes('\0')),
        )
      }
      if (tool === 'apply_patch') {
        try {
          const parsed = parseApplyPatch(String(input.patch ?? ''))
          for (const file of parsed.files) candidatePaths.push(file.path)
        } catch {
          // ignore — parse-failure is reported by other paths
        }
      }
      const deniedPathResult = checkDeniedPathCandidates(candidatePaths, denyPaths, request.cwd)
      if (deniedPathResult) {
        return deniedPathResult
      }
    }

    const shouldEnforceWorkspaceBoundary =
      autonomy === 'workspace-write' || autonomy === 'accept-edits' || autonomy === 'autonomous'
    if (shouldEnforceWorkspaceBoundary) {
      const workspaceBoundaryResult = checkWorkspaceWriteBoundary(request)
      if (workspaceBoundaryResult) {
        return workspaceBoundaryResult
      }
    }

    // 3b. URL allow/deny checks for headless browser tools and the dedicated
    // user-visible browser handoff. Keeping this keyed to URL-bearing tools
    // prevents a custom/file protocol from bypassing policy just because the
    // action does not use the browser.* namespace.
    const isUrlBearingTool = tool.startsWith('browser.') || tool === 'computer.open_url'
    if (toolRule && isUrlBearingTool) {
      const targetUrl = tool === 'browser.remote_action' && input.action !== 'navigate'
        ? input.expectedUrl
        : input.url
      const url = typeof targetUrl === 'string' ? targetUrl : ''
      const allowUrls = toolRule.allow_urls ?? []
      const allowHosts = toolRule.allow_hosts ?? []
      if (allowUrls.length > 0) {
        if (!url || !matchesAny(url, allowUrls)) {
          return { allowed: false, reason: 'URL is outside allow_urls policy' }
        }
      }
      if (allowHosts.length > 0) {
        let hostname = ''
        try {
          hostname = new URL(url).hostname.toLowerCase()
        } catch {
          return { allowed: false, reason: 'URL host could not be validated' }
        }
        const normalizedAllowHosts = allowHosts.map((host) => host.toLowerCase())
        if (!matchesAny(hostname, normalizedAllowHosts)) {
          return { allowed: false, reason: 'URL host is outside allow_hosts policy' }
        }
      }
      const denyUrls = toolRule.deny_urls ?? []
      if (url) {
        for (const denyUrl of denyUrls) {
          if (globToRegex(denyUrl).test(url)) {
            return { allowed: false, reason: `Denied by URL: ${denyUrl}` }
          }
        }
      }
    }

    // 4. deny_executables check
    const denyExecutables = toolRule?.deny_executables ?? []
    const executable = normalizeExecutable(input.executable)
    const normalizedDeniedExecutables = denyExecutables.map(normalizeExecutable)
    const shellWrapperRequested =
      tool === 'terminal.run' && executable && SHELL_WRAPPER_EXECUTABLES.has(executable)

    if (
      normalizedDeniedExecutables.length > 0 &&
      executable &&
      normalizedDeniedExecutables.includes(executable)
    ) {
      return { allowed: false, reason: `Executable '${executable}' is blocked` }
    }

    // 5. Mode check from tool rule. Explicit blocks remain authoritative,
    // but a successfully audited observation/internal-state invocation must
    // not be turned back into an approval request by a legacy `supervised`
    // tool rule. All deny, workspace-boundary, executable, and URL checks have
    // already run above, so this is the single effect-based approval boundary.
    const mode =
      toolRule?.mode ??
      (USER_MEMORY_WRITE_TOOLS.has(tool) ? 'autonomous' : this.config.defaults.mode)
    if (mode === 'blocked') {
      return { allowed: false, reason: `Tool '${tool}' is blocked by policy` }
    }
    if (readOnlySafeInvocation) {
      return { allowed: true }
    }

    if (!toolRule && this.config.defaults.unmatched_policy === 'deny') {
      return { allowed: false, reason: 'Denied by default unmatched_policy' }
    }

    // 4b. Edit fast-paths skip the supervised approval gate once deny_*,
    // workspace, and explicit blocked-mode checks above have passed.
    if (isAcceptEditsFastPath || isWorkspaceWriteFastPath) {
      return { allowed: true }
    }

    // 5b. Approval semantics. Autonomy answers "does the runtime prompt?";
    // the tool rule answers "is this tool allowed at all?". Every hard deny
    // (deny_*, workspace boundary, blocked mode, unmatched_policy: deny,
    // wrapped-script deny, plan/readonly) has already returned above, so the
    // only question left is whether to prompt.
    //
    // - `ask` rules prompt under every autonomy, including autonomous.
    // - `supervised` rules prompt under prompting autonomies; under
    //   autonomous they run, because autonomous means "do not ask", not
    //   "block more". `defaults.autonomous_honors_supervised_rules: true`
    //   restores the legacy prompt (which the autonomous executor blocks).
    const promptingAutonomy =
      autonomy === 'supervised' ||
      autonomy === 'accept-edits' ||
      autonomy === 'workspace-write'
    const honorsSupervisedUnderAutonomous =
      this.config.defaults.autonomous_honors_supervised_rules === true
    const ruleRequestsPrompt =
      mode === 'ask' ||
      (mode === 'supervised' && (autonomy !== 'autonomous' || honorsSupervisedUnderAutonomous))
    const autonomousRunsSupervisedRule =
      mode === 'supervised' && autonomy === 'autonomous' && !honorsSupervisedUnderAutonomous

    if (ruleRequestsPrompt || promptingAutonomy) {
      const allowPatterns = toolRule?.allow_patterns ?? []
      if (allowPatterns.length > 0 && matchesAny(commandStr, allowPatterns)) {
        return { allowed: true }
      }

      if (shellWrapperRequested && (promptingAutonomy || ruleRequestsPrompt)) {
        return {
          allowed: true,
          requiresApproval: true,
          reason: 'Shell wrapper execution requires explicit approval',
        }
      }

      // If not in allow list, require approval
      const supervising =
        ruleRequestsPrompt ||
        (promptingAutonomy && !readOnlySafeInvocation && !USER_MEMORY_WRITE_TOOLS.has(tool))
      if (supervising) {
        if (autoApprove) {
          return { allowed: true, reason: 'Auto-approved by CI mode (supervised)' }
        }
        return {
          allowed: true,
          requiresApproval: true,
          reason:
            mode === 'ask'
              ? 'Ask mode always requires approval'
              : 'Supervised mode requires approval',
        }
      }
    }

    // 5c. A `supervised` rule under autonomous autonomy runs without a
    // prompt. Its allow_patterns are "auto-approve without asking" hints for
    // prompting autonomies, not an allowlist, so they must not turn into a
    // deny here: autonomous may run at least what supervised could run
    // after approval, minus the deny gates that already returned above.
    if (autonomousRunsSupervisedRule) {
      return { allowed: true, reason: AUTONOMOUS_SUPERVISED_RULE_REASON }
    }

    // 6. allow_patterns check for autonomous rules
    const allowPatterns = toolRule?.allow_patterns ?? []
    if (allowPatterns.length > 0 && !matchesAny(commandStr, allowPatterns)) {
      // Has allow patterns but doesn't match → fall through to unmatched
      if (this.config.defaults.unmatched_policy === 'deny') {
        return { allowed: false, reason: 'Command does not match allow_patterns' }
      }
    }

    if (toolRule && mode === 'autonomous') {
      return { allowed: true }
    }

    // 7. Fallback to defaults.unmatched_policy
    if (this.config.defaults.unmatched_policy === 'deny') {
      return { allowed: false, reason: 'Denied by default unmatched_policy' }
    }

    return { allowed: true }
  }

  async reload(): Promise<void> {
    // Hot-reload from disk is not yet implemented. Throwing keeps callers
    // honest: anyone wiring up policy reload must either (a) plumb a real
    // loader through updateConfig() or (b) construct a fresh PolicyEngine.
    // Returning silently here would let policy edits appear to apply while
    // the in-memory config keeps serving stale rules.
    throw new Error(
      'PolicyEngine.reload() is not implemented; rebuild the engine via updateConfig() or construct a new instance instead.',
    )
  }

  updateConfig(config: PolicyEngineConfig): void {
    this.config = config
  }

  private buildCommandString(tool: string, input: Record<string, unknown>): string {
    // process.start and process-backed service.start run an executable just
    // like terminal.run, so build the
    // same "exe args" command string — otherwise deny_patterns (rm -rf /, fork
    // bombs, ...) would never match its JSON-stringified input (CLI_BACKLOG.md B12).
    // service.install registers a boot-persistent OS service from an
    // executable + args, i.e. it runs a command just like terminal.run. Build
    // the same "exe args" string so deny_patterns / download-pipe / shell-wrapper
    // guards apply — otherwise its JSON-stringified input would never match and
    // a `curl … | sh` persistence payload would sail through (PLAN_021 D2).
    if (tool === 'terminal.run' || tool === 'process.start' || tool === 'service.start' || tool === 'service.install') {
      const exe = (input.executable as string) ?? ''
      const args = normalizeTerminalArgs(input.args, exe)
      if (tool === 'service.start' && !exe) {
        const runtime = (input.containerRuntime as string) ?? ''
        const image = (input.image as string) ?? ''
        const command = Array.isArray(input.command)
          ? input.command.filter((arg): arg is string => typeof arg === 'string')
          : []
        return [runtime, image, ...command].filter(Boolean).join(' ').trim()
      }
      return [exe, ...args].filter(Boolean).join(' ').trim()
    }
    if (tool === 'fs.read' || tool === 'fs.write' || tool === 'fs.append') {
      return (input.path as string) ?? ''
    }
    return JSON.stringify(input)
  }
}

export const __testables = {
  containsAny,
  extractWrappedScript,
  canonicalBoundaryPath,
  tokenizeShellForPolicy,
  positionalArgs,
  extractInlineFlagValue,
  decodePolicyStringLiteral,
  collectOutputRedirectionTargets,
  collectMutatingCommandTargets,
  collectPythonInlineWriteTargets,
  collectJavaScriptInlineWriteTargets,
  collectInterpreterSnippetTargets,
  collectShellWriteTargets,
  workspaceTargetsForRequest,
  checkWorkspaceWriteBoundary,
  checkStrictWorkspaceBoundary,
  policyPathVariants,
  checkDeniedPathCandidates,
}

/**
 * Creates a default PolicyConfig with safe hardcoded rules.
 */
// Bound the recursive walk so a pathological deeply-nested argument
// object can't hang the policy check, but keep it far above any real
// tool-call shape: a low cap would let an attacker bury a protected
// path behind enough decoy strings to skip the deny_paths scan.
const MAX_COLLECTED_STRING_VALUES = 10_000

/** Recursively collect string values from a tool-call argument object. */
function collectStringValues(value: unknown, out: string[] = []): string[] {
  if (out.length >= MAX_COLLECTED_STRING_VALUES) return out
  if (typeof value === 'string') {
    out.push(value)
  } else if (Array.isArray(value)) {
    for (const item of value) collectStringValues(item, out)
  } else if (value && typeof value === 'object') {
    for (const item of Object.values(value)) collectStringValues(item, out)
  }
  return out
}

export function createDefaultPolicy(): PolicyConfig {
  const writeDenyPaths = [
    '**/.ssh/**',
    '**/.env',
    '**/.env.*',
    '**/.envrc',
    '**/.netrc',
    '**/.npmrc',
    '**/.pypirc',
    '**/credentials*',
    '**/secrets*',
    '**/.aws/**',
    '**/.gnupg/**',
    '**/*.pem',
    '**/*.key',
    '/etc/**',
    '/usr/**',
    '/bin/**',
    '/sbin/**',
    // The daemon's own state: writes here would let an agent
    // plant a plugin / rewrite a session / overwrite its own
    // bearer token via fs.write, fs.edit, apply_patch, or browser.screenshot.
    '**/.sepilotd/security/**',
    '**/.sepilotd/sessions/**',
    '**/.sepilotd/memory/**',
    '**/.sepilotd/skills/**',
    '**/.sepilotd/plugins/**',
    '**/.sepilotd/audit/**',
  ]

  const dangerousCommandPatterns = [
    'rm -rf /',
    'rm -rf /*',
    ':(){ :|:& };:',
    'mkfs.*',
    'dd if=*of=/dev/*',
    '> /dev/sda',
    'chmod -R 777 /',
    'wget * | sh',
    'curl * | sh',
  ]

  return {
    version: 1,
    defaults: {
      mode: 'supervised',
      unmatched_policy: 'deny',
      max_timeout_ms: 30000,
      max_output_bytes: 10 * 1024 * 1024,
    },
    tools: {
      // Shared default for every MCP-provided tool (overridable per exact
      // tool name). Without it, a filesystem/shell MCP server would bypass
      // the deny lists that protect ~/.ssh, .env, and the daemon's own state.
      // deny_patterns mirror terminal.run so a shell-running MCP tool can't
      // smuggle a destructive command through a string argument.
      'mcp.*': {
        mode: 'supervised',
        deny_paths: [...writeDenyPaths],
        deny_patterns: [...dangerousCommandPatterns],
      },
      'terminal.run': {
        mode: 'supervised',
        deny_patterns: [...dangerousCommandPatterns],
        deny_executables: [],
        allow_patterns: [
          'echo *',
          'cat *',
          'ls *',
          'pwd',
          'whoami',
          'hostname',
          'hostname *',
          'date',
          'date *',
          'head *',
          'tail *',
          'wc *',
          'grep *',
          'find *',
          'git *',
          'uname',
          'uname *',
          'nproc',
          'nproc *',
          'free',
          'free *',
          'df',
          'df *',
          'du',
          'du *',
          'uptime',
          'lscpu',
          'lscpu *',
          'lsblk',
          'lsblk *',
          'printenv',
          'printenv *',
          'env',
          'python3 -m py_compile *',
          'python3 -m py_compile **',
          'node --check *',
          'node --check **',
          'bash -n *',
          'bash -n **',
        ],
      },
      'fs.read': {
        mode: 'autonomous',
        deny_paths: [
          '**/.ssh/**',
          '**/.env',
          '**/.env.*',
          '**/.envrc',
          '**/.netrc',
          '**/.npmrc',
          '**/.pypirc',
          '**/credentials*',
          '**/secrets*',
          '**/.aws/**',
          '**/.gnupg/**',
          '**/*.pem',
          '**/*.key',
          // Linux/BSD account secrets — readable by an over-zealous agent
          // running as root and historically the first stop in any LFI.
          // `**` (rather than `*`) covers /etc/sudoers.d/<file> and similar
          // sub-trees; the simple `*` only spans a single path segment.
          '/etc/shadow**',
          '/etc/gshadow**',
          '/etc/sudoers**',
          '/etc/master.passwd**',
          // The daemon's own state. Keeping these out of fs.read prevents
          // an autonomous agent from exfiltrating its own bearer tokens,
          // session transcripts, memory db, encryption key, audit log, or
          // user-installed skill code.
          '**/.sepilotd/security/**',
          '**/.sepilotd/sessions/**',
          '**/.sepilotd/memory/**',
          '**/.sepilotd/skills/**',
          '**/.sepilotd/plugins/**',
          '**/.sepilotd/audit/**',
        ],
      },
      'doc.get': {
        mode: 'autonomous',
      },
      'doc.outline': {
        mode: 'autonomous',
      },
      'doc.replace_section': {
        mode: 'supervised',
      },
      'doc.replace_range': {
        mode: 'supervised',
      },
      'doc.insert_after_section': {
        mode: 'supervised',
      },
      'doc.append': {
        mode: 'supervised',
      },
      'doc.rewrite': {
        mode: 'supervised',
      },
      'doc.diff_preview': {
        mode: 'autonomous',
      },
      'system.info': {
        mode: 'autonomous',
      },
      'assistant.status': {
        mode: 'autonomous',
      },
      'gitea.actions.runs.inspect': {
        mode: 'autonomous',
      },
      'apps.list': {
        mode: 'autonomous',
      },
      'apps.read': {
        mode: 'autonomous',
      },
      'apps.search': {
        mode: 'autonomous',
      },
      'apps.write': {
        mode: 'supervised',
      },
      'apps.mutate': {
        mode: 'supervised',
      },
      'fs.write': {
        mode: 'supervised',
        deny_paths: [...writeDenyPaths],
      },
      'fs.append': {
        mode: 'supervised',
        deny_paths: [...writeDenyPaths],
      },
      'fs.edit': {
        mode: 'supervised',
        deny_paths: [...writeDenyPaths],
      },
      'fs.move': {
        mode: 'supervised',
        deny_paths: [...writeDenyPaths],
      },
      // apply_patch inherits fs.write deny semantics so a single
      // multi-file patch cannot bypass the deny list path-by-path.
      // The policy engine evaluates deny_paths against every file
      // affected by the patch.
      apply_patch: {
        mode: 'supervised',
        deny_paths: [...writeDenyPaths],
      },
      // Long-running process tools mirror terminal.run: they execute commands,
      // so they are supervised (and carry the same destructive-command deny
      // list) rather than falling through to unmatched_policy: deny, which
      // silently blocked an agent that picked process.start to run a script
      // (CLI_BACKLOG.md B12). process.list/sessions/read/follow/wait are read-only and
      // handled by the canonical observe/internal-state effect descriptors.
      'process.start': {
        mode: 'supervised',
        deny_patterns: [...dangerousCommandPatterns],
        deny_executables: [],
      },
      // Durable service supervision is the intentional escape hatch for a
      // process that should survive the current agent turn. Starting one is
      // command execution and remains approval-gated with the same destructive
      // command protections as process.start. Inspection and log following are
      // classified as read-only above; lifecycle mutations stay supervised.
      'service.start': {
        mode: 'supervised',
        deny_patterns: [...dangerousCommandPatterns],
        deny_executables: [],
      },
      'service.stop': {
        mode: 'supervised',
      },
      'service.restart': {
        mode: 'supervised',
      },
      'service.remove': {
        mode: 'supervised',
      },
      'service.healthcheck': {
        mode: 'supervised',
      },
      // Native OS service tools register/mutate boot-persistent services — the
      // highest-value persistence + RCE surface. service.install carries the
      // command-execution deny lists (its executable+args are built into a
      // command string in buildCommandString); every mutating service tool is
      // supervised so it requires approval even under autonomous autonomy
      // instead of falling through to unmatched_policy (PLAN_021 D2).
      'service.install': {
        mode: 'supervised',
        deny_patterns: [...dangerousCommandPatterns],
        deny_executables: [],
      },
      'service.enable': {
        mode: 'supervised',
      },
      'service.disable': {
        mode: 'supervised',
      },
      'service.uninstall': {
        mode: 'supervised',
      },
      'process.stop': {
        mode: 'supervised',
      },
      'process.signal': {
        mode: 'supervised',
      },
      'browser.remote_snapshot': {
        mode: 'autonomous',
      },
      'browser.remote_action': {
        mode: 'supervised',
        deny_urls: ['file://**', 'chrome://**', 'about://**'],
      },
      'browser.navigate': {
        mode: 'supervised',
        deny_urls: ['file://**', 'chrome://**', 'about://**'],
        max_timeout_ms: 60000,
      },
      'browser.screenshot': {
        mode: 'supervised',
        deny_urls: ['file://**'],
        deny_paths: [...writeDenyPaths],
      },
      'browser.click': {
        mode: 'supervised',
        deny_urls: ['file://**', 'chrome://**', 'about://**'],
      },
      'browser.evaluate': {
        mode: 'supervised',
        deny_urls: ['file://**', 'chrome://**', 'about://**'],
        deny_paths: [...writeDenyPaths],
      },
      'browser.extract': {
        mode: 'supervised',
        deny_urls: ['file://**'],
      },
      'device.delegate': {
        mode: 'supervised',
      },
      'computer.list_windows': {
        mode: 'autonomous',
      },
      'computer.list_elements': {
        mode: 'autonomous',
      },
      'computer.observe': {
        mode: 'autonomous',
      },
      'computer.launch_app': {
        mode: 'supervised',
      },
      // Opening a real browser changes the user's visible desktop and may
      // disclose the destination to the browser/network. Always ask, even
      // under Autonomous, so Desktop can provide an explicit HITL checkpoint.
      'computer.open_url': {
        mode: 'ask',
        deny_urls: ['file://**', 'chrome://**', 'about://**'],
      },
      'computer.focus_window': {
        mode: 'supervised',
      },
      'computer.move_mouse': {
        mode: 'supervised',
      },
      'computer.click': {
        mode: 'supervised',
      },
      'computer.drag': {
        mode: 'supervised',
      },
      'computer.type_text': {
        mode: 'supervised',
      },
      'computer.hotkey': {
        mode: 'supervised',
      },
      'computer.scroll': {
        mode: 'supervised',
      },
      'computer.wait': {
        mode: 'autonomous',
      },
      'office.list_open_documents': {
        mode: 'autonomous',
      },
      'office.open_presentation': {
        mode: 'supervised',
      },
      'office.navigate_slide': {
        mode: 'supervised',
      },
      'office.read_slide': {
        mode: 'supervised',
      },
      'office.capture_slide': {
        mode: 'supervised',
      },
      'office.read_active': {
        mode: 'supervised',
      },
      'office.read_selection': {
        mode: 'supervised',
      },
      'office.preview_edit': {
        mode: 'supervised',
      },
      'office.apply_edit': {
        mode: 'supervised',
      },
      'office.replace_selection': {
        mode: 'supervised',
      },
      'web.search': {
        mode: 'autonomous',
      },
      'market.quote': {
        mode: 'autonomous',
      },
      skill: {
        mode: 'autonomous',
      },
      'skillhub.search': {
        mode: 'autonomous',
      },
      'skillhub.install': {
        mode: 'supervised',
      },
      'knowledge.search': { mode: 'autonomous' },
      'knowledge.read': { mode: 'autonomous' },
      'knowledge.save': { mode: 'supervised' },
      'knowledge.edit': { mode: 'supervised' },
      'knowledge.review': { mode: 'supervised' },
      'memory.remember': {
        mode: 'autonomous',
      },
      'memory.list': {
        mode: 'autonomous',
      },
      'memory.forget': {
        mode: 'autonomous',
      },
      'memory.search': {
        mode: 'autonomous',
      },
      'memory.graph.search': {
        mode: 'autonomous',
      },
      'memory.graph.neighbors': {
        mode: 'autonomous',
      },
      'memory.graph.page': {
        mode: 'autonomous',
      },
      'memory.graph.audit': {
        mode: 'autonomous',
      },
      'memory.graph.repair': {
        mode: 'autonomous',
      },
      'memory.graph.repair.apply': {
        mode: 'autonomous',
      },
      'memory.journal.inspect': { mode: 'autonomous' },
      'memory.journal.manage': { mode: 'autonomous' },
      'memory.daily.read': {
        mode: 'autonomous',
      },
      'memory.daily.append': {
        mode: 'autonomous',
      },
      'memory.daily.replace': {
        mode: 'autonomous',
      },
      todowrite: {
        mode: 'autonomous',
      },
      'memory.update': {
        mode: 'autonomous',
      },
      'memory.documents.ingest': {
        mode: 'autonomous',
      },
      'memory.documents.search': {
        mode: 'autonomous',
      },
      'memory.documents.list': {
        mode: 'autonomous',
      },
      'memory.audit': {
        mode: 'autonomous',
      },
      'memory.maintenance': {
        mode: 'supervised',
      },
      'memory.remind_at': {
        mode: 'autonomous',
      },
      // Scheduler tools were absent from this map entirely, so every one of
      // them fell through to `unmatched_policy: deny` and the whole scheduling
      // feature was unreachable — even `schedule_list`. Autonomous matches the
      // neighbouring assistant-state tools (`memory.remind_at` schedules a
      // future delivery, `memory.forget` deletes user data) and does not grant
      // anything: a scheduled run executes with the configured autonomy, so
      // whatever it later does is policed exactly like an interactive turn.
      schedule_create: {
        mode: 'autonomous',
      },
      schedule_list: {
        mode: 'autonomous',
      },
      schedule_get: {
        mode: 'autonomous',
      },
      schedule_runs: {
        mode: 'autonomous',
      },
      schedule_pause: {
        mode: 'autonomous',
      },
      schedule_resume: {
        mode: 'autonomous',
      },
      schedule_cancel: {
        mode: 'autonomous',
      },
      schedule_cancel_all: {
        mode: 'autonomous',
      },
      schedule_update: {
        mode: 'autonomous',
      },
      schedule_run_now: {
        mode: 'autonomous',
      },
      'notification.publish': {
        mode: 'supervised',
      },
      'monitor.evaluate': {
        mode: 'autonomous',
      },
      'monitor.report': {
        mode: 'autonomous',
      },
      'memory.reminders.list': {
        mode: 'autonomous',
      },
      'memory.reminders.cancel': {
        mode: 'autonomous',
      },
      'memory.documents.delete': {
        mode: 'autonomous',
      },
      'memory.export': {
        mode: 'autonomous',
      },
      'memory.import': {
        mode: 'supervised',
      },
      'memory.tag.suggest': {
        mode: 'autonomous',
      },
      'memory.diff': {
        mode: 'autonomous',
      },
      'memory.documents.get': {
        mode: 'autonomous',
      },
      'memory.documents.update': {
        mode: 'autonomous',
      },
      'memory.usage': {
        mode: 'autonomous',
      },
      'memory.context.snapshot': {
        mode: 'autonomous',
      },
      'memory.conflicts.find': {
        mode: 'autonomous',
      },
      'memory.tag.rename': {
        mode: 'supervised',
      },
      'memory.documents.preview': {
        mode: 'autonomous',
      },
      'memory.daily.list': {
        mode: 'autonomous',
      },
      'memory.daily.search': {
        mode: 'autonomous',
      },
      'memory.search.related': {
        mode: 'autonomous',
      },
      'memory.history': {
        mode: 'autonomous',
      },
      'memory.search.by_tag': {
        mode: 'autonomous',
      },
      'memory.tag.list': {
        mode: 'autonomous',
      },
      'memory.summarize': {
        mode: 'autonomous',
      },
      'memory.access.hot': {
        mode: 'autonomous',
      },
      'memory.access.touch': {
        mode: 'autonomous',
      },
      'memory.pin': {
        mode: 'autonomous',
      },
      'memory.unpin': {
        mode: 'autonomous',
      },
      'memory.pinned.list': {
        mode: 'autonomous',
      },
      'memory.merge': {
        mode: 'supervised',
      },
    },
  }
}

export function withOperatorValidationAllowPatterns(policy: PolicyConfig): PolicyConfig {
  const declared = (process.env.SEPILOTD_VALIDATION_CMD ?? '').trim()
  if (!declared) return policy

  const parts = declared.split(/\s+/).filter(Boolean)
  if (parts.length === 0) return policy

  const pattern = parts.length === 1 ? parts[0] : `${parts[0]} ${parts[1]} *`
  const rule = policy.tools['terminal.run']
  if (!rule) return policy

  const patterns = rule.allow_patterns ?? []
  if (!patterns.includes(pattern)) {
    rule.allow_patterns = [...patterns, pattern]
  }
  return policy
}
