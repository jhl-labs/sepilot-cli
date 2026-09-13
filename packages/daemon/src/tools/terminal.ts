import { spawn } from 'node:child_process'
import { stat } from 'node:fs/promises'
import { basename, isAbsolute, resolve } from 'node:path'
import { AutonomyLevel, type ToolExecutionPosture } from '@sepilotd/core'
import {
  getAbortError,
  isAbortError,
  throwIfAborted,
} from '../abort.js'
import { enforceOutputLimit, resolveToolOutputMaxBytes } from './output-limit.js'
import type {
  ToolDefinitionRuntime,
  ToolResult,
  ToolResumeSafety,
} from './registry.js'

const SAFE_SHELL_WRAPPERS = new Set([
  'bash',
  'dash',
  'sh',
  'zsh',
])

const DISALLOWED_SIMPLE_SHELL_META = /[|;<>\n`$(){}]/
const SHELL_CONTROL_CHARS = new Set(['|', ';', '<', '>', '\n', '`'])

const SAFE_TERMINAL_COMMANDS = new Set([
  'awk',
  'basename',
  'cat',
  'cut',
  'date',
  'df',
  'dirname',
  'du',
  'echo',
  'file',
  'find',
  'grep',
  'head',
  'id',
  'ls',
  'printf',
  'ps',
  'pwd',
  'readlink',
  'realpath',
  'rg',
  'sed',
  'sleep',
  'sort',
  'stat',
  'tail',
  'tr',
  'uname',
  'uniq',
  'wc',
  'which',
  'whoami',
])

const CACHEABLE_TERMINAL_OBSERVATION_COMMANDS = new Set([
  'cat',
  'cut',
  'file',
  'find',
  'git',
  'grep',
  'head',
  'ls',
  'pwd',
  'readlink',
  'realpath',
  'rg',
  'sed',
  'sort',
  'tail',
  'tr',
  'uniq',
  'wc',
])

function cacheableTerminalObservationInvocation(
  executable: string,
  args: readonly string[],
): boolean {
  const command = normalizeExecutableName(executable)
  if (CACHEABLE_TERMINAL_OBSERVATION_COMMANDS.has(command)) return true
  if (command === 'go') {
    return args[0] === 'version' || args[0] === 'env'
  }
  if (!SAFE_SHELL_WRAPPERS.has(command)) return false
  const scriptFlagIndex = args.findIndex((arg) => arg === '-c' || arg === '-lc')
  const script = scriptFlagIndex >= 0 ? args[scriptFlagIndex + 1] : undefined
  if (!script || /[`$(){}<>]/u.test(script)) return false
  const invocations = script
    .split(/(?:&&|\|\||[|;\n])/u)
    .map((segment) => segment.trim())
    .filter(Boolean)
    .map((segment) => segment.split(/\s+/u))
  return invocations.length > 0 && invocations.every(([rawCommand = '', ...commandArgs]) => {
    const nested = normalizeExecutableName(rawCommand)
    return nested === 'cd'
      || nested === 'echo'
      || nested === 'printf'
      || cacheableTerminalObservationInvocation(nested, commandArgs)
  })
}

function terminalObservationIdentity(
  input: Record<string, unknown>,
  contextCwd: string | undefined,
): string | null {
  if (requestsManagedLoopback(input) || requestsExternalNetwork(input)) return null
  const normalized = normalizeTerminalCommandShape(input)
  const executable = typeof normalized.executable === 'string'
    ? normalized.executable.trim()
    : ''
  if (!executable) return null
  const args = normalizeTerminalArgs(normalized.args, executable)
  if (!cacheableTerminalObservationInvocation(executable, args)) return null
  const expectedExitContract = resolveTerminalExpectedExitCodes(normalized)
  if (expectedExitContract.error) return null
  const requestedCwd = typeof normalized.cwd === 'string' && normalized.cwd.trim()
    ? normalized.cwd.trim()
    : contextCwd ?? ''
  const cwd = requestedCwd
    ? resolve(contextCwd ?? process.cwd(), requestedCwd)
    : ''
  const timeoutOutcome = typeof normalized.timeoutOutcome === 'string'
    ? normalized.timeoutOutcome
    : null
  // A normally completed read-only command has the same observation scope
  // regardless of the deadline it happened to receive. Including an omitted
  // versus explicit timeout made byte-equivalent inventory commands evade
  // current-turn reuse. Timed observation modes are different: their window
  // is part of what was observed, so keep both policy and duration there.
  const timedObservation = timeoutOutcome && timeoutOutcome !== 'error'
    ? {
        timeoutMs: normalized.timeoutMs ?? normalized.timeout_ms ?? normalized['timeout-ms'] ?? null,
        timeoutOutcome,
      }
    : {}
  return JSON.stringify({
    executable: normalizeExecutableName(executable),
    args,
    cwd,
    expectedExitCodes: expectedExitContract.codes,
    ...timedObservation,
    network: 'none',
  })
}

const TERMINAL_PLACEHOLDER_PATTERNS: Array<{ pattern: RegExp; reason: string }> = [
  {
    pattern: /\bkubectl\s+create\s+secret\b[\s\S]*\.{3}/i,
    reason: 'kubectl secret command contains an unresolved ellipsis placeholder',
  },
  {
    pattern: /\b--from-(?:literal|file|env-file)=(?:[^\s"'`;=]+=\s*)?\.{3}(?=$|[\s"'`;])/i,
    reason: 'secret input flag contains an unresolved ellipsis placeholder',
  },
  {
    pattern: /\b--from-literal=[^\s"'`;=]+=\s*\.{3}(?=$|[\s"'`;])/i,
    reason: 'secret literal value is an unresolved ellipsis placeholder',
  },
  {
    pattern: /\b(?:password|passwd|secret|token|api[_-]?key|credential|value)\s*=\s*(?:\.{3}|<[^>]+>|YOUR_[A-Z0-9_]*|REPLACE_ME|CHANGEME)(?=$|[\s"'`;])/i,
    reason: 'sensitive value contains a placeholder token',
  },
  {
    pattern: /\b(?:YOUR_[A-Z0-9_]*|REPLACE_ME|CHANGEME|INSERT_[A-Z0-9_]*|TODO_SECRET|TODO_VALUE)\b/i,
    reason: 'command contains a placeholder token',
  },
  {
    pattern: /(?:^|[\s"'=])<(?:your|secret|token|password|api[_-]?key|credential|value)[^>]*>(?=$|[\s"';])/i,
    reason: 'command contains an angle-bracket placeholder token',
  },
]

const GIT_GLOBAL_FLAGS_WITH_VALUE = new Set([
  '-C',
  '-c',
  '--config-env',
  '--exec-path',
  '--git-dir',
  '--namespace',
  '--super-prefix',
  '--work-tree',
])

const SAFE_GIT_SUBCOMMANDS = new Set([
  'diff',
  'grep',
  'log',
  'ls-files',
  'rev-parse',
  'show',
  'status',
])

const KUBECTL_GLOBAL_FLAGS_WITH_VALUE = new Set([
  '--cache-dir',
  '--certificate-authority',
  '--client-certificate',
  '--client-key',
  '--cluster',
  '--context',
  '--kubeconfig',
  '--namespace',
  '--request-timeout',
  '--server',
  '--tls-server-name',
  '--token',
  '--user',
  '-n',
])

const KUBECTL_OUTPUT_FLAGS_WITH_VALUE = new Set([
  '--chunk-size',
  '--field-selector',
  '--label-columns',
  '--output',
  '--selector',
  '--sort-by',
  '--template',
  '-l',
  '-o',
])

const KUBECTL_SENSITIVE_OR_AMBIGUOUS_FLAGS = new Set([
  '--as',
  '--as-group',
  '--certificate-authority',
  '--client-certificate',
  '--client-key',
  '--filename',
  '--insecure-skip-tls-verify',
  '--kubeconfig',
  '--kustomize',
  '--password',
  '--profile',
  '--profile-output',
  '--raw',
  '--server',
  '--tls-server-name',
  '--token',
  '--username',
  '-f',
  '-k',
])

const KUBECTL_READ_ONLY_SUBCOMMANDS = new Set([
  'api-resources',
  'api-versions',
  'describe',
  'events',
  'explain',
  'get',
  'logs',
  'top',
  'version',
  'wait',
])

const TRUSTED_KUBECTL_ABSOLUTE_PATHS = new Set([
  '/usr/bin/kubectl',
  '/usr/local/bin/kubectl',
  '/snap/bin/kubectl',
])

const TRUSTED_TEA_ABSOLUTE_PATHS = new Set([
  '/usr/bin/tea',
  '/usr/local/bin/tea',
  '/snap/bin/tea',
])

const TEA_ACTIONS_RUN_STATUSES = new Set([
  'success',
  'failure',
  'pending',
  'queued',
  'in_progress',
  'skipped',
  'canceled',
])

const TEA_ACTIONS_RUN_LIST_OPTION_NAMES = new Map<string, string>([
  ['--page', 'page'],
  ['-p', 'page'],
  ['--limit', 'limit'],
  ['--lm', 'limit'],
  ['--status', 'status'],
  ['--branch', 'branch'],
  ['--event', 'event'],
  ['--actor', 'actor'],
  ['--since', 'since'],
  ['--until', 'until'],
  ['--remote', 'remote'],
  ['-R', 'remote'],
  ['--output', 'output'],
  ['-o', 'output'],
])

const TEA_ACTIONS_RUN_VIEW_OPTION_NAMES = new Map<string, string>([
  ['--remote', 'remote'],
  ['-R', 'remote'],
  ['--output', 'output'],
  ['-o', 'output'],
])

export type ReadOnlyNetworkTerminalCapability =
  | 'kubectl-readonly'
  | 'gitea-actions-readonly'

function flagName(arg: string): string {
  return arg.split('=', 1)[0]!.toLowerCase()
}

function kubectlSubcommandIndex(args: string[]): number {
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index]!
    if (arg === '--') return -1
    if (!arg.startsWith('-')) return index
    const name = flagName(arg)
    if (!arg.includes('=') && KUBECTL_GLOBAL_FLAGS_WITH_VALUE.has(name)) index += 1
  }
  return -1
}

function kubectlPositionals(args: string[], start: number): string[] {
  const values: string[] = []
  for (let index = start; index < args.length; index += 1) {
    const arg = args[index]!
    if (arg === '--') return []
    if (arg.startsWith('-')) {
      const name = flagName(arg)
      if (
        !arg.includes('=')
        && (KUBECTL_GLOBAL_FLAGS_WITH_VALUE.has(name) || KUBECTL_OUTPUT_FLAGS_WITH_VALUE.has(name))
      ) index += 1
      continue
    }
    values.push(arg.toLowerCase())
  }
  return values
}

function referencesKubernetesSecret(values: string[]): boolean {
  return values.some((value) => value.split(',').some((part) => {
    const resource = part.split('/', 1)[0]!
    return resource === 'secret'
      || resource === 'secrets'
      || resource.startsWith('secret.')
      || resource.startsWith('secrets.')
  }))
}

function kubectlOutputFormats(args: string[]): string[] {
  const formats: string[] = []
  for (let index = 0; index < args.length; index += 1) {
    const normalized = args[index]!.toLowerCase()
    if (normalized === '-o' || normalized === '--output') {
      formats.push((args[index + 1] ?? '').trim().toLowerCase())
      index += 1
      continue
    }
    if (normalized.startsWith('-o=')) {
      formats.push(normalized.slice(3).trim())
      continue
    }
    if (normalized.startsWith('--output=')) {
      formats.push(normalized.slice('--output='.length).trim())
      continue
    }
    // kubectl/pflag also accepts the compact `-oname` spelling.
    if (/^-o[^-]/u.test(normalized)) formats.push(normalized.slice(2).trim())
  }
  return formats
}

/**
 * Secret values remain outside the read-only network capability. The sole
 * exception is an explicit name projection: kubectl owns that renderer and
 * emits only resource identity (`secret/<name>`), including when Secret is
 * one kind in a comma-separated inventory. Do not infer safety from arbitrary
 * jsonpath, template, custom-column, JSON, or YAML expressions.
 */
function isSecretNameOnlyKubectlGet(args: string[]): boolean {
  if (args.some((arg) => flagName(arg) === '--template')) return false
  const formats = kubectlOutputFormats(args)
  return formats.length === 1 && formats[0] === 'name'
}

function isReadOnlyKubectlCommand(args: string[]): boolean {
  if (args.some((arg) => KUBECTL_SENSITIVE_OR_AMBIGUOUS_FLAGS.has(flagName(arg)))) {
    return false
  }

  const subcommandIndex = kubectlSubcommandIndex(args)
  if (subcommandIndex < 0) return false
  const subcommand = args[subcommandIndex]!.toLowerCase()
  const positionals = kubectlPositionals(args, subcommandIndex + 1)

  if (KUBECTL_READ_ONLY_SUBCOMMANDS.has(subcommand)) {
    if (
      (subcommand === 'get' || subcommand === 'describe')
      && referencesKubernetesSecret(positionals)
    ) return subcommand === 'get' && isSecretNameOnlyKubectlGet(args)
    return true
  }

  if (subcommand === 'auth') return positionals[0] === 'can-i'
  if (subcommand === 'config') {
    return ['current-context', 'get-contexts', 'view'].includes(positionals[0] ?? '')
  }
  if (subcommand === 'cluster-info') return positionals.length === 0
  if (subcommand === 'rollout') return ['history', 'status'].includes(positionals[0] ?? '')
  return false
}

const KUBECTL_FAILURE_FANOUT_TARGET_FLAGS = new Set([
  '--cluster',
  '--context',
  '--namespace',
  '--user',
  '-n',
])

/**
 * Preserve the full kubectl invocation template while abstracting only the
 * target scope that a model commonly fans out in one response. A different
 * resource, selector, projection, or output format therefore gets a distinct
 * group and remains executable as a genuine recovery attempt.
 */
export function terminalBatchFailureCouplingKey(
  input: Record<string, unknown>,
): string | null {
  const normalized = normalizeTerminalCommandShape(input)
  const executable = typeof normalized.executable === 'string'
    ? normalized.executable
    : ''
  if (normalizeExecutableName(executable) !== 'kubectl') return null
  const args = normalizeTerminalArgs(normalized.args, executable)
  if (!isReadOnlyKubectlCommand(args)) return null

  const template: string[] = []
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index]!
    const name = flagName(arg)
    if (!KUBECTL_FAILURE_FANOUT_TARGET_FLAGS.has(name)) {
      template.push(arg)
      continue
    }
    if (arg.includes('=')) {
      template.push(`${arg.slice(0, arg.indexOf('='))}=<target>`)
      continue
    }
    template.push(arg, '<target>')
    index += 1
  }
  return JSON.stringify(template)
}

const TERMINAL_INPUT_TEMPLATE_FAILURE_PATTERNS = [
  /(?:^|\n)(?:error:\s*)?(?:unknown|unrecognized|invalid)\s+(?:argument|character|flag|option|output format|shorthand flag)\b/iu,
  /(?:^|\n)(?:error:\s*)?flag provided but not defined\b/iu,
  /(?:^|\n)(?:error:\s*)?(?:flag|option)\s+[^\n]+\s+requires?\s+(?:an?\s+)?argument\b/iu,
  /(?:^|\n)(?:error:\s*)?(?:accepts?|requires?)\s+\d+\s+arg(?:ument)?s?\b/iu,
]

/**
 * Identify client-side argv/template failures, not remote target failures.
 * NotFound, Forbidden, connection, timeout, and ordinary non-zero results do
 * not match and therefore cannot suppress a sibling target observation.
 */
export function isTerminalInputTemplateFailure(result: ToolResult): boolean {
  if (
    result.status !== 'error'
    || /_(?:TRANSIENT|USER)$/u.test(result.code ?? '')
    || /^(?:ABORTED|TIMEOUT)$/u.test(result.code ?? '')
  ) return false
  return TERMINAL_INPUT_TEMPLATE_FAILURE_PATTERNS.some((pattern) => pattern.test(result.output))
}

function isBoundedTeaActionsOptionValue(name: string, value: string): boolean {
  if (!value || value.startsWith('-') || /[\u0000-\u001f\u007f]/u.test(value)) return false
  if (name === 'page') return /^\d{1,3}$/u.test(value) && Number(value) >= 1 && Number(value) <= 100
  if (name === 'limit') return /^\d{1,2}$/u.test(value) && Number(value) >= 1 && Number(value) <= 50
  if (name === 'status') return TEA_ACTIONS_RUN_STATUSES.has(value)
  if (name === 'remote') return /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/u.test(value)
  if (name === 'output') return value === 'json'
  return value.length <= 256
}

/**
 * `tea actions runs` also contains cancel/delete/log and secret-management
 * surfaces. Only bounded list and exact numeric-id view are observational.
 * Repository/login overrides stay outside this capability so the active
 * workspace remote remains the authority boundary.
 */
function isReadOnlyTeaActionsCommand(args: string[]): boolean {
  if (args[0] !== 'actions' || args[1] !== 'runs') return false
  const operation = args[2]
  if (operation !== 'list' && operation !== 'view') return false

  let index = 3
  if (operation === 'view') {
    const runId = args[index]
    if (!runId || !/^\d{1,19}$/u.test(runId) || BigInt(runId) < 1n) return false
    index += 1
  }

  const optionNames = operation === 'list'
    ? TEA_ACTIONS_RUN_LIST_OPTION_NAMES
    : TEA_ACTIONS_RUN_VIEW_OPTION_NAMES
  const seen = new Set<string>()
  while (index < args.length) {
    const raw = args[index]!
    if (raw === '--' || !raw.startsWith('-')) return false
    const equalsIndex = raw.indexOf('=')
    const flag = equalsIndex >= 0 ? raw.slice(0, equalsIndex) : raw
    const name = optionNames.get(flag)
    if (!name || seen.has(name)) return false
    const value = equalsIndex >= 0 ? raw.slice(equalsIndex + 1) : args[index + 1]
    if (!value || !isBoundedTeaActionsOptionValue(name, value)) return false
    seen.add(name)
    index += equalsIndex >= 0 ? 1 : 2
  }
  return true
}

function isTeaActionsRunMetadataSurface(args: string[]): boolean {
  return args[0] === 'actions'
    && args[1] === 'runs'
    && (args[2] === 'list' || args[2] === 'view')
}

function isTrustedTeaExecutable(executable: string): boolean {
  return (!isAbsolute(executable) && executable === 'tea')
    || (isAbsolute(executable) && TRUSTED_TEA_ABSOLUTE_PATHS.has(resolve(executable)))
}

// Retain the public import path while sharing the child-environment policy.
export { scrubChildEnv } from './child-env.js'
import { scrubChildEnv } from './child-env.js'

async function directoryExists(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isDirectory()
  } catch {
    return false
  }
}

function normalizeOptionalCwd(raw: unknown): string | undefined {
  if (typeof raw !== 'string') return undefined
  const trimmed = raw.trim()
  return trimmed ? trimmed : undefined
}

async function resolveTerminalCwd(
  requestedCwd: unknown,
  contextCwd: string | undefined,
): Promise<{ cwd?: string; error?: { code: string; output: string } }> {
  const fallbackCwd = normalizeOptionalCwd(contextCwd)
  const rawCwd = normalizeOptionalCwd(requestedCwd)
  const baseCwd = fallbackCwd && await directoryExists(fallbackCwd)
    ? fallbackCwd
    : process.cwd()

  if (!rawCwd) {
    return { cwd: baseCwd }
  }

  const resolvedCwd = resolve(baseCwd, rawCwd)
  if (await directoryExists(resolvedCwd)) {
    return { cwd: resolvedCwd }
  }

  const hint = fallbackCwd
    ? ` Use the session cwd instead: ${fallbackCwd}, or omit cwd.`
    : ' Omit cwd or run pwd/ls from the default session directory to rediscover the workspace.'
  return {
    error: {
      code: 'CWD_NOT_FOUND_PERMANENT',
      output: `Working directory does not exist: ${resolvedCwd}.${hint}`,
    },
  }
}

async function normalizeTerminalInputCwd(
  input: Record<string, unknown>,
  contextCwd: string | undefined,
): Promise<Record<string, unknown>> {
  const fallbackCwd = normalizeOptionalCwd(contextCwd)
  if (!fallbackCwd || !(await directoryExists(fallbackCwd))) {
    return input
  }

  const rawCwd = normalizeOptionalCwd(input.cwd)
  if (!rawCwd) {
    return { ...input, cwd: fallbackCwd }
  }

  const resolvedCwd = resolve(fallbackCwd, rawCwd)
  if (await directoryExists(resolvedCwd)) {
    return { ...input, cwd: resolvedCwd }
  }

  return { ...input, cwd: fallbackCwd }
}

function normalizeExecutableName(executable: string): string {
  return basename(executable).toLowerCase().replace(/\.exe$/i, '')
}

function hasShellControlSyntax(command: string): boolean {
  let quote: '"' | "'" | null = null
  let escaped = false

  for (let index = 0; index < command.length; index++) {
    const char = command[index]!

    if (escaped) {
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
      }
      continue
    }

    if (char === '"' || char === "'") {
      quote = char
      continue
    }

    if (SHELL_CONTROL_CHARS.has(char)) {
      return true
    }

    if (char === '&' && command[index + 1] === '&') {
      return true
    }
  }

  return false
}

function commandStringInput(
  input: Record<string, unknown>,
  command: string,
): Record<string, unknown> {
  if (hasShellControlSyntax(command)) {
    return {
      ...input,
      executable: 'bash',
      args: ['-lc', command],
    }
  }

  const tokens = tokenizeShellWords(command)
  if (!tokens?.length) {
    return input
  }

  const [executable, ...args] = tokens
  if (!executable) {
    return input
  }

  return {
    ...input,
    executable,
    args: input.args ?? args,
  }
}

function collectTerminalInputText(input: Record<string, unknown>): string[] {
  const normalized = normalizeTerminalCommandShape(input)
  const executable = typeof normalized.executable === 'string'
    ? normalized.executable
    : ''
  const args = normalizeTerminalArgs(normalized.args, executable)
  const texts = [
    executable,
    ...args,
  ].filter((value) => value.trim())

  for (const key of ['cmd', 'command']) {
    const value = normalized[key]
    if (typeof value === 'string' && value.trim()) {
      texts.push(value)
    }
  }

  return texts
}

export function detectTerminalInputPlaceholder(
  input: Record<string, unknown>,
): { reason: string } | null {
  for (const text of collectTerminalInputText(input)) {
    for (const { pattern, reason } of TERMINAL_PLACEHOLDER_PATTERNS) {
      if (pattern.test(text)) {
        return { reason }
      }
    }
  }
  return null
}

function terminalPlaceholderResult(reason: string): ToolResult {
  return {
    output: [
      `terminal.run input contains unresolved placeholder values: ${reason}.`,
      'The command was not executed. Replace placeholders with concrete values, or ask the user for the missing secret/value before running a mutating command.',
    ].join(' '),
    status: 'error',
    durationMs: 0,
    code: 'INVALID_INPUT_PERMANENT',
  }
}

const TERMINAL_SHELL_EXECUTABLES = new Set([
  'bash',
  'cmd',
  'dash',
  'fish',
  'powershell',
  'pwsh',
  'sh',
  'zsh',
])
const STANDALONE_SHELL_CONTROL_ARG_PATTERN =
  /^(?:&&|\|\||[|;]|\d*(?:>>?|<<?)|&>|\d*>&(?:\d+|-))$/u

/**
 * A direct argv invocation is not a shell program.  Off-spec model calls that
 * put `;`, redirects, or pipelines in separate argv elements either fail
 * confusingly or, for wrapper executables such as `npm run`, can be forwarded
 * into a second shell and unexpectedly execute the chain. A shell executable
 * does not make separate argv elements into shell source: the script still
 * has to be one argument after its command-string option (for example,
 * `bash -c "cmd | cmd"`). Fail split control tokens before the runner with a
 * concrete repair.
 */
export function detectDirectArgvShellControl(
  _executable: string,
  args: readonly string[],
): string | null {
  return args.find((arg) => STANDALONE_SHELL_CONTROL_ARG_PATTERN.test(arg.trim())) ?? null
}

function terminalShellControlResult(control: string): ToolResult {
  return {
    output: [
      `terminal.run direct argv contains the shell control token ${JSON.stringify(control)}.`,
      'The command was not executed. Remove redirects, pipelines, chains, and trailing status echoes because terminal.run already captures stdout, stderr, and the exit status; use an explicit shell executable with -c only when a shell script is genuinely required.',
    ].join(' '),
    status: 'error',
    durationMs: 0,
    code: 'INVALID_INPUT_PERMANENT',
  }
}

const TRAILING_STATUS_ECHO_PATTERN = /(?:^|[;&|]\s*)echo\s+(?:-[A-Za-z]+\s+)?[^\n;&|]*\$\?[^\n;&|]*\s*$/u

/**
 * A trailing `echo "$?"` makes the shell itself exit successfully and masks a
 * failed build/test/install. terminal.run already records the real exit code,
 * so reject this structurally misleading wrapper before execution. This is
 * deliberately limited to a final status echo; ordinary shell scripts and
 * diagnostic echoes remain valid.
 */
export function detectTrailingShellStatusEcho(
  executable: string,
  args: readonly string[],
): boolean {
  if (!TERMINAL_SHELL_EXECUTABLES.has(normalizeExecutableName(executable))) {
    return false
  }
  const script = extractShellCommand([...args])
  return typeof script === 'string' && TRAILING_STATUS_ECHO_PATTERN.test(script.trim())
}

function terminalTrailingStatusEchoResult(): ToolResult {
  return {
    output: [
      'terminal.run shell script ends with an echo of $?, which masks the real command exit status.',
      'The command was not executed. Remove the trailing status echo and let terminal.run report stdout, stderr, and the actual exit code. Run independent validation commands separately when each result matters.',
    ].join(' '),
    status: 'error',
    durationMs: 0,
    code: 'INVALID_INPUT_PERMANENT',
  }
}

// execFile's `timeout` option must be a finite non-negative integer. Off-spec
// model output frequently sends it as a string ("60000"), a kebab/snake-cased
// key (`timeout-ms` / `timeout_ms`), or a bare `timeout`. Coerce every shape we
// have seen into a clamped millisecond integer, falling back to the default.
const DEFAULT_TERMINAL_TIMEOUT_MS = 30_000
const MAX_TERMINAL_TIMEOUT_MS = 10 * 60_000 // 10 minutes
const TERMINAL_CAPTURE_EXTRA_BYTES = 8_192
const TERMINAL_TERMINATE_GRACE_MS = 250
const TERMINAL_KILL_SETTLE_MS = 250
const DEFAULT_TERMINAL_EXPECTED_EXIT_CODES = [0] as const
const MAX_TERMINAL_EXPECTED_EXIT_CODES = 16
const MAX_TERMINAL_EXIT_CODE = 255

interface TerminalExpectedExitCodeContract {
  codes: number[]
  explicit: boolean
  error?: string
}

function resolveTerminalExpectedExitCodes(
  input: Record<string, unknown>,
): TerminalExpectedExitCodeContract {
  const raw = input.expectedExitCodes
  if (raw === undefined) {
    return { codes: [...DEFAULT_TERMINAL_EXPECTED_EXIT_CODES], explicit: false }
  }
  if (!Array.isArray(raw)) {
    return {
      codes: [],
      explicit: true,
      error: '`expectedExitCodes` must be a non-empty array of unique integer exit codes.',
    }
  }
  if (raw.length === 0 || raw.length > MAX_TERMINAL_EXPECTED_EXIT_CODES) {
    return {
      codes: [],
      explicit: true,
      error: `\`expectedExitCodes\` must contain between 1 and ${MAX_TERMINAL_EXPECTED_EXIT_CODES} exit codes.`,
    }
  }
  const codes: number[] = []
  for (const value of raw) {
    if (
      typeof value !== 'number'
      || !Number.isInteger(value)
      || value < 0
      || value > MAX_TERMINAL_EXIT_CODE
    ) {
      return {
        codes: [],
        explicit: true,
        error: `Each \`expectedExitCodes\` value must be an integer from 0 through ${MAX_TERMINAL_EXIT_CODE}.`,
      }
    }
    if (codes.includes(value)) {
      return {
        codes: [],
        explicit: true,
        error: '`expectedExitCodes` must not contain duplicate values.',
      }
    }
    codes.push(value)
  }
  return { codes: codes.sort((left, right) => left - right), explicit: true }
}

function terminalExpectedExitCodeInputError(
  input: Record<string, unknown>,
  contract: TerminalExpectedExitCodeContract,
): ToolResult | null {
  const conflict = contract.explicit
    && input.timeoutOutcome !== undefined
    && input.timeoutOutcome !== 'error'
  const output = contract.error
    ?? (conflict
      ? '`expectedExitCodes` requires a normally exited process and cannot be combined with a successful timeout outcome.'
      : null)
  return output
    ? {
        output: `terminal.run input is invalid: ${output} The command was not executed.`,
        status: 'error',
        durationMs: 0,
        code: 'INVALID_INPUT_PERMANENT',
      }
    : null
}

function terminalExitContractOutput(
  output: string,
  exitCode: number,
  expectedExitCodes: readonly number[],
): string {
  const evidence = `[terminal.run exit code: ${exitCode}; expected: ${expectedExitCodes.join(', ')}]`
  return output ? `${output}\n${evidence}` : evidence
}

function terminalExitContractResult(input: {
  output: string
  exitCode: number
  expectedExitCodes: readonly number[]
  durationMs: number
  executionPosture: ToolExecutionPosture
}): ToolResult {
  const matched = input.expectedExitCodes.includes(input.exitCode)
  const inputTemplateFailure = isTerminalInputTemplateFailure({
    output: input.output,
    status: 'error',
    durationMs: input.durationMs,
    code: `EXIT_${input.exitCode}`,
  })
  const accepted = matched && !inputTemplateFailure
  return {
    output: terminalExitContractOutput(
      input.output,
      input.exitCode,
      input.expectedExitCodes,
    ),
    status: accepted ? 'success' : 'error',
    durationMs: input.durationMs,
    ...(accepted
      ? {}
      : {
          code: inputTemplateFailure
            ? 'INPUT_TEMPLATE_FAILURE_PERMANENT'
            : 'UNEXPECTED_EXIT_CODE_PERMANENT',
        }),
    metadata: {
      exitCode: input.exitCode,
      expectedExitCodes: [...input.expectedExitCodes],
      exitCodeMatched: matched,
      ...(inputTemplateFailure ? { inputTemplateFailure: true } : {}),
    },
    executionPosture: input.executionPosture,
  }
}

function isOrdinaryTerminalRunnerExit(result: TerminalRunnerResult): boolean {
  return result.exitCode != null && (
    result.status === 'success'
    || result.code === 'EXIT_NONZERO_PERMANENT'
    || /^EXIT_\d+$/u.test(result.code ?? '')
  )
}

export interface TerminalRunSpec {
  executable: string
  args: string[]
  cwd?: string
  /** Immutable root mounted by strict-workspace runners. */
  workspaceRoot?: string
  timeoutMs: number
  signal?: AbortSignal
  cwdBoundary: ToolExecutionPosture['filesystem']['boundary']
  /** Internal, policy-derived capability. Never accepted directly from tool input. */
  capability?: ReadOnlyNetworkTerminalCapability | 'managed-loopback' | 'workspace-network-write'
  /** Daemon-owned Unix socket bridges for the current session only. */
  managedLoopbackConnections?: Array<{ port: number; socketPath: string }>
}

export interface TerminalRunnerResult {
  stdout: string
  stderr: string
  status: ToolResult['status']
  durationMs: number
  executionPosture: ToolExecutionPosture
  code?: string
  /** Actual child exit code when the sandbox command reached normal process exit. */
  exitCode?: number | null
}

export interface TerminalRunner {
  run(spec: TerminalRunSpec): Promise<TerminalRunnerResult>
}

export interface TerminalToolOptions {
  runner?: TerminalRunner
  /** Fail-closed, read-only runner used whenever a strict workspace is active. */
  strictWorkspaceRunner?: TerminalRunner
  /** Network-isolated, read-write runner used after workspace mutation authority is granted. */
  strictWorkspaceWriteRunner?: TerminalRunner
  /** Strict filesystem sandbox with host network for an audited read-only capability. */
  strictReadOnlyNetworkRunner?: TerminalRunner
  /** Approval-gated workspace-write sandbox with outbound host networking. */
  strictWorkspaceNetworkWriteRunner?: TerminalRunner
  /** Resolve localhost services owned by the active session or strict workspace. */
  getManagedLoopbackConnections?: (
    sessionId: string | undefined,
    workspaceRoot?: string,
  ) => Array<{ port: number; socketPath: string }>
}

function requestsManagedLoopback(input: Record<string, unknown>): boolean {
  if (input.network === 'loopback') return true
  return Boolean(
    input.network
    && typeof input.network === 'object'
    && !Array.isArray(input.network)
    && (input.network as Record<string, unknown>).mode === 'loopback',
  )
}

function requestsExternalNetwork(input: Record<string, unknown>): boolean {
  if (input.network === 'external') return true
  return Boolean(
    input.network
    && typeof input.network === 'object'
    && !Array.isArray(input.network)
    && (input.network as Record<string, unknown>).mode === 'external',
  )
}

export function buildTerminalExecutionPosture(
  cwd: string,
  boundary: ToolExecutionPosture['filesystem']['boundary'] = 'process_cwd',
): ToolExecutionPosture {
  return {
    sandbox: {
      requested: false,
      active: false,
      mode: 'host',
      fallbackReason:
        'terminal.run executes with Node child_process on the host; use the container-sandbox skill for Docker isolation.',
    },
    filesystem: {
      cwd,
      boundary,
      isolated: false,
      readOnly: false,
      note:
        'No filesystem namespace is applied by terminal.run itself; approval policy evaluates the request before execution.',
    },
    network: {
      isolated: false,
      mode: 'host',
    },
  }
}

export function resolveTerminalTimeoutMs(input: Record<string, unknown>): number {
  const raw =
    input.timeoutMs
    ?? input.timeout_ms
    ?? input['timeout-ms']
    ?? input.timeout
  let value: number | null = null
  if (typeof raw === 'number' && Number.isFinite(raw)) {
    value = raw
  } else if (typeof raw === 'string') {
    const trimmed = raw.trim()
    if (/^\d+(?:\.\d+)?$/.test(trimmed)) {
      value = Number.parseFloat(trimmed)
    }
  }
  if (value == null || !Number.isFinite(value) || value <= 0) {
    return DEFAULT_TERMINAL_TIMEOUT_MS
  }
  return Math.min(MAX_TERMINAL_TIMEOUT_MS, Math.max(1, Math.floor(value)))
}

type AcceptedTerminalTimeoutOutcome = 'success_if_output' | 'observation_complete'

function acceptedTerminalTimeoutOutcome(
  input: Record<string, unknown>,
  stdout: string,
): AcceptedTerminalTimeoutOutcome | null {
  if (input.timeoutOutcome === 'observation_complete') return 'observation_complete'
  if (input.timeoutOutcome === 'success_if_output' && stdout.trim().length > 0) {
    return 'success_if_output'
  }
  return null
}

function isTerminalTimeoutCode(code: string | undefined): boolean {
  return code === 'TIMEOUT' || code === 'TIMEOUT_TRANSIENT'
}

function acceptedTimeoutOutput(
  stdout: string,
  timeoutMs: number,
  outcome: AcceptedTerminalTimeoutOutcome,
): string {
  if (stdout.trim().length > 0) return stdout
  if (outcome === 'observation_complete') {
    return `Observation window completed after ${timeoutMs}ms; no stdout was emitted.`
  }
  return stdout
}

function acceptedTimeoutMetadata(
  timeoutMs: number,
  outcome: AcceptedTerminalTimeoutOutcome,
  stdout: string,
): Record<string, unknown> {
  return {
    timedOut: true,
    timeoutOutcome: outcome,
    observationWindowMs: timeoutMs,
    stdoutCaptured: stdout.trim().length > 0,
  }
}

function hasExplicitTerminalArgs(rawArgs: unknown): boolean {
  if (rawArgs == null) {
    return false
  }
  if (Array.isArray(rawArgs)) {
    return rawArgs.length > 0
  }
  if (typeof rawArgs === 'string') {
    return rawArgs.trim().length > 0
  }
  return true
}

export function normalizeTerminalCommandShape(
  input: Record<string, unknown>,
): Record<string, unknown> {
  if (typeof input.executable === 'string' && input.executable.trim()) {
    const executable = input.executable.trim()
    if (!hasExplicitTerminalArgs(input.args) && /\s/.test(executable)) {
      return commandStringInput(input, executable)
    }
    // Some providers echo both the canonical executable/args fields and the
    // legacy command alias. Preserve one unambiguous invocation in policy,
    // approval, evidence, and repeat-detection records.
    const { command: _command, cmd: _cmd, ...canonical } = input
    return { ...canonical, executable }
  }

  const rawCommand = input.cmd ?? input.command
  if (Array.isArray(rawCommand)) {
    const tokens = rawCommand
      .map((value) => String(value).trim())
      .filter(Boolean)
    const [executable, ...args] = tokens
    if (!executable) {
      return input
    }
    return {
      ...input,
      executable,
      args: input.args ?? args,
    }
  }

  if (typeof rawCommand === 'string') {
    const command = rawCommand.trim()
    if (!command) {
      return input
    }
    return commandStringInput(input, command)
  }

  // Off-spec shape some models emit: no `executable` / `command` / `cmd`, only
  // an `args` array that already carries the whole invocation. Treat a single
  // string element as a command line (so `bash -lc "…"` kicks in for shell
  // syntax) and a multi-token array as `[executable, ...rest]`.
  if (typeof input.executable !== 'string' || !input.executable.trim()) {
    if (typeof input.args === 'string' && input.args.trim()) {
      return commandStringInput({ ...input, args: undefined }, input.args.trim())
    }
    if (Array.isArray(input.args) && input.args.length > 0) {
      const tokens = input.args.map((value) => String(value).trim()).filter(Boolean)
      if (tokens.length === 1) {
        return commandStringInput({ ...input, args: undefined }, tokens[0]!)
      }
      const [executable, ...rest] = tokens
      if (executable) {
        return { ...input, executable, args: rest }
      }
    }
  }

  return input
}

/**
 * Normalize a tool-call's `args` field into a string[].
 * Tolerates non-array shapes that some models emit (stringified JSON arrays,
 * raw shell strings, single tokens) so an off-spec model output does not
 * crash execFile with `args must be of type object`.
 */
export function normalizeTerminalArgs(rawArgs: unknown, executable?: string): string[] {
  if (rawArgs == null) return []
  if (Array.isArray(rawArgs)) {
    return normalizeTerminalArgArray(rawArgs.map((arg) => String(arg)), executable)
  }
  if (typeof rawArgs !== 'string') {
    return []
  }
  const trimmed = rawArgs.trim()
  if (!trimmed) return []
  if (trimmed.startsWith('[')) {
    try {
      const parsed = JSON.parse(trimmed)
      if (Array.isArray(parsed)) {
        return normalizeTerminalArgArray(parsed.map((arg) => String(arg)), executable)
      }
    } catch {
      // fall through to whitespace tokenization
    }
  }
  const tokenized = tokenizeSimpleShellCommand(trimmed)
  if (tokenized) return dropRepeatedExecutable(tokenized, executable)
  return trimmed.split(/\s+/).filter(Boolean)
}

function dropRepeatedExecutable(args: string[], executable: string | undefined): string[] {
  if (!executable || args.length === 0) {
    return args
  }

  return normalizeExecutableName(args[0]!) === normalizeExecutableName(executable)
    ? args.slice(1)
    : args
}

function normalizeTerminalArgArray(args: string[], executable: string | undefined): string[] {
  const withoutRepeatedExecutable = dropRepeatedExecutable(args, executable)
  if (withoutRepeatedExecutable.length !== args.length) {
    return withoutRepeatedExecutable
  }

  if (args.length !== 1) {
    return args
  }

  const onlyArg = args[0]?.trim() ?? ''
  if (!onlyArg) {
    return args
  }

  // Smaller prompt-protocol models sometimes select `bash` correctly but put
  // a complete compound script in argv[0] without `-c`. Executing that shape
  // asks bash to open a file literally named "cmd; cmd", producing a confusing
  // ENOENT and another discovery loop. Repair only structurally unambiguous
  // POSIX shell syntax; a single ordinary path remains a legitimate script
  // file argument and is left untouched.
  if (
    executable
    && SAFE_SHELL_WRAPPERS.has(normalizeExecutableName(executable))
    && /(?:&&|\|\||[|;<>\n`$()])/u.test(onlyArg)
  ) {
    return ['-lc', onlyArg]
  }

  const tokenized = tokenizeSimpleShellCommand(onlyArg)
  if (!tokenized || tokenized.length <= 1) {
    return args
  }

  if (
    executable
    && normalizeExecutableName(tokenized[0]!) === normalizeExecutableName(executable)
  ) {
    return tokenized.slice(1)
  }

  return onlyArg.startsWith('-') ? tokenized : args
}

function tokenizeShellWords(command: string): string[] | null {
  if (!command.trim()) {
    return null
  }

  const tokens: string[] = []
  let current = ''
  let quote: '"' | "'" | null = null
  let escaped = false

  for (let index = 0; index < command.length; index++) {
    const char = command[index]!

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
      if (current) {
        tokens.push(current)
        current = ''
      }
      continue
    }

    current += char
  }

  if (escaped || quote) {
    return null
  }

  if (current) {
    tokens.push(current)
  }

  return tokens.length > 0 ? tokens : null
}

function tokenizeSimpleShellCommand(command: string): string[] | null {
  if (DISALLOWED_SIMPLE_SHELL_META.test(command)) {
    return null
  }
  return tokenizeShellWords(command)
}

function splitSimpleShellChain(command: string): string[] | null {
  if (!command.trim()) {
    return null
  }

  const segments: string[] = []
  let current = ''
  let quote: '"' | "'" | null = null
  let escaped = false

  for (let index = 0; index < command.length; index++) {
    const char = command[index]!

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
      }
      current += char
      continue
    }

    if (char === '"' || char === "'") {
      quote = char
      current += char
      continue
    }

    if (char === '&') {
      const next = command[index + 1]
      if (next !== '&') {
        return null
      }

      const trimmed = current.trim()
      if (!trimmed) {
        return null
      }

      segments.push(trimmed)
      current = ''
      index += 1
      continue
    }

    if (DISALLOWED_SIMPLE_SHELL_META.test(char)) {
      return null
    }

    current += char
  }

  if (escaped || quote) {
    return null
  }

  const trailing = current.trim()
  if (!trailing) {
    return null
  }

  segments.push(trailing)
  return segments
}

function extractGitSubcommand(args: string[]): string | null {
  let skipNext = false
  for (const arg of args) {
    if (skipNext) {
      skipNext = false
      continue
    }

    if (GIT_GLOBAL_FLAGS_WITH_VALUE.has(arg)) {
      skipNext = true
      continue
    }

    if (arg.startsWith('-c') && arg !== '-c') {
      continue
    }

    if (arg.startsWith('-')) {
      continue
    }

    return arg.toLowerCase()
  }

  return null
}

function isReplaySafeGitCommand(args: string[]): boolean {
  const subcommand = extractGitSubcommand(args)
  if (!subcommand) {
    return false
  }

  if (SAFE_GIT_SUBCOMMANDS.has(subcommand)) {
    return true
  }

  if (subcommand === 'branch') {
    const branchArgs = args.slice(args.findIndex((arg) => arg.toLowerCase() === subcommand) + 1)
    const nonFlags = branchArgs.filter((arg) => !arg.startsWith('-'))
    return nonFlags.length === 0
  }

  if (subcommand === 'remote') {
    const remoteArgs = args.slice(args.findIndex((arg) => arg.toLowerCase() === subcommand) + 1)
    if (remoteArgs.length === 0) {
      return true
    }
    return ['-v', 'show', 'get-url'].includes(remoteArgs[0]!.toLowerCase())
  }

  return false
}

function extractShellCommand(args: string[]): string | null {
  for (let index = 0; index < args.length; index++) {
    const arg = args[index]!
    if (arg === '-c' || arg === '-lc' || arg === '-cl') {
      const script = args[index + 1]
      if (typeof script !== 'string') {
        return null
      }

      return index + 2 === args.length ? script : null
    }
  }

  return null
}

function classifyCommand(
  executable: string,
  args: string[],
  depth = 0,
): ToolResumeSafety {
  if (depth > 1) {
    return 'replay-risky'
  }

  const command = normalizeExecutableName(executable)
  if (!command) {
    return 'replay-risky'
  }

  if (SAFE_SHELL_WRAPPERS.has(command)) {
    const script = extractShellCommand(args)
    if (!script) {
      return 'replay-risky'
    }

    const segments = splitSimpleShellChain(script)
    if (!segments || segments.length === 0) {
      return 'replay-risky'
    }

    for (const segment of segments) {
      const innerTokens = tokenizeSimpleShellCommand(segment)
      if (!innerTokens || innerTokens.length === 0) {
        return 'replay-risky'
      }

      // Environment-variable prefixes and similar shell-only forms stay risky.
      if (innerTokens[0]!.includes('=')) {
        return 'replay-risky'
      }

      if (
        classifyCommand(
          innerTokens[0]!,
          innerTokens.slice(1),
          depth + 1,
        ) !== 'replay-safe'
      ) {
        return 'replay-risky'
      }
    }

    return 'replay-safe'
  }

  if (command === 'git') {
    return isReplaySafeGitCommand(args) ? 'replay-safe' : 'replay-risky'
  }

  if (command === 'tea') {
    return isReadOnlyTeaActionsCommand(args) ? 'replay-safe' : 'replay-risky'
  }

  if (command === 'sed') {
    const hasInPlaceFlag = args.some(
      (arg) => arg === '-i' || arg === '--in-place' || arg.startsWith('-i'),
    )
    return hasInPlaceFlag ? 'replay-risky' : 'replay-safe'
  }

  return SAFE_TERMINAL_COMMANDS.has(command)
    ? 'replay-safe'
    : 'replay-risky'
}

export function getTerminalResumeSafety(
  input: Record<string, unknown>,
): ToolResumeSafety {
  const executable = typeof input.executable === 'string' ? input.executable : ''
  const args = normalizeTerminalArgs(input.args, executable)

  return classifyCommand(executable, args)
}

/**
 * Conservative network-capability classifier for strict ReadOnly execution.
 * Local commands do not need an executable allowlist because the default
 * strict-workspace runner removes network access and mounts the workspace
 * read-only. This classifier is narrower: a positive result grants host
 * network plus exactly the read-only credentials needed by the capability.
 */
export function resolveReadOnlyNetworkTerminalCapability(
  input: Record<string, unknown>,
): ReadOnlyNetworkTerminalCapability | undefined {
  const normalized = normalizeTerminalCommandShape(input)
  const executable = typeof normalized.executable === 'string' ? normalized.executable : ''
  const command = normalizeExecutableName(executable)
  const args = normalizeTerminalArgs(normalized.args, executable)
  if (
    command === 'kubectl'
    && (
      (!isAbsolute(executable) && executable === 'kubectl')
      || (isAbsolute(executable) && TRUSTED_KUBECTL_ABSOLUTE_PATHS.has(resolve(executable)))
    )
    && isReadOnlyKubectlCommand(args)
  ) return 'kubectl-readonly'
  if (
    command === 'tea'
    && isTrustedTeaExecutable(executable)
    && isReadOnlyTeaActionsCommand(args)
  ) return 'gitea-actions-readonly'
  return undefined
}

export function isReadOnlyNetworkTerminalCommand(input: Record<string, unknown>): boolean {
  return resolveReadOnlyNetworkTerminalCapability(input) !== undefined
}

async function normalizeTerminalInput(
  input: Record<string, unknown>,
  contextCwd: string | undefined,
): Promise<Record<string, unknown>> {
  const withCwd = await normalizeTerminalInputCwd(input, contextCwd)
  const normalizedCommand = normalizeTerminalCommandShape(withCwd)
  if (typeof normalizedCommand.executable !== 'string') {
    return normalizedCommand
  }

  const args = normalizeTerminalArgs(
    normalizedCommand.args,
    normalizedCommand.executable,
  )
  if (normalizedCommand.args == null && args.length === 0) {
    return normalizedCommand
  }

  return { ...normalizedCommand, args }
}

function mergeTerminalOutput(stdout: string, stderr: string): string {
  return stdout + (stderr ? `\n[stderr]\n${stderr}` : '')
}

/**
 * Network-reach failure signatures emitted by common resolvers and runtimes
 * (Node getaddrinfo, glibc, curl, git, Go net) when no network namespace is
 * available. Used only together with the structural execution-posture signal
 * (`network.isolated === true`), never on its own, so an ordinary command
 * failure that merely mentions a hostname does not trigger the hint.
 */
const NETWORK_REACH_FAILURE_PATTERN =
  /(getaddrinfo\s+E|\bENOTFOUND\b|\bEAI_AGAIN\b|\bENETUNREACH\b|Temporary failure in name resolution|Could not resolve host|Name or service not known|nodename nor servname provided|network is unreachable|no route to host|dial tcp: lookup)/i

const NETWORK_ISOLATION_RETRY_HINT =
  '\n[recovery:terminal-network-external-retry]\n[hint] This command ran inside a network-isolated sandbox (network: none), and the failure above is a network-reach error — the tool itself is available. If the command legitimately needs public network access (package install, dependency/registry fetch, crawler), retry the SAME terminal.run call with network: "external". That capability is approval-gated, keeps writes confined to the active workspace, and hides host home/credentials. Do not conclude the task is impossible before trying it.'

/**
 * Read-only filesystem failure signatures (glibc EROFS strings and the raw
 * errno name) emitted when a command mutates a read-only bind mount. Paired
 * with the structural posture signal (`filesystem.readOnly === true`) so an
 * ordinary failure mentioning the phrase does not trigger the hint.
 */
const READ_ONLY_FILESYSTEM_FAILURE_PATTERN = /(Read-only file system|\bEROFS\b)/i

const READ_ONLY_WORKSPACE_MUTATION_HINT =
  '\n[hint] Terminal commands in this sandbox mount the workspace read-only by design — only the terminal mount is read-only, not the workspace itself. To create or modify files use fs.write / fs.edit / fs.append / apply_patch, to move or rename use fs.move, and to delete a file use an apply_patch "*** Delete File" hunk. Do not conclude the filesystem or the task is blocked because an in-sandbox mutation command failed.'

/**
 * Append escalation hints when a sandboxed command failed against one of the
 * sandbox's intentional restrictions. Each hint requires both a structural
 * execution-posture signal (daemon-owned fact) AND a matching failure
 * signature in the captured output. Purely additive — hints never change
 * status, code, or the original output.
 */
export function decorateSandboxRestrictionFailure(
  output: string,
  runResult: {
    status: 'success' | 'error'
    stdout: string
    stderr: string
    executionPosture: ToolExecutionPosture
  },
): string {
  if (runResult.status !== 'error') return output
  const combined = `${runResult.stdout}\n${runResult.stderr}`
  let decorated = output
  if (
    runResult.executionPosture.network.isolated
    && NETWORK_REACH_FAILURE_PATTERN.test(combined)
  ) {
    decorated = `${decorated}${NETWORK_ISOLATION_RETRY_HINT}`
  }
  if (
    runResult.executionPosture.filesystem.readOnly
    && READ_ONLY_FILESYSTEM_FAILURE_PATTERN.test(combined)
  ) {
    decorated = `${decorated}${READ_ONLY_WORKSPACE_MUTATION_HINT}`
  }
  return decorated
}

const SHELL_EXECUTION_NAMES = new Set([
  'bash', 'dash', 'sh', 'zsh', 'fish', 'csh', 'tcsh',
])
const SHELL_CHILD_LAUNCH_FAILURE_PATTERN =
  /(?:^|\n)(?:[^/\s:\n]+\/)*(?:bash|dash|sh|zsh|fish|csh|tcsh):\s*(?:(?:[^:\n\d]+\s+)?\d+:\s*[^:\n]+:\s*\S[^\n]*|[^:\n]+:\s*(?:command not found|not found|No such file or directory|Permission denied))\s*$/imu

function hidesShellChildLaunchFailure(executable: string, stderr: string): boolean {
  const executableName = executable.replace(/\\/g, '/').split('/').at(-1)?.toLowerCase() ?? ''
  return SHELL_EXECUTION_NAMES.has(executableName)
    && SHELL_CHILD_LAUNCH_FAILURE_PATTERN.test(stderr.trim())
}

interface OutputCapture {
  chunks: Buffer[]
  capturedBytes: number
  seenBytes: number
  truncated: boolean
  limitBytes: number
}

interface LocalTerminalProcessResult {
  stdout: string
  stderr: string
  exitCode: number | null
  termSignal: NodeJS.Signals | null
  timedOut: boolean
  durationMs: number
}

function createOutputCapture(limitBytes: number): OutputCapture {
  return {
    chunks: [],
    capturedBytes: 0,
    seenBytes: 0,
    truncated: false,
    limitBytes,
  }
}

function appendOutputCapture(capture: OutputCapture, chunk: Buffer): void {
  capture.seenBytes += chunk.byteLength
  const remaining = capture.limitBytes - capture.capturedBytes
  if (remaining <= 0) {
    capture.truncated = true
    return
  }

  if (chunk.byteLength <= remaining) {
    capture.chunks.push(chunk)
    capture.capturedBytes += chunk.byteLength
    return
  }

  capture.chunks.push(chunk.subarray(0, remaining))
  capture.capturedBytes += remaining
  capture.truncated = true
}

/**
 * Legacy console encodings to try when a child's output is not valid UTF-8.
 *
 * Windows console programs write in the machine's ANSI/OEM code page, not
 * UTF-8, so `cmd`/`powershell` output containing non-ASCII text arrives as
 * mojibake — `영수증 회의록` came back as `������ ȸ�Ƿ�`, and the agent then
 * could not read the result of its own command. Ordered by the locale the
 * daemon is running under, so a Korean machine tries cp949 first.
 */
const LEGACY_CONSOLE_ENCODINGS_BY_LANGUAGE: Record<string, string> = {
  ko: 'euc-kr',
  ja: 'shift_jis',
  zh: 'gbk',
}

export function decodeConsoleOutput(buffer: Buffer): string {
  if (buffer.byteLength === 0) return ''
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(buffer)
  } catch {
    // Not UTF-8. Fall back to the console encoding this machine's locale
    // implies, then to windows-1252, which decodes every byte rather than
    // leaving the caller with replacement characters.
    const language = (() => {
      try {
        return Intl.DateTimeFormat().resolvedOptions().locale.split('-')[0] ?? ''
      } catch {
        return ''
      }
    })()
    // Locale is often inherited from a Linux host even when the child is a
    // Windows console program (for example through a remote shell).  Try the
    // CJK encodings after the locale-specific choice so Korean output remains
    // legible instead of being silently decoded as Windows-1252 mojibake.
    const candidates = [
      LEGACY_CONSOLE_ENCODINGS_BY_LANGUAGE[language],
      'euc-kr',
      'shift_jis',
      'gbk',
      'windows-1252',
    ]
    for (const encoding of candidates) {
      if (!encoding) continue
      try {
        return new TextDecoder(encoding).decode(buffer)
      } catch {
        /* try the next one */
      }
    }
    return buffer.toString('utf8')
  }
}

function finalizeOutputCapture(capture: OutputCapture, streamName: 'stdout' | 'stderr'): string {
  const output = decodeConsoleOutput(Buffer.concat(capture.chunks, capture.capturedBytes))
  if (!capture.truncated) {
    return output
  }

  return [
    output,
    `[terminal.run ${streamName}: stream capture stopped at ${capture.limitBytes} bytes; command produced at least ${capture.seenBytes} bytes. Pipe through head/tail/grep, redirect to a file, or run a more specific command.]`,
  ].filter(Boolean).join('\n')
}

async function runLocalTerminalProcess(spec: {
  executable: string
  args: string[]
  cwd?: string
  timeoutMs: number
  signal?: AbortSignal
}): Promise<LocalTerminalProcessResult> {
  const start = Date.now()
  const captureLimit = resolveToolOutputMaxBytes('terminal.run') + TERMINAL_CAPTURE_EXTRA_BYTES
  const stdout = createOutputCapture(captureLimit)
  const stderr = createOutputCapture(captureLimit)

  return new Promise((resolvePromise, reject) => {
    if (spec.signal?.aborted) {
      reject(getAbortError(spec.signal, `Command ${spec.executable} aborted`))
      return
    }

    // Own one process group per foreground invocation on POSIX. A shell can
    // exit while a descendant keeps stdout/stderr open; signalling only the
    // wrapper PID then leaves this Promise waiting past its declared timeout.
    // Windows child.kill already targets the spawned process through the
    // platform process API and does not support negative process-group PIDs.
    const isolatedProcessGroup = process.platform !== 'win32'
    const child = spawn(spec.executable, spec.args, {
      cwd: spec.cwd,
      env: scrubChildEnv(process.env),
      detached: isolatedProcessGroup,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let settled = false
    let timedOut = false
    let aborted = false
    let terminationStarted = false
    let timeout: ReturnType<typeof setTimeout> | null = null
    let killTimer: ReturnType<typeof setTimeout> | null = null
    let settleTimer: ReturnType<typeof setTimeout> | null = null

    const clearTimers = () => {
      if (timeout) clearTimeout(timeout)
      if (killTimer) clearTimeout(killTimer)
      if (settleTimer) clearTimeout(settleTimer)
    }

    const signalProcessTree = (signal: NodeJS.Signals) => {
      if (isolatedProcessGroup && child.pid) {
        try {
          process.kill(-child.pid, signal)
          return
        } catch (error) {
          // The leader may have exited before its descendants released the
          // pipes. ESRCH means the group is already gone; other failures fall
          // back to the direct child without turning cleanup into a crash.
          if ((error as NodeJS.ErrnoException).code === 'ESRCH') return
        }
      }
      try {
        child.kill(signal)
      } catch {
        // A concurrently exiting process needs no further action.
      }
    }

    const settle = (exitCode: number | null, termSignal: NodeJS.Signals | null) => {
      if (settled) return
      settled = true
      clearTimers()
      spec.signal?.removeEventListener('abort', onAbort)
      if (aborted || spec.signal?.aborted) {
        reject(getAbortError(spec.signal, `Command ${spec.executable} aborted`))
        return
      }
      resolvePromise({
        stdout: finalizeOutputCapture(stdout, 'stdout'),
        stderr: finalizeOutputCapture(stderr, 'stderr'),
        exitCode,
        termSignal,
        timedOut,
        durationMs: Date.now() - start,
      })
    }

    const settleReject = (error: unknown) => {
      if (settled) return
      settled = true
      clearTimers()
      spec.signal?.removeEventListener('abort', onAbort)
      reject(error)
    }

    const beginTermination = () => {
      if (terminationStarted || settled) return
      terminationStarted = true
      signalProcessTree('SIGTERM')
      killTimer = setTimeout(() => {
        signalProcessTree('SIGKILL')
        // Node normally emits close after SIGKILL. Keep a final daemon-owned
        // bound for platform/runtime edge cases so one missing close event can
        // never strand the agent run lease or a detached mobile chat job.
        settleTimer = setTimeout(() => {
          child.stdout.destroy()
          child.stderr.destroy()
          settle(null, 'SIGKILL')
        }, TERMINAL_KILL_SETTLE_MS)
        settleTimer.unref?.()
      }, TERMINAL_TERMINATE_GRACE_MS)
      killTimer.unref?.()
    }

    const onAbort = () => {
      if (aborted || settled) return
      aborted = true
      beginTermination()
    }

    timeout = setTimeout(() => {
      if (timedOut || settled) return
      timedOut = true
      beginTermination()
    }, spec.timeoutMs)
    timeout.unref?.()
    spec.signal?.addEventListener('abort', onAbort, { once: true })

    child.stdout.on('data', (chunk: Buffer) => appendOutputCapture(stdout, chunk))
    child.stderr.on('data', (chunk: Buffer) => appendOutputCapture(stderr, chunk))
    child.on('error', settleReject)
    child.on('close', settle)
  })
}

export function createTerminalTool(options: TerminalToolOptions = {}): ToolDefinitionRuntime {
  return {
    name: 'terminal.run',
    unavailableReason: (context) => context.workspaceRoot?.trim()
      && !options.strictWorkspaceRunner && !options.strictWorkspaceWriteRunner
      && !options.strictReadOnlyNetworkRunner && !options.strictWorkspaceNetworkWriteRunner
      ? 'No strict-workspace terminal runner is configured on this host. Use available file observation tools for file read-back; command execution is unavailable in this boundary.'
      : undefined,
    description:
      'Execute one program through the configured terminal runner. Always pass `executable` as the program name and `args` as an argv array. Shell builtins such as `command`, pipelines (|), redirects (>), or `&&`/`;` chains require an explicit `bash -c "<script>"` wrapper. Common installed developer toolchains are projected read-only into an isolated sandbox PATH while host credentials remain hidden: invoke the intended tool directly once. If the result is `EXECUTABLE_NOT_FOUND_PERMANENT`, treat that tool as unavailable; do not scan `/`, probe unrelated directories, or repeat the same lookup. A depth-, count-, or line-bounded inventory such as `find -maxdepth`, `head`, or `--max-count` proves only the requested scope; never infer that deeper or omitted files do not exist. For a negative test or health-state command that intentionally exits non-zero, declare the acceptable codes in `expectedExitCodes`; do not hide the real result with `|| true` or a status echo. Use `timeoutMs` to bound observation instead of wrapping the command with shell `timeout` or a trailing status echo. For a watch/follow command where silence is a valid "no changes" result, use `timeoutOutcome: observation_complete`; reaching the deadline then succeeds even with no stdout. Use `success_if_output` only when captured stdout is required for a usable bounded-observation result. Both modes are for commands intentionally expected to remain running until the deadline; commands that must exit normally to prove completion must keep the default `error`. When the user asks for changes during the observation window, prefer the command\'s native changes-only/watch-only option so an initial state replay is not reported as a new change. If a before/after comparison genuinely requires a delay, call `sleep` directly with a finite duration; do not wrap it in a shell. After an observation completes, interpret its output (including explicit no-output completion) instead of adding another delay or repeating the same observation. Under ReadOnly autonomy, a strict workspace lets local commands run in a fail-closed read-only, no-network sandbox; audited remote observation capabilities may receive narrower read-only network access. A direct classified read-only Kubernetes inspection is automatically confined to the read-only network runner even if the call includes `network: external`; shell wrappers, mutations, and Secret value reads never receive that downgrade. To run curl or another API client against a development server previously started with process.start managed loopback ports and available to the same strict workspace, set network to `loopback`; only those capability-bound ports become reachable and external network access stays disabled. For a user-authorized generator, crawler, package installer, or integration command that must both write inside the active workspace and reach public network services, set network to `external`; this uses a workspace-confined bubblewrap sandbox with hidden host home/credentials and requires explicit approval unless a narrower audited read-only capability applies. Prefer dedicated tools when one fits: fs.read/fs.search for files, git.* for repo state, code.* for symbols, and webfetch for a simple GET. Reach for terminal.run only when no other tool covers the job. Legacy `command`/`cmd` inputs are accepted at runtime for compatibility but are intentionally omitted from the model schema.',
    resumeSafety: 'replay-risky',
    resumeSafetyForInput: getTerminalResumeSafety,
    batchFailureCoupling: {
      groupKey: terminalBatchFailureCouplingKey,
      blocksSiblings: isTerminalInputTemplateFailure,
    },
    normalizeInput: (input, context) =>
      normalizeTerminalInput(input, context.cwd),
    observationCoverage: {
      covers: (observed, requested, context) => {
        const observedIdentity = terminalObservationIdentity(observed, context.cwd)
        return observedIdentity !== null
          && observedIdentity === terminalObservationIdentity(requested, context.cwd)
      },
    },
    validateInput: (input) => {
      const placeholder = detectTerminalInputPlaceholder(input)
      if (placeholder) return terminalPlaceholderResult(placeholder.reason)
      const normalized = normalizeTerminalCommandShape(input)
      const executable = typeof normalized.executable === 'string' ? normalized.executable : ''
      if (executable && detectTrailingShellStatusEcho(
        executable,
        normalizeTerminalArgs(normalized.args, executable),
      )) {
        return terminalTrailingStatusEchoResult()
      }
      const shellControl = executable
        ? detectDirectArgvShellControl(
            executable,
            normalizeTerminalArgs(normalized.args, executable),
          )
        : null
      if (shellControl) return terminalShellControlResult(shellControl)
      const expectedExitContract = resolveTerminalExpectedExitCodes(normalized)
      return terminalExpectedExitCodeInputError(normalized, expectedExitContract)
    },
    inputSchema: {
      type: 'object',
      properties: {
        executable: {
          type: 'string',
          description: 'Exact program to execute, such as `python3`, `go`, or `curl`. The runner does not parse a command line. For shell syntax, set this to `bash` and put `-c` plus the complete script in `args`; do not set it to `bash` merely to invoke another program.',
        },
        args: {
          type: 'array',
          items: { type: 'string' },
          description: 'Argument vector passed verbatim to the executable, one array item per argv element. Redirects, pipelines, and shell operators are not interpreted unless the executable is an explicit shell with `-c`.',
        },
        cwd: { type: 'string', description: 'Working directory. Omit to use the active session cwd; do not invent placeholder paths.' },
        timeoutMs: { type: 'number', description: 'Timeout in milliseconds' },
        timeoutOutcome: {
          type: 'string',
          enum: ['error', 'success_if_output', 'observation_complete'],
          description: 'How an execution deadline is classified. Default `error`. Use `success_if_output` when captured stdout is required for a usable bounded-observation result. Use `observation_complete` for an intentional watch/follow window where no stdout validly means no observed changes.',
        },
        expectedExitCodes: {
          type: 'array',
          items: { type: 'integer', minimum: 0, maximum: MAX_TERMINAL_EXIT_CODE },
          minItems: 1,
          maxItems: MAX_TERMINAL_EXPECTED_EXIT_CODES,
          uniqueItems: true,
          description: 'Exact normal process exit codes that satisfy this invocation. Omit for the default `[0]`. Declare intentional non-zero outcomes here instead of masking them with `|| true`, a trailing status echo, or a shell wrapper. Timeout, signal, launch, and sandbox failures never satisfy this contract.',
        },
        network: {
          type: 'string',
          enum: ['none', 'loopback', 'external'],
          description: 'Optional network capability. `loopback` reaches only localhost ports exposed by process.start in this same session. Direct audited read-only Kubernetes or Gitea Actions run inspection is narrowed to a read-only host-network runner with only its exact daemon-owned credential file. Other `external` calls are approval-gated and run with public network plus write access confined to the active workspace; host home and credentials stay hidden.',
        },
        actionPurpose: {
          type: 'string',
          enum: ['observe', 'mutate', 'validate', 'unblock'],
          description: 'Semantic purpose of this command. Use `observe` for inspection and diagnostic captures even when they write temporary/workspace samples; `mutate` only when the intended result is a durable user-requested product/source/data/config/test artifact; `validate` for tests, builds, formatting checks, and audits even when they emit caches or reports; `unblock` for dependency or environment preparation. This declaration does not weaken filesystem or approval policy.',
        },
      },
      required: ['executable', 'actionPurpose'],
    },
    async execute(input: Record<string, unknown>, context): Promise<ToolResult> {
      const normalizedInput = normalizeTerminalCommandShape(input)
      const expectedExitContract = resolveTerminalExpectedExitCodes(normalizedInput)
      const expectedExitInputError = terminalExpectedExitCodeInputError(
        normalizedInput,
        expectedExitContract,
      )
      if (expectedExitInputError) return expectedExitInputError
      const placeholder = detectTerminalInputPlaceholder(normalizedInput)
      if (placeholder) {
        return terminalPlaceholderResult(placeholder.reason)
      }
      const executable = typeof normalizedInput.executable === 'string'
        ? normalizedInput.executable
        : ''
      if (!executable) {
        return {
          output: 'Missing terminal executable. Provide executable/args or a simple cmd/command value.',
          status: 'error',
          durationMs: 0,
          code: 'INVALID_INPUT_PERMANENT',
        }
      }
      const args = normalizeTerminalArgs(normalizedInput.args, executable)
      const shellControl = detectDirectArgvShellControl(executable, args)
      if (shellControl) {
        return terminalShellControlResult(shellControl)
      }
      if (
        isTrustedTeaExecutable(executable)
        && isTeaActionsRunMetadataSurface(args)
        && !isReadOnlyTeaActionsCommand(args)
      ) {
        return {
          output: 'This Gitea Actions metadata query is outside the credential-safe read-only contract. Use a local workspace remote alias such as `origin`, never a URL: `tea actions runs list --remote <alias> --limit 20 --output json` or `tea actions runs view <positive-id> --remote <same-alias> --output json`. Repository/login overrides, --jobs/action-log surfaces, management/debug flags, and non-JSON output are unavailable.',
          status: 'error',
          durationMs: 0,
          code: 'GITEA_ACTIONS_READ_CONTRACT_PERMANENT',
        }
      }
      const timeoutMs = resolveTerminalTimeoutMs(normalizedInput)

      const start = Date.now()
      let executionPosture = buildTerminalExecutionPosture(process.cwd())
      try {
        throwIfAborted(context?.signal, `Command ${executable} aborted`)
        const cwdResolution = await resolveTerminalCwd(normalizedInput.cwd, context?.cwd)
        if (cwdResolution.error) {
          return {
            output: cwdResolution.error.output,
            status: 'error',
            durationMs: Date.now() - start,
            code: cwdResolution.error.code,
          }
        }
        const cwdBoundary = normalizeOptionalCwd(normalizedInput.cwd)
          ? 'tool_specific'
          : context?.cwd
            ? 'session_cwd'
            : 'process_cwd'
        executionPosture = buildTerminalExecutionPosture(
          cwdResolution.cwd ?? process.cwd(),
          cwdBoundary,
        )
        const strictWorkspaceRoot = context?.workspaceRoot?.trim() || undefined
        const managedLoopbackRequested = requestsManagedLoopback(normalizedInput)
        const externalNetworkRequested = requestsExternalNetwork(normalizedInput)
        if (managedLoopbackRequested && externalNetworkRequested) {
          return {
            output: 'Choose exactly one terminal network capability: loopback or external.',
            status: 'error',
            durationMs: Date.now() - start,
            code: 'INVALID_INPUT_PERMANENT',
          }
        }
        if (externalNetworkRequested && !strictWorkspaceRoot) {
          return {
            output: 'External network terminal access requires an active strict workspace.',
            status: 'error',
            durationMs: Date.now() - start,
            code: 'STRICT_WORKSPACE_REQUIRED_PERMANENT',
          }
        }
        const managedLoopbackConnections = managedLoopbackRequested
          ? options.getManagedLoopbackConnections?.(
              context?.sessionId,
              context?.workspaceRoot,
            ) ?? []
          : []
        if (managedLoopbackRequested && !strictWorkspaceRoot) {
          return {
            output: 'Managed loopback terminal access requires an active strict workspace.',
            status: 'error',
            durationMs: Date.now() - start,
            code: 'STRICT_WORKSPACE_REQUIRED_PERMANENT',
          }
        }
        if (managedLoopbackRequested && managedLoopbackConnections.length === 0) {
          return {
            output: 'No running localhost service with managed loopback ports is available to this session or strict workspace. Start one with process.start network { mode: loopback, ports: [...] } first.',
            status: 'error',
            durationMs: Date.now() - start,
            code: 'MANAGED_LOOPBACK_UNAVAILABLE_PERMANENT',
          }
        }
        const readOnlyNetworkCapability = strictWorkspaceRoot
          ? resolveReadOnlyNetworkTerminalCapability(normalizedInput)
          : undefined
        const readOnlyNetworkCommand = readOnlyNetworkCapability !== undefined
        const workspaceWriteAuthorized = Boolean(
          context?.delegatedAgentPolicy
          && context.delegatedAgentPolicy.autonomy !== AutonomyLevel.ReadOnly,
        )
        const effectiveRunner = strictWorkspaceRoot
          ? readOnlyNetworkCommand
            ? options.strictReadOnlyNetworkRunner
            : externalNetworkRequested
              ? options.strictWorkspaceNetworkWriteRunner
              : managedLoopbackRequested
                ? options.strictWorkspaceRunner
                : workspaceWriteAuthorized
                  ? options.strictWorkspaceWriteRunner
                  : options.strictWorkspaceRunner
          : options.runner
        const workspaceReadOnlyExpected = !(
          (externalNetworkRequested && !readOnlyNetworkCommand)
          || (
            workspaceWriteAuthorized
            && !readOnlyNetworkCommand
            && !managedLoopbackRequested
          )
        )
        const effectiveCwdBoundary = strictWorkspaceRoot
          ? 'strict_workspace'
          : cwdBoundary
        if (strictWorkspaceRoot && !effectiveRunner) {
          return {
            output: 'Strict workspace terminal execution requires an active read-only workspace sandbox, but no supported sandbox runner is available on this host.',
            status: 'error',
            durationMs: Date.now() - start,
            code: 'SANDBOX_UNAVAILABLE',
            executionPosture: {
              sandbox: {
                requested: true,
                active: false,
                mode: 'host',
                fallbackReason: 'No strict-workspace terminal runner is configured.',
              },
              filesystem: {
                cwd: cwdResolution.cwd ?? strictWorkspaceRoot,
                boundary: effectiveCwdBoundary,
                isolated: false,
                readOnly: false,
                note: 'The command was not executed on the host.',
              },
              network: { isolated: false, mode: 'host' },
            },
          }
        }
        if (effectiveRunner) {
          const runResult = await effectiveRunner.run({
            executable,
            args,
            cwd: cwdResolution.cwd,
            ...(strictWorkspaceRoot ? { workspaceRoot: strictWorkspaceRoot } : {}),
            timeoutMs,
            signal: context?.signal,
            cwdBoundary: effectiveCwdBoundary,
            ...(readOnlyNetworkCapability ? { capability: readOnlyNetworkCapability } : {}),
            ...(externalNetworkRequested && !readOnlyNetworkCommand
              ? { capability: 'workspace-network-write' as const }
              : {}),
            ...(managedLoopbackRequested ? {
              capability: 'managed-loopback' as const,
              managedLoopbackConnections,
            } : {}),
          })
          if (
            strictWorkspaceRoot
            && (
              !runResult.executionPosture.sandbox.active
              || !runResult.executionPosture.filesystem.isolated
              || runResult.executionPosture.filesystem.readOnly !== workspaceReadOnlyExpected
              || (!readOnlyNetworkCommand
                && !externalNetworkRequested
                && !runResult.executionPosture.network.isolated)
              || (readOnlyNetworkCommand && runResult.executionPosture.network.mode !== 'host')
              || (externalNetworkRequested && runResult.executionPosture.network.mode !== 'host')
            )
          ) {
            return {
              output: runResult.stderr.trim()
                || 'Strict workspace terminal sandbox did not become active; the command was not run on the host.',
              status: 'error',
              durationMs: runResult.durationMs,
              code: runResult.code ?? 'SANDBOX_UNAVAILABLE',
              executionPosture: runResult.executionPosture,
            }
          }
          const acceptedTimeoutOutcome = isTerminalTimeoutCode(runResult.code)
            ? acceptedTerminalTimeoutOutcome(normalizedInput, runResult.stdout)
            : null
          const limited = enforceOutputLimit(
            acceptedTimeoutOutcome
              ? acceptedTimeoutOutput(runResult.stdout, timeoutMs, acceptedTimeoutOutcome)
              : mergeTerminalOutput(runResult.stdout, runResult.stderr),
            {
              toolName: 'terminal.run',
              resumeHint: readOnlyNetworkCapability === 'kubectl-readonly'
                ? 'run a narrower direct kubectl query with --field-selector, -l, -o jsonpath, or --tail; shell wrappers and pipelines are blocked in ReadOnly autonomy'
                : readOnlyNetworkCapability === 'gitea-actions-readonly'
                  ? 'run tea actions runs list with a smaller --limit or view one exact numeric run id using the same local --remote alias (never a URL); --jobs/action-log surfaces, repository/login overrides, and management commands are blocked in ReadOnly autonomy'
                : 'pipe through head/tail/grep, redirect to a file, or run a more specific command',
            },
          )
          if (
            acceptedTimeoutOutcome
          ) {
            return {
              output: limited.output,
              status: 'success',
              durationMs: runResult.durationMs,
              metadata: acceptedTimeoutMetadata(
                timeoutMs,
                acceptedTimeoutOutcome,
                runResult.stdout,
              ),
              executionPosture: runResult.executionPosture,
            }
          }
          if (
            runResult.status === 'success'
            && hidesShellChildLaunchFailure(executable, runResult.stderr)
          ) {
            return {
              output: limited.output,
              status: 'error',
              durationMs: runResult.durationMs,
              code: 'SHELL_CHILD_FAILURE_PERMANENT',
              executionPosture: runResult.executionPosture,
            }
          }
          if (expectedExitContract.explicit) {
            if (!isOrdinaryTerminalRunnerExit(runResult)) {
              if (runResult.exitCode == null && runResult.status === 'success') {
                return {
                  output: `${limited.output}${limited.output ? '\n' : ''}[terminal.run could not verify the actual exit code required by expectedExitCodes]`,
                  status: 'error',
                  durationMs: runResult.durationMs,
                  code: 'EXIT_CODE_UNAVAILABLE_PERMANENT',
                  executionPosture: runResult.executionPosture,
                }
              }
            } else {
              return terminalExitContractResult({
                output: limited.output,
                exitCode: runResult.exitCode!,
                expectedExitCodes: expectedExitContract.codes,
                durationMs: runResult.durationMs,
                executionPosture: runResult.executionPosture,
              })
            }
          }
          return {
            output: decorateSandboxRestrictionFailure(limited.output, runResult),
            status: runResult.status,
            durationMs: runResult.durationMs,
            ...(runResult.code ? { code: runResult.code } : {}),
            executionPosture: runResult.executionPosture,
          }
        }
        const runResult = await runLocalTerminalProcess({
          executable,
          args,
          cwd: cwdResolution.cwd,
          timeoutMs,
          signal: context?.signal,
        })
        const merged = mergeTerminalOutput(runResult.stdout, runResult.stderr)
        const limited = enforceOutputLimit(merged, {
          toolName: 'terminal.run',
          resumeHint:
            'pipe through head/tail/grep, redirect to a file, or run a more specific command',
        })
        if (runResult.timedOut) {
          const acceptedTimeoutOutcome = acceptedTerminalTimeoutOutcome(
            normalizedInput,
            runResult.stdout,
          )
          if (acceptedTimeoutOutcome) {
            return {
              output: enforceOutputLimit(acceptedTimeoutOutput(
                runResult.stdout,
                timeoutMs,
                acceptedTimeoutOutcome,
              ), {
                toolName: 'terminal.run',
                resumeHint: 'run a more specific command',
              }).output,
              status: 'success',
              durationMs: runResult.durationMs,
              metadata: acceptedTimeoutMetadata(
                timeoutMs,
                acceptedTimeoutOutcome,
                runResult.stdout,
              ),
              executionPosture,
            }
          }
          return {
            output: limited.output || `Command ${executable} timed out after ${timeoutMs}ms`,
            status: 'error',
            durationMs: runResult.durationMs,
            code: 'TIMEOUT_TRANSIENT',
            executionPosture,
          }
        }
        if (hidesShellChildLaunchFailure(executable, runResult.stderr)) {
          return {
            output: limited.output,
            status: 'error',
            durationMs: runResult.durationMs,
            code: 'SHELL_CHILD_FAILURE_PERMANENT',
            executionPosture,
          }
        }
        if (expectedExitContract.explicit && runResult.exitCode != null) {
          return terminalExitContractResult({
            output: limited.output,
            exitCode: runResult.exitCode,
            expectedExitCodes: expectedExitContract.codes,
            durationMs: runResult.durationMs,
            executionPosture,
          })
        }
        if (expectedExitContract.explicit && runResult.exitCode == null) {
          return {
            output: `${limited.output}${limited.output ? '\n' : ''}[terminal.run process ended without a verifiable exit code]`,
            status: 'error',
            durationMs: runResult.durationMs,
            code: 'EXIT_CODE_UNAVAILABLE_PERMANENT',
            executionPosture,
          }
        }
        if (runResult.exitCode !== 0) {
          return {
            output: limited.output
              || `Command ${executable} exited with code ${runResult.exitCode ?? 'null'}${runResult.termSignal ? ` signal ${runResult.termSignal}` : ''}`,
            status: 'error',
            durationMs: runResult.durationMs,
            code: 'EXIT_NONZERO_PERMANENT',
            executionPosture,
          }
        }
        return {
          output: limited.output,
          status: 'success',
          durationMs: runResult.durationMs,
          executionPosture,
        }
      } catch (err: unknown) {
        if (isAbortError(err) || context?.signal?.aborted) {
          throw getAbortError(context?.signal, `Command ${executable} aborted`)
        }
        const e = err as { stderr?: string; message?: string; code?: string | number; killed?: boolean }
        // execFile sets `killed=true` and code='SIGTERM'/null when the
        // child was killed by the timeout — that's the canonical
        // transient signal. ENOENT/EACCES from spawn (not from the
        // child process exit code) are permanent: a different executable
        // is needed. A non-zero exit code with no other context is
        // permanent for this invocation but the agent may still try
        // different args.
        const code = e.killed
          ? 'TIMEOUT_TRANSIENT'
          : e.code === 'ENOENT'
            ? 'ENOENT_PERMANENT'
            : e.code === 'EACCES' || e.code === 'EPERM'
              ? 'EACCES_PERMANENT'
              : 'EXIT_NONZERO_PERMANENT'
        const limited = enforceOutputLimit(e.stderr ?? e.message ?? String(err), {
          toolName: 'terminal.run',
          resumeHint:
            'pipe through head/tail/grep, redirect to a file, or run a more specific command',
        })
        return {
          output: limited.output,
          status: 'error',
          durationMs: Date.now() - start,
          code,
          executionPosture,
        }
      }
    },
  }
}
