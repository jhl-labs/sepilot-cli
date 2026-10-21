import { basename, isAbsolute, relative, resolve, sep } from 'node:path'

export function terminalAutoApprovalEscalation(input: {
  executable: string
  args: readonly string[]
  cwd?: string
  requestCwd?: string
  isProtectedPath: (path: string, cwd: string | undefined) => boolean
}): string | null {
  const executable = basename(input.executable).toLowerCase().replace(/\.exe$/, '')
  const args = input.args
  const cwd = input.cwd ?? input.requestCwd
  if (executable === 'git') return gitEscalation(args)
  if (executable === 'find') {
    const action = args.find((arg) => FIND_EXECUTING_OR_WRITING_ACTIONS.has(arg))
    if (action) return `find ${action} runs a command or writes files`
  }
  if (!READER_EXECUTABLES.has(executable)) return null

  if (cwd && input.isProtectedPath(resolve(cwd, '.protected-path-probe'), undefined)) {
    return `${executable} runs inside a protected directory`
  }
  const operands = readerOperands(executable, args)
  for (const operand of operands) {
    if (input.isProtectedPath(operand, cwd)) return `${executable} reads a protected path`
  }
  if (readsRecursively(executable, args)) {
    const root = input.requestCwd ?? cwd
    const targets = operands.length > 0 ? operands : ['.']
    for (const target of targets) {
      if (!root || !isWithin(resolve(cwd ?? root, target), root)) {
        return `${executable} reads recursively outside the working directory`
      }
    }
  }
  return null
}

const READER_EXECUTABLES = new Set(['cat', 'head', 'tail', 'wc', 'grep', 'find', 'du', 'ls', 'date'])

const FIND_EXECUTING_OR_WRITING_ACTIONS = new Set([
  '-exec', '-execdir', '-ok', '-okdir', '-delete', '-fprint', '-fprint0', '-fprintf', '-fls',
])

const GIT_ESCALATING_GLOBAL_OPTIONS = ['-c', '--config-env', '--exec-path', '--git-dir', '--work-tree', '-C']

const GIT_PROGRAM_OPTIONS = ['--upload-pack', '--receive-pack', '--exec', '-x', '--extcmd', '--tool', '--tree-filter', '--index-filter', '--msg-filter', '--env-filter', '--commit-filter', '--parent-filter', '--tag-name-filter', '--subdirectory-filter']

const GIT_PROGRAM_SUBCOMMANDS = new Set(['filter-branch', 'difftool', 'mergetool', 'daemon', 'instaweb', 'credential', 'credential-store', 'credential-cache', 'submodule'])

function gitEscalation(args: readonly string[]): string | null {
  let index = 0
  for (; index < args.length; index += 1) {
    const arg = args[index]!
    if (!arg.startsWith('-')) break
    if (GIT_ESCALATING_GLOBAL_OPTIONS.some((option) => arg === option || arg.startsWith(`${option}=`) || (option === '-c' && arg.startsWith('-c') && arg.length > 2) || (option === '-C' && arg.startsWith('-C') && arg.length > 2))) {
      return `git ${arg.split('=')[0]} changes configuration or repository`
    }
  }
  const subcommand = args[index]
  if (!subcommand) return null
  const rest = args.slice(index + 1)
  if (GIT_PROGRAM_SUBCOMMANDS.has(subcommand)) return `git ${subcommand} can run programs`
  if (subcommand === 'config' && !rest.some((arg) => GIT_CONFIG_READ_OPTIONS.has(arg))) {
    return 'git config writes configuration'
  }
  if (subcommand === 'bisect' && rest[0] === 'run') return 'git bisect run executes a command'
  for (const arg of rest) {
    if (arg === '--') break
    if (GIT_PROGRAM_OPTIONS.some((option) => arg === option || arg.startsWith(`${option}=`))) {
      return `git ${subcommand} ${arg.split('=')[0]} runs a program`
    }
    if (/^ext::/i.test(arg) || /^-u./.test(arg) && ['clone', 'fetch', 'pull', 'ls-remote'].includes(subcommand)) {
      return `git ${subcommand} names a transport program`
    }
    if (arg === '-u' && ['clone', 'fetch', 'pull', 'ls-remote', 'archive'].includes(subcommand)) {
      return `git ${subcommand} -u names an upload program`
    }
  }
  return null
}

const GIT_CONFIG_READ_OPTIONS = new Set(['--get', '--get-all', '--get-regexp', '--get-urlmatch', '--list', '-l', 'get', 'list'])

function readerOperands(executable: string, args: readonly string[]): string[] {
  const operands: string[] = []
  const valueOptions = READER_VALUE_OPTIONS[executable] ?? new Set<string>()
  const fileValueOptions = READER_FILE_VALUE_OPTIONS[executable] ?? new Set<string>()
  let grepPatternPending = executable === 'grep'
  let afterDoubleDash = false
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index]!
    if (!afterDoubleDash && arg === '--') { afterDoubleDash = true; continue }
    if (!afterDoubleDash && arg.startsWith('-') && arg !== '-') {
      const [flag, inlineValue] = arg.split('=', 2) as [string, string | undefined]
      if (fileValueOptions.has(flag)) {
        const value = inlineValue ?? args[index + 1]
        if (inlineValue === undefined) index += 1
        if (value) operands.push(value)
        if (executable === 'grep' && (flag === '-f' || flag === '--file')) grepPatternPending = false
        continue
      }
      if (valueOptions.has(flag) && inlineValue === undefined) index += 1
      if (executable === 'grep' && (flag === '-e' || flag === '--regexp')) grepPatternPending = false
      continue
    }
    if (executable === 'find' && !afterDoubleDash && /^[-(!),]/.test(arg)) break
    if (grepPatternPending) { grepPatternPending = false; continue }
    if (executable === 'date') continue
    operands.push(arg)
  }
  return operands
}

const READER_VALUE_OPTIONS: Record<string, Set<string>> = {
  head: new Set(['-n', '-c', '--lines', '--bytes']),
  tail: new Set(['-n', '-c', '--lines', '--bytes', '-s', '--sleep-interval', '--pid']),
  grep: new Set(['-e', '--regexp', '-m', '--max-count', '-A', '-B', '-C', '--include', '--exclude', '--exclude-dir', '--color', '--colour', '-d', '--directories', '-D', '--devices', '--label', '--binary-files']),
  du: new Set(['-d', '--max-depth', '-B', '--block-size', '-t', '--threshold', '--exclude', '--time-style']),
  ls: new Set(['-I', '--ignore', '-w', '--width', '--color', '--time-style', '--sort', '--format', '-T', '--tabsize', '--hide', '--block-size']),
  date: new Set(['-d', '--date', '-I', '--iso-8601', '--rfc-3339', '-r', '--reference']),
}

const READER_FILE_VALUE_OPTIONS: Record<string, Set<string>> = {
  grep: new Set(['-f', '--file']),
  date: new Set(['-f', '--file', '-r', '--reference']),
  du: new Set(['--files0-from', '-X', '--exclude-from']),
  wc: new Set(['--files0-from']),
}

function readsRecursively(executable: string, args: readonly string[]): boolean {
  if (executable === 'find' || executable === 'du') return true
  if (executable === 'grep') {
    return args.some((arg) => arg === '--recursive' || arg === '--dereference-recursive'
      || /^-[a-zA-Z]*[rR][a-zA-Z]*$/.test(arg))
  }
  if (executable === 'ls') return args.some((arg) => arg === '-R' || arg === '--recursive' || /^-[a-zA-Z]*R[a-zA-Z]*$/.test(arg))
  return false
}

function isWithin(target: string, root: string): boolean {
  const rel = relative(resolve(root), resolve(target))
  return rel === '' || (!rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel))
}
