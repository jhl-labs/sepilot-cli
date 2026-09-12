import { accessSync, constants, readdirSync } from 'node:fs'
import { homedir } from 'node:os'
import { delimiter, join } from 'node:path'
import type { DaemonInvocation, DaemonResolveOptions } from './types.js'

interface RuntimePackageResolveConfig {
  envNames: string[]
  packageName: 'daemon' | 'gateway'
  resourceBinaryName: string
  pathBinaryName: string
}

function fileExists(path: string): boolean {
  try {
    accessSync(path, constants.F_OK)
    return true
  } catch {
    return false
  }
}

function executableExists(path: string): boolean {
  try {
    accessSync(path, constants.X_OK)
    return true
  } catch {
    return false
  }
}

function findExecutableInPath(
  name: string,
  env: NodeJS.ProcessEnv,
): string | null {
  const pathValue = env.PATH
  if (!pathValue) return null

  const suffixes =
    process.platform === 'win32'
      ? (env.PATHEXT?.split(';').filter(Boolean) ?? ['.EXE', '.CMD', '.BAT'])
      : ['']

  for (const entry of pathValue.split(delimiter)) {
    if (!entry) continue
    for (const suffix of suffixes) {
      const candidate = join(entry, `${name}${suffix}`)
      if (executableExists(candidate)) {
        return candidate
      }
    }
  }

  return null
}

function invocationFromTarget(
  target: string,
  execPath: string,
): DaemonInvocation {
  return target.endsWith('.js')
    ? { command: execPath, args: [target], kind: 'module' }
    : { command: target, args: [], kind: 'binary' }
}

function scanPnpmPackageEntry(
  baseDir: string,
  packageName: RuntimePackageResolveConfig['packageName'],
): string | null {
  const pnpmDir = join(baseDir, 'node_modules', '.pnpm')
  try {
    const entries = readdirSync(pnpmDir, { withFileTypes: true })
    for (const entry of entries) {
      if (!entry.isDirectory() || !entry.name.startsWith(`@sepilotd+${packageName}@`)) {
        continue
      }

      const candidate = join(
        pnpmDir,
        entry.name,
        'node_modules',
        '@sepilotd',
        packageName,
        'dist',
        'index.js',
      )

      if (fileExists(candidate)) {
        return candidate
      }
    }
  } catch {
    // Ignore missing packaged pnpm layout.
  }

  return null
}

function resolveRuntimePackageInvocation(
  config: RuntimePackageResolveConfig,
  options: DaemonResolveOptions = {},
): DaemonInvocation | null {
  const moduleSearchRoots = options.moduleSearchRoots ?? []
  const resourceSearchRoots = options.resourceSearchRoots ?? []
  const scanPnpm = options.scanPnpmLayout ?? false
  const env = options.env ?? process.env
  const execPath = options.execPath ?? process.execPath
  const home = options.homeDir ?? homedir()

  // 1. Explicit env var — highest priority
  const explicit = config.envNames
    .map((name) => env[name]?.trim())
    .find((value): value is string => Boolean(value))
  if (explicit && (fileExists(explicit) || executableExists(explicit))) {
    return invocationFromTarget(explicit, execPath)
  }

  // 2. Resource roots — packaged binary or module
  for (const root of resourceSearchRoots) {
    const candidates = [
      join(root, config.packageName, 'index.js'),
      join(root, config.packageName, 'dist', 'index.js'),
      join(root, config.resourceBinaryName),
    ]
    for (const candidate of candidates) {
      if (fileExists(candidate) || executableExists(candidate)) {
        return invocationFromTarget(candidate, execPath)
      }
    }
  }

  // 3. Module search roots — node_modules/@sepilotd/daemon/dist/index.js
  for (const root of moduleSearchRoots) {
    const candidate = join(
      root,
      'node_modules',
      '@sepilotd',
      config.packageName,
      'dist',
      'index.js',
    )
    if (fileExists(candidate)) {
      return { command: execPath, args: [candidate], kind: 'module' }
    }
  }

  // 4. pnpm layout scan
  if (scanPnpm) {
    for (const root of [...moduleSearchRoots, ...resourceSearchRoots]) {
      const pnpmEntry = scanPnpmPackageEntry(root, config.packageName)
      if (pnpmEntry) {
        return { command: execPath, args: [pnpmEntry], kind: 'module' }
      }
    }
  }

  // 5. Relative sibling paths from each module root
  for (const root of moduleSearchRoots) {
    const candidates = [
      join(root, '..', config.packageName, 'dist', 'index.js'),
      join(root, '..', '..', config.packageName, 'dist', 'index.js'),
      join(root, '..', '..', '..', config.packageName, 'dist', 'index.js'),
    ]
    for (const candidate of candidates) {
      if (fileExists(candidate)) {
        return { command: execPath, args: [candidate], kind: 'module' }
      }
    }
  }

  // 6. Home directory fallback
  const homeCandidate = join(
    home,
    '.local',
    'share',
    'sepilotd',
    'packages',
    config.packageName,
    'dist',
    'index.js',
  )
  if (fileExists(homeCandidate)) {
    return { command: execPath, args: [homeCandidate], kind: 'module' }
  }

  // 7. $PATH fallback
  const fromPath = findExecutableInPath(config.pathBinaryName, env)
  if (fromPath) {
    return { command: fromPath, args: [], kind: 'binary' }
  }

  return null
}

export function resolveDaemonInvocation(
  options: DaemonResolveOptions = {},
): DaemonInvocation | null {
  return resolveRuntimePackageInvocation(
    {
      envNames: ['SEPILOTD_BIN'],
      packageName: 'daemon',
      resourceBinaryName: 'sepilotd',
      pathBinaryName: 'sepilotd',
    },
    options,
  )
}

export function resolveGatewayInvocation(
  options: DaemonResolveOptions = {},
): DaemonInvocation | null {
  return resolveRuntimePackageInvocation(
    {
      envNames: ['SEPILOT_DESKTOP_GATEWAY_BIN', 'SEPILOTD_GATEWAY_BIN'],
      packageName: 'gateway',
      resourceBinaryName: 'sepilotd-gateway',
      pathBinaryName: 'sepilotd-gateway',
    },
    options,
  )
}
