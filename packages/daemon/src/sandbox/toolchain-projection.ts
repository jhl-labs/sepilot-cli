import { constants } from 'node:fs'
import { access, realpath, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { delimiter, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'

const SANDBOX_TOOLCHAIN_ROOT = '/run/sepilotd/toolchains'
const DEFAULT_SANDBOX_PATH_ENTRIES = ['/usr/local/bin', '/usr/bin', '/bin']
const SYSTEM_TOOLCHAIN_ROOTS = ['/usr', '/bin', '/sbin']

export interface SandboxToolchainBind {
  source: string
  target: string
}

export interface SandboxToolchainProjection {
  path: string
  readOnlyBinds: SandboxToolchainBind[]
  env: Record<string, string>
}

export interface SandboxToolchainDiscoveryOptions {
  hostPath?: string
  hostHome?: string
}

let defaultProjectionPromise: Promise<SandboxToolchainProjection> | undefined

interface HomeToolchainDescriptor {
  id: string
  sourceRoot: string
  pathRelativeToRoot: string
  env?: Record<string, string>
  companion?: {
    sourceRoot: string
    targetName: string
    env: Record<string, string>
  }
}

function isWithin(candidate: string, root: string): boolean {
  const child = relative(root, candidate)
  return child === '' || (!child.startsWith(`..${sep}`) && child !== '..' && !isAbsolute(child))
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)]
}

async function existingDirectory(path: string): Promise<string | null> {
  try {
    const canonical = await realpath(path)
    return (await stat(canonical)).isDirectory() ? canonical : null
  } catch {
    return null
  }
}

async function existingExecutable(path: string): Promise<string | null> {
  try {
    await access(path, constants.X_OK)
    const canonical = await realpath(path)
    return (await stat(canonical)).isFile() ? canonical : null
  } catch {
    return null
  }
}

async function existingFile(path: string): Promise<string | null> {
  try {
    const canonical = await realpath(path)
    return (await stat(canonical)).isFile() ? canonical : null
  } catch {
    return null
  }
}

/**
 * Select PATH entries already covered by immutable system mounts. Canonical
 * containment prevents a symlink under /usr or /opt from projecting an
 * arbitrary host directory into the sandbox.
 */
export async function discoverSystemToolchainPathEntries(hostPath: string): Promise<string[]> {
  const entries: string[] = []
  for (const rawEntry of hostPath.split(delimiter)) {
    if (!rawEntry || !isAbsolute(rawEntry)) continue
    const entry = resolve(rawEntry)
    const canonical = await existingDirectory(entry)
    if (!canonical) continue
    if (!SYSTEM_TOOLCHAIN_ROOTS.some((root) => isWithin(canonical, root))) continue
    entries.push(entry)
  }
  return unique([...entries, ...DEFAULT_SANDBOX_PATH_ENTRIES])
}

function describeHomeToolchain(entry: string, home: string): HomeToolchainDescriptor | null {
  if (!isWithin(entry, home)) return null
  const homeRelative = relative(home, entry).split(sep).join('/')

  const nvm = /^\.nvm\/versions\/node\/([^/]+)\/bin$/.exec(homeRelative)
  if (nvm) {
    return {
      id: 'node',
      sourceRoot: join(home, '.nvm', 'versions', 'node', nvm[1]!),
      pathRelativeToRoot: 'bin',
    }
  }

  const conda = /^(miniforge3|miniconda3|anaconda3)\/bin$/.exec(homeRelative)
  if (conda) {
    return {
      id: 'python',
      sourceRoot: join(home, conda[1]!),
      pathRelativeToRoot: 'bin',
    }
  }

  const sdkman = /^\.sdkman\/candidates\/(java|gradle|maven|kotlin)\/([^/]+)\/bin$/.exec(
    homeRelative,
  )
  if (sdkman) {
    const id = `sdkman-${sdkman[1]!}`
    const target = `${SANDBOX_TOOLCHAIN_ROOT}/${id}`
    return {
      id,
      sourceRoot: join(home, '.sdkman', 'candidates', sdkman[1]!, sdkman[2]!),
      pathRelativeToRoot: 'bin',
      ...(sdkman[1] === 'java' ? { env: { JAVA_HOME: target } } : {}),
    }
  }

  const patterns: Array<{
    pattern: RegExp
    id: string
    source: string
    relativePath: (match: RegExpExecArray) => string
    env?: (target: string) => Record<string, string>
  }> = [
    {
      pattern: /^\.bun\/bin$/,
      id: 'bun',
      source: '.bun',
      relativePath: () => 'bin',
      env: (target) => ({ BUN_INSTALL: target }),
    },
    {
      pattern: /^\.local\/share\/pnpm$/,
      id: 'pnpm',
      source: '.local/share/pnpm',
      relativePath: () => '',
    },
    {
      pattern: /^\.cargo\/bin$/,
      id: 'cargo-bin',
      source: '.cargo/bin',
      relativePath: () => '',
    },
    {
      pattern: /^\.pyenv\/(bin|shims)$/,
      id: 'pyenv',
      source: '.pyenv',
      relativePath: (match) => match[1]!,
      env: (target) => ({ PYENV_ROOT: target }),
    },
    {
      pattern: /^\.volta\/bin$/,
      id: 'volta',
      source: '.volta',
      relativePath: () => 'bin',
      env: (target) => ({ VOLTA_HOME: target }),
    },
    {
      pattern: /^\.asdf\/(bin|shims)$/,
      id: 'asdf',
      source: '.asdf',
      relativePath: (match) => match[1]!,
      env: (target) => ({ ASDF_DATA_DIR: target }),
    },
    {
      pattern: /^\.dotnet(?:\/(tools))?$/,
      id: 'dotnet',
      source: '.dotnet',
      relativePath: (match) => match[1] ?? '',
      env: (target) => ({ DOTNET_ROOT: target }),
    },
    {
      pattern: /^\.deno\/bin$/,
      id: 'deno',
      source: '.deno',
      relativePath: () => 'bin',
      env: (target) => ({ DENO_INSTALL_ROOT: target }),
    },
    {
      pattern: /^flutter\/bin(?:\/cache\/dart-sdk\/bin)?$/,
      id: 'flutter',
      source: 'flutter',
      relativePath: () => homeRelative.slice('flutter/'.length),
    },
    {
      pattern: /^Android\/Sdk\/(.+)$/,
      id: 'android-sdk',
      source: 'Android/Sdk',
      relativePath: (match) => match[1]!,
      env: (target) => ({ ANDROID_HOME: target, ANDROID_SDK_ROOT: target }),
    },
  ]

  for (const candidate of patterns) {
    const match = candidate.pattern.exec(homeRelative)
    if (!match) continue
    const target = `${SANDBOX_TOOLCHAIN_ROOT}/${candidate.id}`
    const descriptor: HomeToolchainDescriptor = {
      id: candidate.id,
      sourceRoot: join(home, ...candidate.source.split('/')),
      pathRelativeToRoot: candidate.relativePath(match),
      ...(candidate.env ? { env: candidate.env(target) } : {}),
    }
    if (candidate.id === 'cargo-bin') {
      descriptor.companion = {
        sourceRoot: join(home, '.rustup'),
        targetName: 'rustup',
        env: {
          RUSTUP_HOME: `${SANDBOX_TOOLCHAIN_ROOT}/rustup`,
          CARGO_HOME: '/tmp/sepilot-cargo',
        },
      }
    }
    return descriptor
  }
  return null
}

function describeOptToolchain(entry: string): HomeToolchainDescriptor | null {
  const normalized = entry.split(sep).join('/')
  const candidates: Array<{
    pattern: RegExp
    id: string
    sourceRoot: (match: RegExpExecArray) => string
    pathRelativeToRoot: (match: RegExpExecArray) => string
    env?: (target: string) => Record<string, string>
  }> = [
    {
      pattern: /^\/opt\/gradle\/([^/]+)\/bin$/,
      id: 'gradle',
      sourceRoot: (match) => `/opt/gradle/${match[1]!}`,
      pathRelativeToRoot: () => 'bin',
    },
    {
      pattern: /^\/opt\/flutter\/bin(?:\/cache\/dart-sdk\/bin)?$/,
      id: 'flutter-opt',
      sourceRoot: () => '/opt/flutter',
      pathRelativeToRoot: () => normalized.slice('/opt/flutter/'.length),
    },
    {
      pattern: /^\/opt\/android-sdk\/(.+)$/,
      id: 'android-sdk-opt',
      sourceRoot: () => '/opt/android-sdk',
      pathRelativeToRoot: (match) => match[1]!,
      env: (target) => ({ ANDROID_HOME: target, ANDROID_SDK_ROOT: target }),
    },
  ]
  for (const candidate of candidates) {
    const match = candidate.pattern.exec(normalized)
    if (!match) continue
    const target = `${SANDBOX_TOOLCHAIN_ROOT}/${candidate.id}`
    return {
      id: candidate.id,
      sourceRoot: candidate.sourceRoot(match),
      pathRelativeToRoot: candidate.pathRelativeToRoot(match),
      ...(candidate.env ? { env: candidate.env(target) } : {}),
    }
  }
  return null
}

/**
 * Build a conservative, read-only projection of host development toolchains.
 *
 * System PATH entries are accepted only below immutable /usr, /bin, or /sbin
 * mounts. Known /opt and home-managed runtimes are remounted individually
 * under /run/sepilotd/toolchains; the rest of /opt, the host home path, SSH
 * material, npmrc, cloud credentials, and arbitrary PATH directories remain
 * invisible.
 */
export async function discoverSandboxToolchainProjection(
  options: SandboxToolchainDiscoveryOptions = {},
): Promise<SandboxToolchainProjection> {
  // A systemd user service commonly starts with a minimal PATH even when the
  // interactive shell that installed it has Go, NVM, Cargo, Conda, etc. The
  // installer records only the candidate PATH; all entries still pass the
  // conservative system/known-home-toolchain checks below before any bind or
  // sandbox PATH entry is created.
  const hostPath = options.hostPath
    ?? process.env.SEPILOTD_HOST_TOOLCHAIN_PATH
    ?? process.env.PATH
    ?? ''
  const home = resolve(options.hostHome ?? homedir())
  const systemPathEntries = await discoverSystemToolchainPathEntries(hostPath)
  const projectedPathEntries: string[] = []
  const readOnlyBinds: SandboxToolchainBind[] = []
  const env: Record<string, string> = {}
  const boundSources = new Set<string>()
  const projectedIds = new Map<string, string>()

  for (const rawEntry of hostPath.split(delimiter)) {
    if (!rawEntry || !isAbsolute(rawEntry)) continue
    const entry = resolve(rawEntry)
    const homeDescriptor = describeHomeToolchain(entry, home)
    const descriptor = homeDescriptor ?? describeOptToolchain(entry)
    if (!descriptor) continue
    const canonicalSource = await existingDirectory(descriptor.sourceRoot)
    const canonicalHome = await existingDirectory(home)
    const trustedRoot = homeDescriptor ? canonicalHome : '/opt'
    if (!canonicalSource || !trustedRoot || !isWithin(canonicalSource, trustedRoot)) continue

    const target = `${SANDBOX_TOOLCHAIN_ROOT}/${descriptor.id}`
    const existingProjection = projectedIds.get(descriptor.id)
    if (existingProjection && existingProjection !== canonicalSource) continue
    if (!boundSources.has(canonicalSource)) {
      readOnlyBinds.push({ source: canonicalSource, target })
      boundSources.add(canonicalSource)
    }
    projectedIds.set(descriptor.id, canonicalSource)
    projectedPathEntries.push(
      descriptor.pathRelativeToRoot ? join(target, descriptor.pathRelativeToRoot) : target,
    )
    Object.assign(env, descriptor.env ?? {})

    // Home-managed Python distributions commonly compile OpenSSL with an
    // absolute certificate path below their original installation prefix.
    // The runtime is intentionally remounted at a neutral sandbox path, so
    // that compiled-in location no longer exists even though the distribution
    // and its public CA bundle are both present. Project the bundle location
    // through standard TLS environment variables when it exists. This keeps
    // HTTPS package managers, crawlers, and test clients functional without
    // mounting the host home or weakening certificate verification.
    if (descriptor.id === 'python') {
      const certificateFile = await existingFile(join(canonicalSource, 'ssl', 'cert.pem'))
      const certificateDirectory = await existingDirectory(join(canonicalSource, 'ssl', 'certs'))
      if (certificateFile) {
        const sandboxCertificateFile = join(target, 'ssl', 'cert.pem')
        env.SSL_CERT_FILE = sandboxCertificateFile
        env.REQUESTS_CA_BUNDLE = sandboxCertificateFile
        env.CURL_CA_BUNDLE = sandboxCertificateFile
        env.PIP_CERT = sandboxCertificateFile
      }
      if (certificateDirectory) {
        env.SSL_CERT_DIR = join(target, 'ssl', 'certs')
      }
    }

    if (descriptor.companion) {
      const companionSource = await existingDirectory(descriptor.companion.sourceRoot)
      if (companionSource && canonicalHome && isWithin(companionSource, canonicalHome)) {
        const companionTarget = `${SANDBOX_TOOLCHAIN_ROOT}/${descriptor.companion.targetName}`
        if (!boundSources.has(companionSource)) {
          readOnlyBinds.push({ source: companionSource, target: companionTarget })
          boundSources.add(companionSource)
        }
        Object.assign(env, descriptor.companion.env)
      }
    }
  }

  // uv is commonly installed as one self-contained executable in ~/.local/bin.
  // Bind only that file, never the surrounding arbitrary user-script directory.
  const localBin = join(home, '.local', 'bin')
  const hostPathEntries = hostPath.split(delimiter).map((entry) => resolve(entry))
  if (hostPathEntries.includes(localBin)) {
    const uv = await existingExecutable(join(localBin, 'uv'))
    const canonicalHome = await existingDirectory(home)
    if (uv && canonicalHome && isWithin(uv, canonicalHome)) {
      const target = `${SANDBOX_TOOLCHAIN_ROOT}/bin/uv`
      readOnlyBinds.push({ source: uv, target })
      projectedPathEntries.push(dirname(target))
    }
  }

  // Debian/Ubuntu JDKs keep java.security below /etc and link to it from the
  // otherwise read-only /usr/lib/jvm tree. Project only that versioned Java
  // configuration directory so Gradle/Maven work without exposing all /etc.
  const javaCandidates = unique([
    ...hostPathEntries.map((entry) => join(entry, 'java')),
    '/usr/bin/java',
  ])
  for (const candidate of javaCandidates) {
    const java = await existingExecutable(candidate)
    if (!java) continue
    const javaHome = dirname(dirname(java))
    if (!SYSTEM_TOOLCHAIN_ROOTS.some((root) => isWithin(javaHome, root))) continue
    const securityFile = await existingFile(join(javaHome, 'conf', 'security', 'java.security'))
    if (securityFile) {
      const configRoot = dirname(dirname(securityFile))
      if (isWithin(configRoot, '/etc') && /^java-[^/]+/.test(relative('/etc', configRoot))) {
        readOnlyBinds.push({ source: configRoot, target: configRoot })
      }
    }
    env.JAVA_HOME ??= javaHome
    break
  }

  return {
    path: unique([...projectedPathEntries, ...systemPathEntries]).join(':'),
    readOnlyBinds,
    env,
  }
}

/** One immutable inventory per daemon process; all runner profiles reuse it. */
export function defaultSandboxToolchainProjection(): Promise<SandboxToolchainProjection> {
  defaultProjectionPromise ??= discoverSandboxToolchainProjection()
  return defaultProjectionPromise
}
