import { createHash } from 'node:crypto'
import {
  chmodSync,
  copyFileSync,
  createWriteStream,
  existsSync,
  readFileSync,
  renameSync,
  rmSync,
} from 'node:fs'
import { dirname, join } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import type { ReadableStream as WebReadableStream } from 'node:stream/web'
import chalk from 'chalk'
import { getStandaloneDaemon } from '../client/standalone-daemon.js'
import { loadCliVersion } from '../entrypoint.js'

const DEFAULT_REPO = 'jhl-labs/sepilot-cli'
const USER_AGENT = 'sepilot-cli-upgrade'

interface UpgradeOptions {
  check?: boolean
  force?: boolean
}

interface GitHubAsset {
  name: string
  browser_download_url: string
  size: number
}

interface GitHubRelease {
  tag_name: string
  html_url: string
  assets: GitHubAsset[]
}

/** Resolve `owner/repo` for GitHub Releases (env override → package.json → default). */
function resolveReleaseRepo(): string {
  const fromEnv = process.env.SEPILOT_UPDATE_REPO?.trim()
  if (fromEnv && /^[^/\s]+\/[^/\s]+$/.test(fromEnv)) return fromEnv

  // Try the CLI's package.json, then the monorepo root, for a `repository` field.
  for (const url of [
    new URL('../../package.json', import.meta.url),
    new URL('../../../../package.json', import.meta.url),
  ]) {
    try {
      const pkg = JSON.parse(readFileSync(url, 'utf8')) as {
        repository?: string | { url?: string }
      }
      const repoStr =
        typeof pkg.repository === 'string' ? pkg.repository : pkg.repository?.url
      if (repoStr) {
        const m = repoStr.match(/github\.com[/:]([^/\s]+\/[^/\s.]+)/)
        if (m) return m[1]
      }
    } catch {
      // ignore — fall through to default
    }
  }
  return DEFAULT_REPO
}

/** Naive numeric `major.minor.patch` compare. Ignores pre-release/build metadata. */
export function compareSemver(a: string, b: string): number {
  const norm = (v: string) =>
    v
      .trim()
      .replace(/^v/, '')
      .split(/[-+]/, 1)[0]
      .split('.')
      .map((n) => Number.parseInt(n, 10) || 0)
  const pa = norm(a)
  const pb = norm(b)
  for (let i = 0; i < Math.max(pa.length, pb.length, 3); i += 1) {
    const da = pa[i] ?? 0
    const db = pb[i] ?? 0
    if (da !== db) return da < db ? -1 : 1
  }
  return 0
}

function platformOs(): 'linux' | 'windows' | 'darwin' | null {
  if (process.platform === 'win32') return 'windows'
  if (process.platform === 'darwin') return 'darwin'
  if (process.platform === 'linux') return 'linux'
  return null
}

function platformArch(): 'x64' | 'arm64' | null {
  if (process.arch === 'x64') return 'x64'
  if (process.arch === 'arm64') return 'arm64'
  return null
}

async function fetchLatestRelease(repo: string): Promise<GitHubRelease> {
  const res = await fetch(`https://api.github.com/repos/${repo}/releases/latest`, {
    headers: { Accept: 'application/vnd.github+json', 'User-Agent': USER_AGENT },
  })
  if (!res.ok) {
    if (res.status === 404) {
      throw new Error(
        `No releases found for ${repo} (HTTP 404). See https://github.com/${repo}/releases`,
      )
    }
    throw new Error(`GitHub API request failed: HTTP ${res.status} ${res.statusText}`)
  }
  return (await res.json()) as GitHubRelease
}

async function downloadTo(url: string, dest: string): Promise<void> {
  const res = await fetch(url, { headers: { 'User-Agent': USER_AGENT } })
  if (!res.ok || !res.body) {
    throw new Error(`Download failed: HTTP ${res.status} ${res.statusText}`)
  }
  // `res.body` is a web ReadableStream; adapt it to a Node stream and pipe to disk.
  await pipeline(
    Readable.fromWeb(res.body as unknown as WebReadableStream<Uint8Array>),
    createWriteStream(dest),
  )
}

async function downloadText(url: string): Promise<string> {
  const res = await fetch(url, { headers: { 'User-Agent': USER_AGENT } })
  if (!res.ok) throw new Error(`Download failed: HTTP ${res.status} ${res.statusText}`)
  return (await res.text()).trim()
}

function sha256OfFile(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex')
}

/** Parse a `.sha256` payload of form `<hex>  <filename>` or just `<hex>`. */
function parseSha256Payload(text: string): string {
  return (text.trim().split(/\s+/, 1)[0] || '').toLowerCase()
}

function bestEffortRm(path: string): void {
  try {
    rmSync(path, { force: true })
  } catch {
    // ignore
  }
}

export async function upgradeCommand(options: UpgradeOptions = {}): Promise<void> {
  const selfBinaryPath = getStandaloneDaemon()?.selfBinaryPath
  if (!selfBinaryPath) {
    console.log(
      'sepilot upgrade only applies to the standalone single-file binary. You installed sepilot via a package manager — update it the same way (e.g. npm i -g @sepilotd/cli, or your install script).',
    )
    return
  }

  const repo = resolveReleaseRepo()
  const releasesUrl = `https://github.com/${repo}/releases`
  const currentVersion = loadCliVersion()

  const os = platformOs()
  const arch = platformArch()
  if (!os || !arch) {
    console.error(
      chalk.red(
        `Unsupported platform ${process.platform}-${process.arch}; download manually from ${releasesUrl}`,
      ),
    )
    process.exit(1)
  }

  let release: GitHubRelease
  try {
    release = await fetchLatestRelease(repo)
  } catch (err) {
    console.error(chalk.red(err instanceof Error ? err.message : String(err)))
    process.exit(1)
  }

  const latestVersion = release.tag_name.replace(/^v/, '')
  // `0.0.0` is the "version unknown" fallback (a known wart in bundled binaries);
  // treat it as older than any real tag so `upgrade` still works.
  const cmp = currentVersion === '0.0.0' ? -1 : compareSemver(currentVersion, latestVersion)
  const updateAvailable = cmp < 0

  if (options.check) {
    if (updateAvailable) {
      console.log(chalk.yellow(`update available: ${currentVersion} → ${latestVersion}`))
      console.log(chalk.gray(`  ${release.html_url}`))
    } else {
      console.log(`sepilot is up to date (${currentVersion}).`)
    }
    return
  }

  if (!updateAvailable && !options.force) {
    console.log(`sepilot is up to date (${currentVersion}).`)
    return
  }

  // Pick the platform asset + its checksum.
  const ext = os === 'windows' ? '.exe' : ''
  const assetName = `sepilot-${os}-${arch}${ext}`
  const shaName = `${assetName}.sha256`
  const binAsset = release.assets.find((a) => a.name === assetName)
  const shaAsset = release.assets.find((a) => a.name === shaName)
  if (!binAsset || !shaAsset) {
    console.error(
      chalk.red(
        `Release ${release.tag_name} is missing ${binAsset ? shaName : assetName}; download manually from ${releasesUrl}`,
      ),
    )
    process.exit(1)
  }

  const dir = dirname(selfBinaryPath)
  const tmpPath = join(dir, `.sepilot-upgrade-${process.pid}.tmp`)

  const sizeMb = (binAsset.size / (1024 * 1024)).toFixed(1)
  console.log(`Downloading sepilot ${latestVersion} (${sizeMb} MB)...`)
  let expectedSha = ''
  try {
    await downloadTo(binAsset.browser_download_url, tmpPath)
    expectedSha = parseSha256Payload(await downloadText(shaAsset.browser_download_url))
  } catch (err) {
    bestEffortRm(tmpPath)
    console.error(chalk.red(err instanceof Error ? err.message : String(err)))
    process.exit(1)
  }

  const actualSha = sha256OfFile(tmpPath)
  if (!expectedSha || actualSha !== expectedSha) {
    bestEffortRm(tmpPath)
    console.error(chalk.red('checksum mismatch — aborting'))
    process.exit(1)
  }
  if (process.platform !== 'win32') chmodSync(tmpPath, 0o755)

  // Swap in the new binary.
  if (process.platform === 'win32') {
    const oldPath = `${selfBinaryPath}.old`
    bestEffortRm(oldPath)
    try {
      renameSync(selfBinaryPath, oldPath)
    } catch (err) {
      // Can't move the running .exe (AV / locked). Leave the new one as `.new`.
      const newPath = `${selfBinaryPath}.new`
      try {
        if (existsSync(newPath)) bestEffortRm(newPath)
        renameSync(tmpPath, newPath)
      } catch {
        bestEffortRm(tmpPath)
        console.error(chalk.red(err instanceof Error ? err.message : String(err)))
        process.exit(1)
      }
      console.error(
        chalk.yellow(
          `Downloaded the update to ${newPath}. Close all sepilot processes and run: move /Y "${newPath}" "${selfBinaryPath}"  (or just re-run the install.ps1).`,
        ),
      )
      process.exit(1)
    }
    try {
      renameSync(tmpPath, selfBinaryPath)
    } catch (err) {
      // Roll back the rename of the original, then bail.
      try {
        renameSync(oldPath, selfBinaryPath)
      } catch {
        // ignore — leave .old in place for manual recovery
      }
      bestEffortRm(tmpPath)
      console.error(chalk.red(err instanceof Error ? err.message : String(err)))
      process.exit(1)
    }
    bestEffortRm(oldPath) // best-effort; may still be in use
  } else {
    try {
      renameSync(tmpPath, selfBinaryPath)
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code
      if (code === 'EXDEV') {
        // Cross-device (shouldn't happen since temp is next to target). Copy + unlink.
        try {
          copyFileSync(tmpPath, selfBinaryPath)
          chmodSync(selfBinaryPath, 0o755)
          bestEffortRm(tmpPath)
        } catch (copyErr) {
          bestEffortRm(tmpPath)
          console.error(chalk.red(copyErr instanceof Error ? copyErr.message : String(copyErr)))
          process.exit(1)
        }
      } else if (code === 'EACCES' || code === 'EPERM') {
        bestEffortRm(tmpPath)
        console.error(
          chalk.red(
            `Cannot replace ${selfBinaryPath} (permission denied). Re-run with sudo, or re-run your install script.`,
          ),
        )
        process.exit(1)
      } else {
        bestEffortRm(tmpPath)
        console.error(chalk.red(err instanceof Error ? err.message : String(err)))
        process.exit(1)
      }
    }
  }

  console.log(`Upgraded sepilot to ${latestVersion}. Restart sepilot to use the new version.`)
}
