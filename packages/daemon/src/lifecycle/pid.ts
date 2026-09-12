import { link, mkdir, open, readFile, rename, rm, stat, unlink, writeFile } from 'node:fs/promises'
import { unlinkSync } from 'node:fs'
import { randomUUID } from 'node:crypto'
import { dirname, join } from 'node:path'

export interface PidAcquireOptions {
  /**
   * If the PID file is held by a live process, retry this many extra times
   * before giving up. Lets a freshly-spawned instance wait out a predecessor
   * that is still mid-shutdown (e.g. `tsup --watch` restarts in dev). Default 0.
   */
  retries?: number
  /** Delay between retry attempts, in milliseconds. Default 500. */
  retryDelayMs?: number
}

type PidFileState = 'missing' | 'live' | 'dead' | 'indeterminate'

// Multiple PidManager instances can exist inside one daemon process (tests,
// embedded startup probes). A numeric PID file alone cannot distinguish that
// legitimate in-process owner from a stale file whose PID was reused after a
// container/host restart. Track ownership in-process so an unowned file that
// points back to the current PID can be safely reclaimed, while a second
// manager in the same process still fails closed.
const ownedPidPaths = new Set<string>()

interface ReclaimMarkerState {
  state: PidFileState
  raw?: string
  token?: string
}

export interface PidHolder {
  pid: number
  /** How long the pid file has existed, in milliseconds. */
  ageMs: number
  alive: boolean
}

/**
 * Build the error a failed single-instance acquisition reports. The bare
 * "Another sepilotd instance is already running" left an operator with no way
 * to tell a genuinely running daemon from a wedged one, so name the holder,
 * its age, and the concrete next step.
 */
export function formatPidConflictMessage(
  pidPath: string,
  holder: PidHolder | null,
): string {
  if (!holder) {
    return (
      'Another sepilotd instance is already running, or its lock file is in an '
      + `indeterminate state. Lock file: ${pidPath}. `
      + 'Inspect it and remove it only if no sepilotd process is alive.'
    )
  }
  const ageSeconds = Math.max(0, Math.round(holder.ageMs / 1000))
  const liveness = holder.alive ? 'still alive' : 'no longer alive'
  return (
    `Another sepilotd instance is already running: pid ${holder.pid} (${liveness}), `
    + `lock held for ${ageSeconds}s via ${pidPath}. `
    + `Stop it with \`kill ${holder.pid}\` (or POST /api/v1/system/shutdown), `
    + 'wait for it to exit, then start again. If that pid is gone, remove the '
    + 'lock file.'
  )
}

export class PidManager {
  private pidPath: string
  private reclaimPath: string
  private acquired = false

  constructor(pidPath: string) {
    this.pidPath = pidPath
    this.reclaimPath = `${pidPath}.reclaim`
  }

  async acquire(options: PidAcquireOptions = {}): Promise<boolean> {
    const retries = Math.max(0, options.retries ?? 0)
    const retryDelayMs = Math.max(0, options.retryDelayMs ?? 500)

    await mkdir(dirname(this.pidPath), { recursive: true })

    let reclaimMarker: string | undefined
    try {
      for (let attempt = 0; ; attempt++) {
        // Atomic create-exclusive: open(2) with O_CREAT|O_EXCL. If two daemons
        // race to boot, exactly one open succeeds; the other gets EEXIST. This
        // closes the check->write TOCTOU window that let both pass and briefly
        // share the SQLite files.
        try {
          const handle = await open(this.pidPath, 'wx')
          ownedPidPaths.add(this.pidPath)
          try {
            await handle.writeFile(String(process.pid))
          } catch (error) {
            await handle.close().catch(() => undefined)
            await unlink(this.pidPath).catch(() => undefined)
            ownedPidPaths.delete(this.pidPath)
            throw error
          }
          await handle.close()
          this.acquired = true
          return true
        } catch (error) {
          if ((error as NodeJS.ErrnoException)?.code !== 'EEXIST') {
            throw error
          }
        }

        // The file already exists. Only a strict PID whose process is confirmed
        // absent is safe to reclaim. Empty/partial/malformed files can be a
        // concurrent winner that has created the inode but not written yet, so
        // those states fail closed instead of being unlinked.
        const state = await this.inspectPidFile()
        if (state === 'missing') {
          continue
        }
        if (state !== 'dead') {
          if (attempt < retries) {
            await new Promise((resolve) => setTimeout(resolve, retryDelayMs))
            continue
          }
          return false
        }

        // Serialize stale-file reclamation across processes. The exclusive
        // marker stays held until one contender has installed the replacement
        // PID, and the winner revalidates the PID state inside the critical
        // section before unlinking anything.
        reclaimMarker = await this.tryAcquireReclaimLock()
        if (!reclaimMarker) {
          if (attempt < retries) {
            await new Promise((resolve) => setTimeout(resolve, retryDelayMs))
            continue
          }
          return false
        }

        const confirmedState = await this.inspectPidFile()
        if (confirmedState === 'missing') {
          continue
        }
        if (confirmedState !== 'dead') {
          await this.releaseReclaimLock(reclaimMarker)
          reclaimMarker = undefined
          if (attempt < retries) {
            await new Promise((resolve) => setTimeout(resolve, retryDelayMs))
            continue
          }
          return false
        }

        try {
          await unlink(this.pidPath)
        } catch (error) {
          if ((error as NodeJS.ErrnoException)?.code !== 'ENOENT') throw error
        }
      }
    } finally {
      if (reclaimMarker) {
        await this.releaseReclaimLock(reclaimMarker)
      }
    }
  }

  /**
   * Describe whoever currently holds the pid file, for diagnostics. Returns
   * null when the file is absent or unparseable.
   */
  async readHolder(): Promise<PidHolder | null> {
    let raw: string
    try {
      raw = (await readFile(this.pidPath, 'utf-8')).trim()
    } catch {
      return null
    }
    if (!/^[1-9]\d*$/.test(raw)) return null
    const pid = Number(raw)
    if (!Number.isSafeInteger(pid)) return null

    let ageMs = 0
    try {
      ageMs = Date.now() - (await stat(this.pidPath)).mtimeMs
    } catch {
      ageMs = 0
    }

    let alive = false
    try {
      process.kill(pid, 0)
      alive = true
    } catch (error) {
      alive = (error as NodeJS.ErrnoException)?.code !== 'ESRCH'
    }
    return { pid, ageMs, alive }
  }

  async release(): Promise<void> {
    if (!this.acquired) return
    try {
      await unlink(this.pidPath)
      this.acquired = false
      ownedPidPaths.delete(this.pidPath)
    } catch (error) {
      if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') {
        this.acquired = false
        ownedPidPaths.delete(this.pidPath)
        return
      }
      throw error
    }
  }

  /**
   * Synchronous PID removal for the `process.on('exit')` last-resort hook.
   * The exit event runs with a stopped event loop, so an async unlink there
   * never completes and the stale PID file blocks the next daemon start.
   */
  releaseSync(): void {
    if (!this.acquired) return
    try {
      unlinkSync(this.pidPath)
      this.acquired = false
      ownedPidPaths.delete(this.pidPath)
    } catch (error) {
      if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') {
        this.acquired = false
        ownedPidPaths.delete(this.pidPath)
      }
    }
  }

  private async tryAcquireReclaimLock(): Promise<string | undefined> {
    // Publish a complete marker with an exclusive hard link. Unlike opening
    // the final path and then writing it, this never exposes an empty/partial
    // marker if the process is interrupted between those two operations.
    for (let attempt = 0; attempt < 3; attempt++) {
      const token = randomUUID()
      const raw = JSON.stringify({ version: 1, pid: process.pid, token })
      const tempPath = `${this.reclaimPath}.${token}.tmp`
      try {
        const handle = await open(tempPath, 'wx')
        try {
          await handle.writeFile(raw)
        } finally {
          await handle.close()
        }
        await link(tempPath, this.reclaimPath)
        return raw
      } catch (error) {
        if ((error as NodeJS.ErrnoException)?.code !== 'EEXIST') throw error

        const marker = await this.inspectReclaimMarker()
        if (marker.state !== 'dead' || !marker.raw || !marker.token) {
          return undefined
        }
        if (!(await this.quarantineDeadReclaimMarker(marker))) {
          return undefined
        }
      } finally {
        await unlink(tempPath).catch(() => undefined)
      }
    }
    return undefined
  }

  private async releaseReclaimLock(expectedRaw: string): Promise<void> {
    let current: string
    try {
      current = await readFile(this.reclaimPath, 'utf-8')
    } catch (error) {
      if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') return
      throw error
    }
    if (current !== expectedRaw) return
    await unlink(this.reclaimPath)
  }

  private async inspectReclaimMarker(): Promise<ReclaimMarkerState> {
    let raw: string
    try {
      raw = await readFile(this.reclaimPath, 'utf-8')
    } catch (error) {
      return {
        state: (error as NodeJS.ErrnoException)?.code === 'ENOENT'
          ? 'missing'
          : 'indeterminate',
      }
    }

    let parsed: unknown
    try {
      parsed = JSON.parse(raw)
    } catch {
      return { state: 'indeterminate' }
    }
    if (
      !parsed
      || typeof parsed !== 'object'
      || (parsed as { version?: unknown }).version !== 1
      || !Number.isSafeInteger((parsed as { pid?: unknown }).pid)
      || Number((parsed as { pid?: unknown }).pid) <= 0
      || typeof (parsed as { token?: unknown }).token !== 'string'
      || !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
        .test((parsed as { token: string }).token)
    ) {
      return { state: 'indeterminate' }
    }

    const pid = Number((parsed as { pid: number }).pid)
    const token = (parsed as { token: string }).token
    try {
      process.kill(pid, 0)
      return { state: 'live', raw, token }
    } catch (error) {
      return {
        state: (error as NodeJS.ErrnoException)?.code === 'ESRCH'
          ? 'dead'
          : 'indeterminate',
        raw,
        token,
      }
    }
  }

  private async quarantineDeadReclaimMarker(marker: ReclaimMarkerState): Promise<boolean> {
    const { raw, token } = marker
    if (!raw || !token) return false

    // A prepared, non-empty directory is renamed to a deterministic token
    // path. Directory replacement cannot overwrite this tombstone, so exactly
    // one contender claims cleanup and late contenders cannot remove a newer
    // marker through an ABA race.
    const quarantinePath = `${this.reclaimPath}.quarantine.${token}`
    const preparedPath = `${quarantinePath}.${randomUUID()}.tmp`
    const recordPath = join(quarantinePath, 'record.json')
    await mkdir(preparedPath)
    try {
      await writeFile(join(preparedPath, 'record.json'), raw, { flag: 'wx' })
      try {
        await rename(preparedPath, quarantinePath)
      } catch (error) {
        const code = (error as NodeJS.ErrnoException)?.code
        if (!['EEXIST', 'ENOTEMPTY', 'EPERM'].includes(code ?? '')) throw error
      }
    } finally {
      await rm(preparedPath, { recursive: true, force: true })
    }

    let quarantineRecord: string
    try {
      quarantineRecord = await readFile(recordPath, 'utf-8')
    } catch {
      return false
    }
    if (quarantineRecord !== raw) return false

    let current: string
    try {
      current = await readFile(this.reclaimPath, 'utf-8')
    } catch (error) {
      return (error as NodeJS.ErrnoException)?.code === 'ENOENT'
    }
    if (current !== raw) return true

    try {
      await rename(this.reclaimPath, join(quarantinePath, 'marker.json'))
      return true
    } catch (error) {
      return (error as NodeJS.ErrnoException)?.code === 'ENOENT'
    }
  }

  private async inspectPidFile(): Promise<PidFileState> {
    let raw: string
    try {
      raw = (await readFile(this.pidPath, 'utf-8')).trim()
    } catch (error) {
      return (error as NodeJS.ErrnoException)?.code === 'ENOENT'
        ? 'missing'
        : 'indeterminate'
    }

    if (!/^[1-9]\d*$/.test(raw)) return 'indeterminate'
    const pid = Number(raw)
    if (!Number.isSafeInteger(pid)) return 'indeterminate'

    if (pid === process.pid) {
      return ownedPidPaths.has(this.pidPath) ? 'live' : 'dead'
    }

    try {
      process.kill(pid, 0) // Signal 0 = check if process exists
      return 'live'
    } catch (error) {
      return (error as NodeJS.ErrnoException)?.code === 'ESRCH'
        ? 'dead'
        : 'indeterminate'
    }
  }
}
