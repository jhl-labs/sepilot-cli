import { link, mkdir, open, readFile, rename, rm, stat, unlink, writeFile } from 'node:fs/promises'
import { unlinkSync } from 'node:fs'
import { randomUUID } from 'node:crypto'
import { dirname, join } from 'node:path'

export interface PidAcquireOptions {
  retries?: number
  retryDelayMs?: number
}

type PidFileState = 'missing' | 'live' | 'dead' | 'indeterminate'

const ownedPidPaths = new Set<string>()

interface ReclaimMarkerState {
  state: PidFileState
  raw?: string
  token?: string
}

export interface PidHolder {
  pid: number
  ageMs: number
  alive: boolean
}

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
      process.kill(pid, 0)
      return 'live'
    } catch (error) {
      return (error as NodeJS.ErrnoException)?.code === 'ESRCH'
        ? 'dead'
        : 'indeterminate'
    }
  }
}
