import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

export interface SpawnLockOptions {
  lockPath: string
  staleMs?: number
  pid?: number
}

export interface AcquiredSpawnLock {
  release(): Promise<void>
}

const DEFAULT_STALE_MS = 30_000

function isProcessAlive(pid: number): boolean {
  if (!Number.isFinite(pid) || pid <= 0) return false
  try {
    process.kill(pid, 0)
    return true
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code
    return code === 'EPERM'
  }
}

async function readLockMeta(lockPath: string): Promise<{ pid: number; ts: number } | null> {
  try {
    const raw = await readFile(`${lockPath}/meta`, 'utf-8')
    const parsed = JSON.parse(raw) as { pid?: unknown; ts?: unknown }
    const pid = typeof parsed.pid === 'number' ? parsed.pid : Number(parsed.pid)
    const ts = typeof parsed.ts === 'number' ? parsed.ts : Number(parsed.ts)
    if (!Number.isFinite(pid) || !Number.isFinite(ts)) return null
    return { pid, ts }
  } catch {
    return null
  }
}

export async function tryAcquireSpawnLock(
  options: SpawnLockOptions,
): Promise<AcquiredSpawnLock | null> {
  const lockPath = options.lockPath
  const staleMs = options.staleMs ?? DEFAULT_STALE_MS
  const pid = options.pid ?? process.pid

  await mkdir(dirname(lockPath), { recursive: true })

  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      await mkdir(lockPath)
      try {
        await writeFile(
          `${lockPath}/meta`,
          JSON.stringify({ pid, ts: Date.now() }),
          'utf-8',
        )
      } catch {
      }
      let released = false
      return {
        async release() {
          if (released) return
          released = true
          try {
            await rm(lockPath, { recursive: true, force: true })
          } catch {
          }
        },
      }
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'EEXIST') throw err
    }

    const meta = await readLockMeta(lockPath)
    const now = Date.now()
    const expired = !meta
      || (now - meta.ts > staleMs)
      || !isProcessAlive(meta.pid)
    if (!expired) return null

    try {
      await rm(lockPath, { recursive: true, force: true })
    } catch {
    }
  }

  return null
}
