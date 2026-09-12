import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

export interface SpawnLockOptions {
  /** Absolute path to the lock directory (mkdir-exclusive). */
  lockPath: string
  /** Stale lock window. If the lock file is older than this and the recorded PID isn't alive, the lock is reclaimed. */
  staleMs?: number
  /** Owning process pid for stale detection. Defaults to process.pid. */
  pid?: number
}

export interface AcquiredSpawnLock {
  /** Release the lock. Idempotent and safe even if the directory was already removed. */
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

/**
 * Atomic mkdir-based lock used to serialise concurrent daemon spawns.
 *
 * If two surfaces (desktop + cli, or two desktops) start at the same time and
 * both decide the daemon is missing, only one acquires the lock and runs the
 * spawn. The other returns null and is expected to fall back to health-polling
 * the existing-or-soon-to-exist daemon.
 *
 * Stale locks (process died mid-spawn) are reclaimed once `staleMs` elapses.
 */
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
        // Best-effort metadata; the directory itself is the real lock.
      }
      let released = false
      return {
        async release() {
          if (released) return
          released = true
          try {
            await rm(lockPath, { recursive: true, force: true })
          } catch {
            // Lock dir may have already been reaped (stale-cleanup, OS).
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
      // Another process may have just won the reclaim race; loop will retry mkdir.
    }
  }

  return null
}
