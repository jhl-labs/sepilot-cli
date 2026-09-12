import { chmodSync } from 'node:fs'
import { chmod as chmodAsync } from 'node:fs/promises'

const POSIX = process.platform !== 'win32'

/**
 * Enforce `0600` on a file we just created or replaced. CLAUDE.md
 * requires sensitive files (session jsonl, memory db, team-docs PATs,
 * personal-docs markdown, backup tarballs) to be 0600 — without this
 * helper they default to 0644 and become readable by every other OS
 * user on the same host. Windows is a no-op because POSIX modes do not
 * apply; rely on ACLs there.
 */
export function secureFile(path: string): void {
  if (!POSIX) return
  try { chmodSync(path, 0o600) } catch { /* tolerate */ }
}

export async function secureFileAsync(path: string): Promise<void> {
  if (!POSIX) return
  try { await chmodAsync(path, 0o600) } catch { /* tolerate */ }
}

/**
 * Enforce `0700` on a directory we just created. Pair with `secureFile`
 * on the contained files; together they prevent stat-based traversal of
 * the daemon's private state on shared hosts.
 */
export function secureDir(path: string): void {
  if (!POSIX) return
  try { chmodSync(path, 0o700) } catch { /* tolerate */ }
}
