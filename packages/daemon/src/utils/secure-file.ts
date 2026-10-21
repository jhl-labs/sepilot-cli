import { chmodSync } from 'node:fs'
import { chmod as chmodAsync } from 'node:fs/promises'

const POSIX = process.platform !== 'win32'

export function secureFile(path: string): void {
  if (!POSIX) return
  try { chmodSync(path, 0o600) } catch {}
}

export async function secureFileAsync(path: string): Promise<void> {
  if (!POSIX) return
  try { await chmodAsync(path, 0o600) } catch {}
}

export function secureDir(path: string): void {
  if (!POSIX) return
  try { chmodSync(path, 0o700) } catch {}
}
