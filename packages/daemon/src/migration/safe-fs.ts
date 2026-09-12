import {
  copyFileSync as fsCopyFileSync,
  lstatSync,
  readFileSync as fsReadFileSync,
} from 'node:fs'

/**
 * Wrappers used by migration steps that read or copy files from a
 * caller-supplied `sourcePath`. A hostile source tree could include a
 * symlink (`SKILL.md -> /etc/shadow`) and trick a follow-target copy
 * into placing privileged host content under `~/.sepilotd/`. These
 * helpers refuse symlinks outright so each migration step opts in to
 * "real files only".
 */

function refuseSymlink(path: string): void {
  const st = lstatSync(path)
  if (st.isSymbolicLink()) {
    throw new Error(`refusing to follow symlink at ${path}`)
  }
}

export function safeReadFileSync(
  path: string,
  encoding?: BufferEncoding,
): string | Buffer {
  refuseSymlink(path)
  return encoding ? fsReadFileSync(path, encoding) : fsReadFileSync(path)
}

export function safeCopyFileSync(src: string, dest: string): void {
  refuseSymlink(src)
  // The dest is created by the migration step (mkdir + copyFile) so its
  // own symlink-ness is enforced at write time by the filesystem semantics
  // we already use (copyFile follows the dest path, but our migration
  // dirs are freshly created so there is no symlink to follow).
  fsCopyFileSync(src, dest)
}
