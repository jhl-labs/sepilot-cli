import {
  copyFileSync as fsCopyFileSync,
  lstatSync,
  readFileSync as fsReadFileSync,
} from 'node:fs'

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
  fsCopyFileSync(src, dest)
}
