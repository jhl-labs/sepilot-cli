import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, extname, join, relative, resolve } from 'node:path'
import { sepilotdHome } from '../../storage/home.js'
import { assertSafeId, isSafeId } from '../../utils/safe-id.js'
import { secureDir, secureFile } from '../../utils/secure-file.js'

const MIME_EXTENSIONS: Record<string, string> = {
  'image/png': '.png',
  'image/jpeg': '.jpg',
  'image/webp': '.webp',
  'image/gif': '.gif',
  'video/mp4': '.mp4',
  'video/webm': '.webm',
}

const EXTENSION_MIME: Record<string, string> = Object.fromEntries(
  Object.entries(MIME_EXTENSIONS).map(([mime, extension]) => [extension, mime]),
)

const SUPPORTED_EXTENSIONS = Object.values(MIME_EXTENSIONS)

function normalizeOutputId(outputId: string): string {
  const extension = extname(outputId).toLowerCase()
  return SUPPORTED_EXTENSIONS.includes(extension) ? outputId.slice(0, -extension.length) : outputId
}

function extensionForMime(mime: string): string {
  return MIME_EXTENSIONS[mime] ?? '.bin'
}

function assertInsideRoot(root: string, file: string): void {
  const rel = relative(root, file)
  if (rel.startsWith('..') || resolve(rel) === rel) {
    throw new Error('invalid image-gen path')
  }
}

export function pathFor(jobId: string, outputId: string, mime = 'image/png'): string {
  assertSafeId(jobId, 'image-gen jobId')
  const normalizedOutputId = normalizeOutputId(outputId)
  assertSafeId(normalizedOutputId, 'image-gen outputId')
  const root = resolve(join(sepilotdHome(), 'image-gen'))
  const file = resolve(join(root, jobId, `${normalizedOutputId}${extensionForMime(mime)}`))
  assertInsideRoot(root, file)
  return file
}

export function writeOutput(
  jobId: string,
  outputId: string,
  bytes: Buffer,
  mime = 'image/png',
): string {
  const file = pathFor(jobId, outputId, mime)
  const dir = dirname(file)
  mkdirSync(dir, { recursive: true, mode: 0o700 })
  secureDir(dir)
  writeFileSync(file, bytes)
  secureFile(file)
  return file
}

export function readOutput(fileId: string): { bytes: Buffer; mime: string } | null {
  const [jobIdPart, outputPart] = fileId.includes(':')
    ? fileId.split(':', 2)
    : [fileId.split('-')[0], fileId]
  if (!jobIdPart || !outputPart) return null
  const extension = extname(outputPart).toLowerCase()
  const outputId = normalizeOutputId(outputPart)
  if (!isSafeId(jobIdPart) || !isSafeId(outputId)) return null
  const candidates = extension
    ? [EXTENSION_MIME[extension] ?? 'application/octet-stream']
    : Object.keys(MIME_EXTENSIONS)
  for (const mime of candidates) {
    const file = pathFor(jobIdPart, outputId, mime)
    if (existsSync(file)) return { bytes: readFileSync(file), mime }
  }
  return null
}

export function deleteJobOutputs(jobId: string): void {
  assertSafeId(jobId, 'image-gen jobId')
  const root = resolve(join(sepilotdHome(), 'image-gen'))
  const dir = resolve(join(root, jobId))
  assertInsideRoot(root, dir)
  rmSync(dir, { recursive: true, force: true })
}
