import { readFile, readdir, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import {
  basename as posixBasename,
  dirname as posixDirname,
  isAbsolute,
  join,
  relative,
  resolve,
  sep,
} from 'node:path'
import { fuzzyScore } from './fuzzy.js'
import { getFileIndex, type FileIndexEntry } from './file-index.js'
import { getRecentAttachments } from './attachment-history.js'

const MAX_ATTACHMENT_CANDIDATES = 8
const MAX_FUZZY_PRESCORE = 200
const MAX_ATTACHMENT_UPLOADS = 32
const MAX_DIRECTORY_SUMMARY_FILES = 200
const MAX_DIRECTORY_SCAN_FILES = 2_000

export interface AttachmentReference {
  raw: string
  path: string
  start: number
  end: number
  quoted: boolean
}

export interface AttachmentCandidate {
  path: string
  basename: string
  dir: string
  isDirectory: boolean
  matchIndices?: number[]
}

interface PreparedAttachmentUpload {
  path: string
  filename: string
  content: Buffer
}

export interface PreparedAttachmentPlan {
  uploads: PreparedAttachmentUpload[]
  notices: string[]
}

const BOUNDARY_CHARS = new Set([',', '!', '?', ';', ':', ')', ']', '}', '>', '<', '`', '|'])
const KOREAN_ATTACHMENT_PARTICLES = [
  '으로',
  '에서',
  '은',
  '는',
  '이',
  '가',
  '을',
  '를',
  '와',
  '과',
  '로',
  '에',
  '의',
  '도',
  '만',
] as const

function isReservedAtReference(path: string): boolean {
  const normalized = path.toLowerCase()
  return normalized.startsWith('skill:') || normalized.startsWith('skills:')
}

function isReservedAtReferencePrefix(path: string, nextChar: string | undefined): boolean {
  const normalized = path.toLowerCase()
  return (normalized === 'skill' || normalized === 'skills') && nextChar === ':'
}

function isBoundary(value: string | undefined): boolean {
  if (value === undefined) return true
  if (/\s/.test(value)) return true
  return BOUNDARY_CHARS.has(value)
}

function shrinkReferenceToKnownPrefix(
  reference: AttachmentReference,
  input: string,
  knownPaths: ReadonlySet<string>,
): AttachmentReference {
  if (reference.quoted) return reference
  if (knownPaths.has(reference.path)) return reference

  for (const known of knownPaths) {
    if (known.length > reference.path.length && known.startsWith(reference.path)) {
      return reference
    }
  }

  for (const known of knownPaths) {
    const directoryPrefix = `${known.replace(/[\\/]+$/, '')}/`
    if (!reference.path.startsWith(directoryPrefix)) {
      continue
    }

    const tail = reference.path.slice(directoryPrefix.length)
    if (
      !KOREAN_ATTACHMENT_PARTICLES.includes(tail as (typeof KOREAN_ATTACHMENT_PARTICLES)[number])
    ) {
      continue
    }

    const newEnd = reference.start + 1 + directoryPrefix.length
    return {
      ...reference,
      path: directoryPrefix,
      end: newEnd,
      raw: input.slice(reference.start, newEnd),
    }
  }

  for (let length = reference.path.length - 1; length > 0; length -= 1) {
    const candidate = reference.path.slice(0, length)
    const nextChar = reference.path[length]
    if (nextChar === '/' || nextChar === '\\') {
      continue
    }
    if (knownPaths.has(candidate)) {
      const newEnd = reference.start + 1 + length
      return {
        ...reference,
        path: candidate,
        end: newEnd,
        raw: input.slice(reference.start, newEnd),
      }
    }
  }

  return reference
}

export function extractAttachmentReferences(
  input: string,
  knownPaths?: ReadonlySet<string>,
): AttachmentReference[] {
  const references: AttachmentReference[] = []

  for (let index = 0; index < input.length; index += 1) {
    if (input[index] !== '@' || !isBoundary(input[index - 1])) {
      continue
    }

    const start = index
    const next = input[index + 1]
    if (!next || isBoundary(next)) {
      continue
    }

    if (next === '"' || next === "'") {
      const quote = next
      let cursor = index + 2
      while (cursor < input.length && input[cursor] !== quote) {
        cursor += 1
      }

      const end = cursor < input.length ? cursor + 1 : input.length
      const path = input.slice(index + 2, cursor < input.length ? cursor : input.length)
      if (path) {
        references.push({
          raw: input.slice(start, end),
          path,
          start,
          end,
          quoted: true,
        })
      }
      index = end - 1
      continue
    }

    let cursor = index + 1
    while (cursor < input.length && !isBoundary(input[cursor])) {
      cursor += 1
    }

    const path = input.slice(index + 1, cursor)
    if (path && !isReservedAtReference(path) && !isReservedAtReferencePrefix(path, input[cursor])) {
      let reference: AttachmentReference = {
        raw: input.slice(start, cursor),
        path,
        start,
        end: cursor,
        quoted: false,
      }
      if (knownPaths) {
        reference = shrinkReferenceToKnownPrefix(reference, input, knownPaths)
      }
      references.push(reference)
      index = reference.end - 1
      continue
    }
    index = cursor - 1
  }

  return references
}

export function findActiveAttachmentReference(
  input: string,
  knownPaths?: ReadonlySet<string>,
): AttachmentReference | null {
  const references = extractAttachmentReferences(input, knownPaths)
  const active = references.at(-1) ?? null
  if (!active) return null
  return active.end === input.length ? active : null
}

export function resolveAttachmentPath(inputPath: string, cwd = process.cwd()): string {
  if (inputPath === '~') {
    return homedir()
  }

  if (inputPath.startsWith('~/')) {
    return join(homedir(), inputPath.slice(2))
  }

  if (isAbsolute(inputPath)) {
    return resolve(inputPath)
  }

  return resolve(cwd, inputPath)
}

export function toAttachmentPath(absolutePath: string, cwd = process.cwd()): string {
  const relativePath = relative(cwd, absolutePath)
  if (
    relativePath &&
    relativePath !== '' &&
    !relativePath.startsWith('..') &&
    !isAbsolute(relativePath)
  ) {
    return relativePath
  }
  return absolutePath
}

function formatAttachmentReference(inputPath: string): string {
  if (/\s/.test(inputPath)) {
    return `@"${inputPath}"`
  }
  return `@${inputPath}`
}

export function applyAttachmentCompletion(
  input: string,
  reference: AttachmentReference,
  candidate: string,
): string {
  const preservedTail =
    !reference.quoted &&
    candidate.length > 0 &&
    reference.path !== candidate &&
    reference.path.startsWith(candidate)
      ? reference.path.slice(candidate.length)
      : ''
  const tail = input.slice(reference.end)
  const completionSuffix = !preservedTail && tail.length === 0 ? ' ' : ''
  return (
    input.slice(0, reference.start) +
    formatAttachmentReference(candidate) +
    preservedTail +
    completionSuffix +
    tail
  )
}

function toPosix(value: string): string {
  return sep === '/' ? value : value.split(sep).join('/')
}

function joinAttachmentDisplayPath(basePath: string, relativePath: string): string {
  const trimmedBase = basePath.replace(/[\\/]+$/, '')
  const separator = trimmedBase.includes('\\') && !trimmedBase.includes('/') ? '\\' : '/'
  const normalizedRelative =
    separator === '\\' ? relativePath.split('/').join('\\') : toPosix(relativePath)
  if (!trimmedBase) return normalizedRelative
  return `${trimmedBase}${separator}${normalizedRelative}`
}

function makeDirectorySummaryFilename(inputPath: string): string {
  const normalized = toPosix(inputPath).replace(/\/+$/, '')
  const base = posixBasename(normalized) || 'folder'
  return `${base}.folder.txt`
}

async function buildDirectoryReferenceUpload(
  inputPath: string,
  resolvedPath: string,
): Promise<{
  upload: PreparedAttachmentUpload
  notice: string | null
}> {
  const listedPaths: string[] = []
  let totalFiles = 0
  let scanTruncated = false

  async function walk(currentDir: string): Promise<void> {
    if (scanTruncated) return

    const entries = await readdir(currentDir, { withFileTypes: true })
    entries.sort((left, right) => left.name.localeCompare(right.name))

    for (const entry of entries) {
      if (scanTruncated) return

      const absolutePath = join(currentDir, entry.name)
      if (entry.isDirectory()) {
        await walk(absolutePath)
        continue
      }
      if (!entry.isFile()) {
        continue
      }

      totalFiles += 1
      if (listedPaths.length < MAX_DIRECTORY_SUMMARY_FILES) {
        listedPaths.push(
          joinAttachmentDisplayPath(inputPath, toPosix(relative(resolvedPath, absolutePath))),
        )
      }
      if (totalFiles >= MAX_DIRECTORY_SCAN_FILES) {
        scanTruncated = true
        return
      }
    }
  }

  await walk(resolvedPath)

  const lines = [
    `[Folder reference: ${inputPath}]`,
    'This folder was attached as a lightweight reference summary.',
    'Inspect specific files under this path with workspace tools as needed.',
    '',
    `Files scanned: ${totalFiles}${scanTruncated ? ` (stopped at ${MAX_DIRECTORY_SCAN_FILES} for safety)` : ''}`,
  ]

  if (listedPaths.length === 0) {
    lines.push('No files were found under this folder.')
  } else {
    lines.push(`Files listed below: ${listedPaths.length}`)
    lines.push(...listedPaths.map((path) => `- ${path}`))
    if (totalFiles > listedPaths.length) {
      lines.push(`- ... ${totalFiles - listedPaths.length} more file(s) not listed`)
    }
  }

  const notice =
    scanTruncated || totalFiles > listedPaths.length
      ? `Attached folder reference ${formatAttachmentReference(inputPath)} as a summary (${totalFiles} files scanned, ${listedPaths.length} listed). Narrow the path if you need a smaller scope.`
      : `Attached folder reference ${formatAttachmentReference(inputPath)} as a summary. Inspect specific files under that path as needed.`

  return {
    upload: {
      path: inputPath,
      filename: makeDirectorySummaryFilename(inputPath),
      content: Buffer.from(lines.join('\n'), 'utf-8'),
    },
    notice,
  }
}

function entryFromIndex(entry: FileIndexEntry): AttachmentCandidate {
  return {
    path: entry.isDirectory ? `${entry.path}/` : entry.path,
    basename: entry.basename,
    dir: entry.dir,
    isDirectory: entry.isDirectory,
  }
}

function rankCandidates(
  scored: Array<{ candidate: AttachmentCandidate; score: number; mtimeMs: number }>,
  recent: readonly string[],
): AttachmentCandidate[] {
  const recencyRank = new Map<string, number>()
  recent.forEach((path, index) => recencyRank.set(path, recent.length - index))

  return scored
    .slice()
    .sort((left, right) => {
      const leftRecency = recencyRank.get(left.candidate.path) ?? 0
      const rightRecency = recencyRank.get(right.candidate.path) ?? 0
      if (leftRecency !== rightRecency) return rightRecency - leftRecency
      if (left.score !== right.score) return right.score - left.score
      if (left.mtimeMs !== right.mtimeMs) return right.mtimeMs - left.mtimeMs
      return left.candidate.path.localeCompare(right.candidate.path)
    })
    .slice(0, MAX_ATTACHMENT_CANDIDATES)
    .map((item) => item.candidate)
}

async function attachMtime(
  candidates: AttachmentCandidate[],
  rootDir: string,
): Promise<Map<string, number>> {
  const slice = candidates.slice(0, MAX_FUZZY_PRESCORE)
  const entries = await Promise.all(
    slice.map(async (candidate) => {
      try {
        const info = await stat(resolve(rootDir, candidate.path))
        return [candidate.path, info.mtimeMs] as const
      } catch {
        return [candidate.path, 0] as const
      }
    }),
  )
  return new Map(entries)
}

async function listDirectoryCandidates(
  rootDir: string,
  dirRelativePath: string,
): Promise<AttachmentCandidate[]> {
  const index = await getFileIndex(rootDir)
  const normalizedDir = toPosix(dirRelativePath).replace(/\/+$/, '')
  return index.entries.filter((entry) => entry.dir === normalizedDir).map(entryFromIndex)
}

export async function resolveAttachmentCandidates(
  inputPath: string,
  rootDir: string = process.cwd(),
): Promise<AttachmentCandidate[]> {
  const normalized = inputPath.trim()

  if (normalized.endsWith('/')) {
    const dirPart = normalized.slice(0, -1)
    const items = await listDirectoryCandidates(rootDir, dirPart)
    return items.slice(0, MAX_ATTACHMENT_CANDIDATES)
  }

  const containsSlash = normalized.includes('/')
  const index = await getFileIndex(rootDir)
  const recent = getRecentAttachments()

  let pool: FileIndexEntry[]
  let query: string

  if (containsSlash) {
    const dirScope = toPosix(posixDirname(normalized))
    const normalizedScope = dirScope === '.' ? '' : dirScope
    query = posixBasename(normalized)
    pool = index.entries.filter((entry) => entry.dir === normalizedScope)
  } else {
    query = normalized
    pool = index.entries
  }

  if (query.length === 0) {
    const baseCandidates = pool.map(entryFromIndex)
    const mtimes = await attachMtime(baseCandidates, rootDir)
    const scored = baseCandidates.map((candidate) => ({
      candidate,
      score: 0,
      mtimeMs: mtimes.get(candidate.path) ?? 0,
    }))
    return rankCandidates(scored, recent)
  }

  function collectMatches(forQuery: string) {
    const found: Array<{ candidate: AttachmentCandidate; score: number }> = []
    for (const entry of pool) {
      const target = containsSlash ? entry.basename : entry.path
      const result = fuzzyScore(forQuery, target)
      if (result === null) continue
      found.push({
        candidate: { ...entryFromIndex(entry), matchIndices: result.indices },
        score: result.score,
      })
    }
    return found
  }

  let matches = collectMatches(query)
  if (matches.length === 0) {
    const maxBackoff = Math.min(8, query.length - 1)
    for (let trim = 1; trim <= maxBackoff; trim += 1) {
      const shorter = query.slice(0, query.length - trim)
      matches = collectMatches(shorter)
      if (matches.length > 0) break
    }
  }

  matches.sort((left, right) => right.score - left.score)
  const top = matches.slice(0, MAX_FUZZY_PRESCORE)
  const mtimes = await attachMtime(
    top.map((item) => item.candidate),
    rootDir,
  )
  const scored = top.map((item) => ({
    candidate: item.candidate,
    score: item.score,
    mtimeMs: mtimes.get(item.candidate.path) ?? 0,
  }))

  return rankCandidates(scored, recent)
}

export async function prepareAttachmentUploads(
  inputs: ReadonlyArray<{ path: string }>,
  cwd: string = process.cwd(),
): Promise<PreparedAttachmentPlan> {
  const descriptors = new Map<string, PreparedAttachmentUpload>()
  const notices: string[] = []

  for (const input of inputs) {
    const resolvedPath = resolveAttachmentPath(input.path, cwd)
    const info = await stat(resolvedPath)

    if (info.isFile()) {
      const fileKey = `file:${resolvedPath}`
      if (descriptors.has(fileKey)) {
        continue
      }
      descriptors.set(fileKey, {
        path: input.path,
        filename: posixBasename(toPosix(resolvedPath)),
        content: await readFile(resolvedPath),
      })
      continue
    }

    if (!info.isDirectory()) {
      throw new Error(`${input.path} is not a file or directory`)
    }

    const directoryKey = `dir:${resolvedPath}`
    if (descriptors.has(directoryKey)) {
      continue
    }

    const summary = await buildDirectoryReferenceUpload(input.path, resolvedPath)
    descriptors.set(directoryKey, summary.upload)
    if (summary.notice) {
      notices.push(summary.notice)
    }
  }

  const uploads = [...descriptors.values()]
  if (uploads.length > MAX_ATTACHMENT_UPLOADS) {
    throw new Error(
      `Attachment set is too large (${uploads.length} items). Max ${MAX_ATTACHMENT_UPLOADS}. Attach fewer files or use a folder reference.`,
    )
  }

  return { uploads, notices }
}
