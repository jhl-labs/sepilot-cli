import { execFile, type ExecFileOptionsWithStringEncoding } from 'node:child_process'
import { mkdtemp, readdir, readFile, realpath, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, extname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { promisify } from 'node:util'
import type { ContentPart } from '@sepilotd/core'
import { extractPptxText, PptxExtractionError } from './pptx.js'

const execFileAsync = promisify(execFile)

/**
 * Minimal slice of `pdf-parse`'s `PDFParse` that this module uses.
 *
 * `pdf-parse` pulls in `pdfjs-dist`, whose top-level code references DOM globals
 * (`DOMMatrix`, …) and probes for `@napi-rs/canvas`. A static `import` of it at
 * module load would crash a `bun build --compile` binary at startup, so the real
 * parser is `await import('pdf-parse')`d lazily inside {@link defaultExtractionDeps}
 * — only when a PDF actually needs extracting. Tests pass their own
 * `createPdfParser` and never load `pdf-parse`.
 */
export interface PdfParserLike {
  getText(): Promise<{ text: string }>
  destroy(): Promise<unknown>
}

const MIME_MAP: Record<string, string> = {
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp',
  '.tif': 'image/tiff', '.tiff': 'image/tiff', '.svg': 'image/svg+xml',
  '.pdf': 'application/pdf',
  '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
}

export const TEXT_FILE_EXTENSIONS: readonly string[] = [
  '.txt', '.md', '.ts', '.tsx', '.js', '.jsx', '.py', '.json',
  '.yaml', '.yml', '.sh', '.css', '.html', '.xml', '.csv', '.log',
  '.toml', '.rs', '.go', '.java', '.c', '.cpp', '.h', '.rb',
  '.php', '.sql', '.mmd', '.mermaid',
]

const TEXT_EXT = new Set(TEXT_FILE_EXTENSIONS)

const MAX_FILE_SIZE = 20 * 1024 * 1024 // 20MB
// Pre-read size ceiling for the text-extraction path (PDF/OCR/text). Reuses the
// same 20MB budget as inline media so a crafted large document cannot be read
// wholesale into memory before any bound applies.
const MAX_EXTRACTABLE_FILE_SIZE = MAX_FILE_SIZE
// Sniff window used to reject binary payloads mislabeled with a text extension
// (a NUL byte never occurs in valid UTF-8 text).
const BINARY_SNIFF_BYTES = 8192
export const MAX_CHAT_ATTACHMENTS = 16
export const MAX_CHAT_ATTACHMENT_TOTAL_BYTES = 64 * 1024 * 1024
const OCR_DEFAULT_LANGUAGES = ['eng']
// Hard ceiling on PDF pages rasterized for OCR, independent of caller input, so
// a huge scanned PDF cannot fan out an unbounded number of subprocesses.
const OCR_MAX_PAGES_CEILING = 200
export const OCR_DEFAULT_MAX_PAGES = 20
const OCR_DEFAULT_CONCURRENCY = 2
const OCR_MAX_CONCURRENCY = 8
const OCR_DEFAULT_TIMEOUT_MS = 60_000

type ExecFileResult = { stdout: string; stderr: string }
type ExecFileFn = (
  file: string,
  args?: readonly string[],
  options?: ExecFileOptionsWithStringEncoding,
) => Promise<ExecFileResult>

interface ExtractionDeps {
  execFile: ExecFileFn
  mkdtemp: (prefix: string) => Promise<string>
  readDir: (path: string) => Promise<string[]>
  rm: (path: string, options: { recursive: boolean; force: boolean }) => Promise<void>
  createPdfParser: (data: Buffer) => PdfParserLike | Promise<PdfParserLike>
}

const defaultExtractionDeps: ExtractionDeps = {
  execFile: (file, args = [], options) => execFileAsync(file, args, options) as Promise<ExecFileResult>,
  mkdtemp,
  readDir: (path) => readdir(path),
  rm,
  createPdfParser: async (data) => {
    const { PDFParse } = await import('pdf-parse')
    return new PDFParse({ data }) as PdfParserLike
  },
}

export interface ExtractableFileOptions {
  allowOcr?: boolean
  ocrLanguages?: string[]
  ocrMaxPages?: number
}

export class FileExtractionError extends Error {
  constructor(
    message: string,
    readonly code:
      | 'UNSUPPORTED_FILE_TYPE'
      | 'NO_EXTRACTABLE_TEXT'
      | 'OCR_UNAVAILABLE'
      | 'OCR_FAILED'
      | 'FILE_TOO_LARGE'
      | 'BINARY_CONTENT'
      | 'INVALID_DOCUMENT'
      | 'ARCHIVE_LIMIT_EXCEEDED',
  ) {
    super(message)
    this.name = 'FileExtractionError'
  }
}

export type ChatAttachmentLimitCode = 'TOO_MANY_ATTACHMENTS' | 'ATTACHMENTS_TOO_LARGE'

export interface ChatAttachmentBudgetSource {
  path?: string
  url?: string
}

export interface ChatAttachmentLimitOptions {
  maxAttachments?: number
  maxTotalBytes?: number
}

export class AttachmentLimitError extends Error {
  readonly statusCode = 413

  constructor(
    readonly code: ChatAttachmentLimitCode,
    message: string,
  ) {
    super(message)
    this.name = 'AttachmentLimitError'
  }
}

export async function assertChatAttachmentLimits(
  attachments: readonly ChatAttachmentBudgetSource[],
  options: ChatAttachmentLimitOptions = {},
): Promise<void> {
  const maxAttachments = options.maxAttachments ?? MAX_CHAT_ATTACHMENTS
  if (attachments.length > maxAttachments) {
    throw new AttachmentLimitError(
      'TOO_MANY_ATTACHMENTS',
      `Too many chat attachments: ${attachments.length} (max ${maxAttachments})`,
    )
  }

  const maxTotalBytes = options.maxTotalBytes ?? MAX_CHAT_ATTACHMENT_TOTAL_BYTES
  let totalBytes = 0
  for (const attachment of attachments) {
    const byteLength = await estimateAttachmentByteLength(attachment)
    if (byteLength === null) continue
    totalBytes += byteLength
    if (totalBytes > maxTotalBytes) {
      throw new AttachmentLimitError(
        'ATTACHMENTS_TOO_LARGE',
        `Chat attachments are too large: ${totalBytes} bytes (max ${maxTotalBytes})`,
      )
    }
  }
}

async function estimateAttachmentByteLength(
  attachment: ChatAttachmentBudgetSource,
): Promise<number | null> {
  if (attachment.path) {
    try {
      const info = await stat(attachment.path)
      return info.isFile() ? info.size : null
    } catch {
      return null
    }
  }

  if (attachment.url?.startsWith('data:')) {
    return estimateDataUrlDecodedBytes(attachment.url)
  }

  return null
}

function estimateDataUrlDecodedBytes(url: string): number | null {
  const match = /^data:[^,]*;base64,([\s\S]*)$/i.exec(url)
  if (!match) return null
  const data = match[1].replace(/\s/g, '')
  if (!data) return 0
  const padding = data.endsWith('==') ? 2 : data.endsWith('=') ? 1 : 0
  return Math.max(0, Math.floor((data.length * 3) / 4) - padding)
}

// A client-controlled `attachments[].path` is otherwise an arbitrary-file-read
// (LFI): the daemon would `readFile` any path it can access (e.g. /etc/passwd,
// source, logs) and stream it back inside the model context. Attachment paths
// must be constrained to a fileId-registered upload or a path canonically
// contained within an allow-listed root (uploads dir + the run workspace).

export class AttachmentPathError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'AttachmentPathError'
  }
}

// Legacy escape hatch: only when the operator explicitly opts in. Default off.
export function attachmentPathLegacyAllowed(): boolean {
  return process.env.SEPILOTD_ALLOW_ATTACHMENT_PATH === '1'
}

function isPathWithinRoot(candidate: string, root: string): boolean {
  if (!root) return false
  const rel = relative(resolve(root), resolve(candidate))
  // Empty means candidate === root; a non-'..' relative that is not absolute
  // means candidate sits inside root.
  return rel === '' || (!rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel))
}

/**
 * Return the canonical attachment path only if it is a trusted upload path or
 * is contained within one of the allow-listed roots; otherwise throw. When the
 * operator has enabled the legacy escape hatch, any path resolves (unsafe).
 */
export function assertAttachmentPathAllowed(
  candidatePath: string,
  options: { allowedRoots: readonly string[]; trusted?: boolean },
): string {
  const resolved = resolve(candidatePath)
  if (options.trusted || attachmentPathLegacyAllowed()) {
    return resolved
  }
  for (const root of options.allowedRoots) {
    if (root && isPathWithinRoot(resolved, root)) {
      return resolved
    }
  }
  throw new AttachmentPathError(
    `Attachment path is outside the allowed roots (uploads/workspace): ${candidatePath}`,
  )
}

/**
 * Resolve a client-controlled attachment path only after both the lexical and
 * canonical filesystem boundaries agree. The lexical guard rejects obvious
 * escapes without touching outside paths; the realpath guard then rejects an
 * in-workspace symlink or junction whose target is outside the workspace.
 */
async function resolveAttachmentPathAllowed(
  candidatePath: string,
  options: { allowedRoots: readonly string[]; trusted?: boolean },
): Promise<string> {
  const lexicallyAllowed = assertAttachmentPathAllowed(candidatePath, options)
  if (options.trusted || attachmentPathLegacyAllowed()) return lexicallyAllowed

  let canonicalCandidate: string
  try {
    canonicalCandidate = await realpath(lexicallyAllowed)
  } catch (error) {
    throw new AttachmentPathError(
      `Attachment path could not be resolved: ${error instanceof Error ? error.message : String(error)}`,
    )
  }

  for (const root of options.allowedRoots) {
    if (!root) continue
    try {
      const canonicalRoot = await realpath(resolve(root))
      if (isPathWithinRoot(canonicalCandidate, canonicalRoot)) return canonicalCandidate
    } catch {
      // A missing optional root (for example an upload directory that has not
      // been created yet) cannot authorize the candidate. Check the next root.
    }
  }

  throw new AttachmentPathError(
    `Attachment path resolves outside the allowed roots (uploads/workspace): ${candidatePath}`,
  )
}

export async function fileToContentPart(
  filePath: string,
  displayName?: string,
): Promise<ContentPart> {
  const ext = extname(filePath).toLowerCase()
  if (TEXT_EXT.has(ext)) {
    const text = await readFile(filePath, 'utf-8')
    const filename = displayName ?? basename(filePath)
    return {
      type: 'text',
      text: `[File: ${filename}]\n${text}`,
    }
  }

  if (ext === '.pptx') {
    await assertExtractableFileSize(filePath)
    const data = await readFile(filePath)
    try {
      return {
        type: 'text',
        text: extractPptxText(data, displayName ?? basename(filePath)),
      }
    } catch (error) {
      if (error instanceof PptxExtractionError) {
        throw new FileExtractionError(
          error.message,
          error.code === 'PPTX_LIMIT_EXCEEDED'
            ? 'ARCHIVE_LIMIT_EXCEEDED'
            : 'INVALID_DOCUMENT',
        )
      }
      throw error
    }
  }

  const mimeType = MIME_MAP[ext]
  if (!mimeType) throw new Error(`Unsupported file type: ${ext}`)

  const data = await readFile(filePath)
  if (data.length > MAX_FILE_SIZE) throw new Error(`File too large: ${data.length} bytes (max ${MAX_FILE_SIZE})`)

  const base64 = data.toString('base64')

  if (mimeType.startsWith('image/')) {
    return { type: 'image', source: { type: 'base64', mediaType: mimeType, data: base64 } }
  }
  return { type: 'document', source: { type: 'base64', mediaType: mimeType, data: base64 } }
}

export interface AttachmentContentInput {
  type?: string
  path?: string
  url?: string
  filename?: string
  // Set for fileId-resolved uploads whose server-generated path is trusted.
  trusted?: boolean
}

export interface SkippedAttachment {
  source: 'path' | 'url' | 'unknown'
  filename?: string
  reason: string
}

export interface BuiltAttachmentContentParts {
  parts: ContentPart[]
  skipped: SkippedAttachment[]
}

export interface BuildAttachmentContentPartsOptions {
  allowedRoots: readonly string[]
  /**
   * Keep a presentation attachment as a lightweight reference instead of
   * expanding every slide into the first model request. The presentation
   * review skill then opens and reads exactly one slide through Office tools.
   */
  deferPptxTextExtraction?: boolean
}

/**
 * Turn chat attachments into model ContentParts, enforcing the path-containment
 * guard for client-controlled paths (fileId uploads are trusted). Malformed,
 * oversized, wrong-type, or out-of-root attachments are collected in `skipped`
 * (with a reason) instead of being silently dropped, so callers can surface and
 * log them.
 */
export async function buildAttachmentContentParts(
  attachments: readonly AttachmentContentInput[],
  options: BuildAttachmentContentPartsOptions,
): Promise<BuiltAttachmentContentParts> {
  const parts: ContentPart[] = []
  const skipped: SkippedAttachment[] = []
  for (const att of attachments) {
    try {
      if (att.path) {
        const safePath = await resolveAttachmentPathAllowed(att.path, {
          allowedRoots: options.allowedRoots,
          trusted: att.trusted === true,
        })
        if (options.deferPptxTextExtraction && isPptxFilePath(safePath)) {
          parts.push(deferredPresentationReference(att.filename ?? basename(safePath)))
        } else {
          parts.push(await fileToContentPart(safePath, att.filename))
        }
      } else if (att.url) {
        if (
          options.deferPptxTextExtraction
          && (att.filename?.toLowerCase().endsWith('.pptx')
            || new URL(att.url).pathname.toLowerCase().endsWith('.pptx'))
        ) {
          parts.push(deferredPresentationReference(att.filename ?? basename(new URL(att.url).pathname)))
        } else {
          parts.push(urlToContentPart(att.url))
        }
      } else {
        skipped.push({ source: 'unknown', reason: 'attachment has neither a path nor a url' })
      }
    } catch (error) {
      skipped.push({
        source: att.path ? 'path' : att.url ? 'url' : 'unknown',
        filename: att.filename,
        reason: error instanceof Error ? error.message : String(error),
      })
    }
  }
  return { parts, skipped }
}

function deferredPresentationReference(filename: string): ContentPart {
  const safeFilename = filename
    .replace(/[\u0000-\u001f\u007f\[\]]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 240) || 'presentation.pptx'
  return {
    type: 'text',
    text: [
      `[Presentation attachment: ${safeFilename}]`,
      'Full slide text is intentionally deferred for one-slide-at-a-time review.',
      'Use fs.glob to locate this exact filename inside the current workspace, then use the read-only presentation tools to open it and read only the requested slide.',
      'If no workspace-contained file matches, ask the user to place or select the presentation in the workspace; do not open an upload-cache path or bypass the workspace boundary.',
    ].join('\n'),
  }
}

export function urlToContentPart(url: string, mediaType?: string): ContentPart {
  if (url.startsWith('data:')) {
    const match = /^data:([^;,]+)?(;base64)?,([\s\S]*)$/i.exec(url)
    if (!match?.[2]) {
      throw new Error('Invalid data URL attachment: expected data:<mime>;base64,<data>')
    }
    const mime = mediaType ?? match[1] ?? 'application/octet-stream'
    const data = match[3].replace(/\s/g, '')
    if (
      data.length === 0
      || data.length % 4 !== 0
      || !/^[A-Za-z0-9+/]*={0,2}$/.test(data)
    ) {
      throw new Error('Invalid data URL attachment: base64 payload is malformed')
    }

    if (mime.startsWith('image/')) {
      return { type: 'image', source: { type: 'base64', mediaType: mime, data } }
    }
    return { type: 'document', source: { type: 'base64', mediaType: mime, data } }
  }

  const ext = new URL(url).pathname.split('.').pop()?.toLowerCase() ?? ''
  const mime = mediaType ?? MIME_MAP[`.${ext}`] ?? 'application/octet-stream'

  if (mime.startsWith('image/')) {
    return { type: 'image', source: { type: 'url', mediaType: mime, data: url } }
  }
  return { type: 'document', source: { type: 'url', mediaType: mime, data: url } }
}

export function isImageType(mimeType: string): boolean {
  return mimeType.startsWith('image/')
}

export function isTextFilePath(filePath: string): boolean {
  return TEXT_EXT.has(extname(filePath).toLowerCase())
}

export function isPdfFilePath(filePath: string): boolean {
  return extname(filePath).toLowerCase() === '.pdf'
}

export function isPptxFilePath(filePath: string): boolean {
  return extname(filePath).toLowerCase() === '.pptx'
}

export function isOcrImageFilePath(filePath: string): boolean {
  return ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff', '.gif'].includes(extname(filePath).toLowerCase())
}

export async function readTextFileContent(filePath: string): Promise<string> {
  if (!isTextFilePath(filePath)) {
    throw new FileExtractionError(
      `Text extraction is not supported for ${extname(filePath).toLowerCase() || 'unknown'} files`,
      'UNSUPPORTED_FILE_TYPE',
    )
  }
  return readFile(filePath, 'utf-8')
}

function normalizeExtractedText(text: string): string {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .join('\n')
    .trim()
}

function getOcrLanguages(options?: ExtractableFileOptions): string[] {
  if (options?.ocrLanguages?.length) return options.ocrLanguages
  const env = process.env.SEPILOTD_OCR_LANGS?.trim()
  if (!env) return OCR_DEFAULT_LANGUAGES
  return env
    .split(/[+,]/)
    .map((part) => part.trim())
    .filter(Boolean)
}

function getOcrTimeoutMs(): number {
  const raw = Number(process.env.SEPILOTD_OCR_TIMEOUT_MS)
  return Number.isFinite(raw) && raw > 0 ? Math.trunc(raw) : OCR_DEFAULT_TIMEOUT_MS
}

function getOcrConcurrency(): number {
  const raw = Number(process.env.SEPILOTD_OCR_CONCURRENCY)
  if (Number.isFinite(raw) && raw >= 1) return Math.min(OCR_MAX_CONCURRENCY, Math.trunc(raw))
  return OCR_DEFAULT_CONCURRENCY
}

// Resolve a subprocess file argument to an absolute path so a model-controlled
// leading-dash path (e.g. "-r99999.png", "--tessdata-dir=/x") can never be
// parsed as a CLI flag by pdftoppm/tesseract. execFile already blocks shell
// injection; this closes the remaining option-smuggling vector.
function toSafeSubprocessPath(filePath: string): string {
  return isAbsolute(filePath) ? filePath : resolve(filePath)
}

// Run an async mapper over items with a bounded number of in-flight tasks so a
// large PDF cannot spawn one OCR subprocess per page simultaneously (fork bomb).
async function mapWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length)
  let cursor = 0
  const workerCount = Math.max(1, Math.min(limit, items.length))
  const runWorker = async (): Promise<void> => {
    while (true) {
      const index = cursor++
      if (index >= items.length) return
      results[index] = await fn(items[index], index)
    }
  }
  await Promise.all(Array.from({ length: workerCount }, runWorker))
  return results
}

async function assertExtractableFileSize(filePath: string): Promise<void> {
  let info: Awaited<ReturnType<typeof stat>>
  try {
    info = await stat(filePath)
  } catch {
    // Let the downstream read surface a proper ENOENT/permission error.
    return
  }
  if (info.isFile() && info.size > MAX_EXTRACTABLE_FILE_SIZE) {
    throw new FileExtractionError(
      `File too large to extract: ${info.size} bytes (max ${MAX_EXTRACTABLE_FILE_SIZE})`,
      'FILE_TOO_LARGE',
    )
  }
}

function assertNotBinary(data: Buffer, filePath: string): void {
  const sample = data.subarray(0, BINARY_SNIFF_BYTES)
  if (sample.includes(0)) {
    throw new FileExtractionError(
      `Refusing to extract text from a binary file: ${basename(filePath)}`,
      'BINARY_CONTENT',
    )
  }
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message
  return String(error)
}

function getCommandFailureDetails(error: unknown): string {
  if (typeof error === 'object' && error !== null && 'stderr' in error && typeof (error as { stderr?: unknown }).stderr === 'string') {
    const stderr = (error as { stderr: string }).stderr.trim()
    if (stderr) return stderr
  }
  return getErrorMessage(error)
}

async function extractTextFromPdfBuffer(
  data: Buffer,
  deps: ExtractionDeps,
): Promise<string> {
  const parser = await deps.createPdfParser(data)
  try {
    const result = await parser.getText()
    return normalizeExtractedText(result.text)
  } finally {
    await parser.destroy()
  }
}

async function runTesseractOcr(
  imagePath: string,
  options: ExtractableFileOptions,
  deps: ExtractionDeps,
): Promise<string> {
  const languages = getOcrLanguages(options)
  const args = [toSafeSubprocessPath(imagePath), 'stdout', '-l', languages.join('+'), '--psm', '6']
  try {
    const { stdout } = await deps.execFile('tesseract', args, {
      maxBuffer: 16 * 1024 * 1024,
      timeout: getOcrTimeoutMs(),
      killSignal: 'SIGKILL',
    })
    const text = normalizeExtractedText(stdout)
    if (!text) {
      throw new FileExtractionError('OCR completed but no text was recognized.', 'NO_EXTRACTABLE_TEXT')
    }
    return text
  } catch (error) {
    if (typeof error === 'object' && error !== null && 'code' in error && (error as { code?: unknown }).code === 'ENOENT') {
      throw new FileExtractionError(
        'OCR fallback requires the "tesseract" binary to be installed.',
        'OCR_UNAVAILABLE',
      )
    }
    if (error instanceof FileExtractionError) throw error
    throw new FileExtractionError(
      `OCR extraction failed: ${getCommandFailureDetails(error)}`,
      'OCR_FAILED',
    )
  }
}

async function extractPdfWithOcr(
  filePath: string,
  options: ExtractableFileOptions,
  deps: ExtractionDeps,
): Promise<string> {
  const tmpDir = await deps.mkdtemp(join(tmpdir(), 'sepilot-pdf-ocr-'))
  const imagePrefix = join(tmpDir, 'page')
  try {
    const args = ['-png', '-r', '200']
    if (options.ocrMaxPages) {
      const cappedPages = Math.min(OCR_MAX_PAGES_CEILING, Math.max(1, Math.trunc(options.ocrMaxPages)))
      args.push('-f', '1', '-l', String(cappedPages))
    }
    // `--` ends option parsing so a leading-dash file path cannot be smuggled as
    // a flag; the path itself is also resolved to an absolute path.
    args.push('--', toSafeSubprocessPath(filePath), imagePrefix)
    try {
      await deps.execFile('pdftoppm', args, {
        maxBuffer: 16 * 1024 * 1024,
        timeout: getOcrTimeoutMs(),
        killSignal: 'SIGKILL',
      })
    } catch (error) {
      if (typeof error === 'object' && error !== null && 'code' in error && (error as { code?: unknown }).code === 'ENOENT') {
        throw new FileExtractionError(
          'OCR fallback requires the "pdftoppm" binary to be installed.',
          'OCR_UNAVAILABLE',
        )
      }
      throw new FileExtractionError(
        `PDF rasterization failed before OCR: ${getCommandFailureDetails(error)}`,
        'OCR_FAILED',
      )
    }

    const pageImages = (await deps.readDir(tmpDir))
      .filter((entry) => entry.endsWith('.png'))
      .sort((left, right) => left.localeCompare(right, undefined, { numeric: true }))
      .map((entry) => join(tmpDir, entry))

    if (pageImages.length === 0) {
      throw new FileExtractionError('OCR fallback could not rasterize any PDF pages.', 'OCR_FAILED')
    }

    const pages = await mapWithConcurrency(
      pageImages,
      getOcrConcurrency(),
      (pagePath) => runTesseractOcr(pagePath, options, deps),
    )
    const text = normalizeExtractedText(pages.join('\n\n'))
    if (!text) {
      throw new FileExtractionError('OCR completed but no text was recognized.', 'NO_EXTRACTABLE_TEXT')
    }
    return text
  } finally {
    await deps.rm(tmpDir, { recursive: true, force: true })
  }
}

export async function readExtractableFileContentWithDeps(
  filePath: string,
  options: ExtractableFileOptions = {},
  deps: ExtractionDeps,
): Promise<string> {
  await assertExtractableFileSize(filePath)

  if (isTextFilePath(filePath)) {
    const raw = await readFile(filePath)
    assertNotBinary(raw, filePath)
    return raw.toString('utf-8')
  }

  if (isPdfFilePath(filePath)) {
    const data = await readFile(filePath)
    try {
      const text = await extractTextFromPdfBuffer(data, deps)
      if (!text) throw new FileExtractionError('No extractable text found in PDF', 'NO_EXTRACTABLE_TEXT')
      return text
    } catch (error) {
      if (options.allowOcr === false || !(error instanceof FileExtractionError) || error.code !== 'NO_EXTRACTABLE_TEXT') {
        throw error
      }
      return extractPdfWithOcr(filePath, options, deps)
    }
  }

  if (isPptxFilePath(filePath)) {
    await assertExtractableFileSize(filePath)
    const data = await readFile(filePath)
    try {
      return extractPptxText(data, basename(filePath))
    } catch (error) {
      if (error instanceof PptxExtractionError) {
        throw new FileExtractionError(
          error.message,
          error.code === 'PPTX_LIMIT_EXCEEDED'
            ? 'ARCHIVE_LIMIT_EXCEEDED'
            : 'INVALID_DOCUMENT',
        )
      }
      throw error
    }
  }

  if (isOcrImageFilePath(filePath)) {
    if (options.allowOcr === false) {
      throw new FileExtractionError(
        `Text extraction requires OCR for ${extname(filePath).toLowerCase()} image files.`,
        'UNSUPPORTED_FILE_TYPE',
      )
    }
    return runTesseractOcr(filePath, options, deps)
  }

  throw new FileExtractionError(
    `Text extraction is not supported for ${extname(filePath).toLowerCase() || 'unknown'} files`,
    'UNSUPPORTED_FILE_TYPE',
  )
}

export async function readExtractableFileContent(
  filePath: string,
  options: ExtractableFileOptions = {},
): Promise<string> {
  return readExtractableFileContentWithDeps(filePath, options, defaultExtractionDeps)
}

export function getSupportedExtensions(): string[] {
  return Object.keys(MIME_MAP)
}

export function getSupportedTextExtensions(): string[] {
  return [...TEXT_FILE_EXTENSIONS]
}
