import type { Dirent } from 'node:fs'
import { createReadStream } from 'node:fs'
import { readdir, readFile, stat } from 'node:fs/promises'
import { createInterface } from 'node:readline'
import { throwIfAborted } from '../abort.js'
import { describeMissingPath } from './missing-path-hint.js'
import { resolveToolPath } from './path-utils.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

// Keep implicit reads compact enough for the model to reason about without
// flooding every later turn. Callers that genuinely need a larger contiguous
// view can still opt in with `limit`.
const DEFAULT_LIMIT = 200
const MAX_FULL_READ_BYTES = 10 * 1024 * 1024
const BINARY_SAMPLE_BYTES = 8192

/**
 * Internal observation-ledger metadata. This is never part of the public
 * fs.read input schema and must never be forwarded to tool execution. It lets
 * the coverage layer distinguish "the requested page ended here because the
 * file ended" from "the page ended here because the read was truncated".
 */
export const FS_READ_OBSERVATION_EOF_LINE = '__observationEofLine'

export function createFsReadTool(): ToolDefinitionRuntime {
  return {
    name: 'fs.read',
    description:
      'Read a text file from the filesystem. Optional offset (1-based line) and limit (line count, default 200) for paging through large files. Output lines are prefixed "NNN<tab>" with the absolute file line number; strip this prefix before reusing the text in edits. Search first and read a small slice around relevant hits; do not read an entire large source or test file speculatively. An unchanged identical view is cached, so use the prior observation instead of requesting it again. Binary-looking files are rejected unless you explicitly request a binary-safe encoding such as base64 or hex.',
    resumeSafety: 'replay-safe',
    scheduling: {
      mode: 'parallel-safe',
      resource: 'filesystem',
      key: (input) => typeof input.path === 'string' ? input.path : null,
    },
    observationCoverage: {
      covers: (observedInput, requestedInput, context) => {
        const observedPath = typeof observedInput.path === 'string'
          ? resolveToolPath(observedInput.path, context.cwd)
          : ''
        const requestedPath = typeof requestedInput.path === 'string'
          ? resolveToolPath(requestedInput.path, context.cwd)
          : ''
        if (!observedPath || observedPath !== requestedPath) return false
        const observedEncoding = String(observedInput.encoding ?? 'utf-8').toLowerCase()
        const requestedEncoding = String(requestedInput.encoding ?? 'utf-8').toLowerCase()
        if (observedEncoding !== requestedEncoding) return false
        const observedLineNumbers = effectiveLineNumbers(observedInput, observedEncoding)
        const requestedLineNumbers = effectiveLineNumbers(requestedInput, requestedEncoding)
        if (observedLineNumbers !== requestedLineNumbers) return false
        const observedOffset = normalisePositiveInt(observedInput.offset, 1)
        const requestedOffset = normalisePositiveInt(requestedInput.offset, 1)
        const observedEnd = observedOffset + normalisePositiveInt(observedInput.limit, DEFAULT_LIMIT) - 1
        const requestedEnd = requestedOffset + normalisePositiveInt(requestedInput.limit, DEFAULT_LIMIT) - 1
        const eofLine = observationEofLine(observedInput)
        if (eofLine !== undefined && requestedOffset > eofLine) return true
        const effectiveRequestedEnd = eofLine === undefined
          ? requestedEnd
          : Math.min(requestedEnd, eofLine)
        return observedOffset <= requestedOffset && observedEnd >= effectiveRequestedEnd
      },
      coversCollectively: (observedInputs, requestedInput, context) => {
        const requestedPath = typeof requestedInput.path === 'string'
          ? resolveToolPath(requestedInput.path, context.cwd)
          : ''
        if (!requestedPath) return false
        const requestedEncoding = String(requestedInput.encoding ?? 'utf-8').toLowerCase()
        const requestedLineNumbers = effectiveLineNumbers(requestedInput, requestedEncoding)
        const requestedOffset = normalisePositiveInt(requestedInput.offset, 1)
        const requestedEnd = requestedOffset
          + normalisePositiveInt(requestedInput.limit, DEFAULT_LIMIT) - 1
        const matchingInputs = observedInputs.filter((input) => {
          const path = typeof input.path === 'string'
            ? resolveToolPath(input.path, context.cwd)
            : ''
          const encoding = String(input.encoding ?? 'utf-8').toLowerCase()
          return !(
            path !== requestedPath
            || encoding !== requestedEncoding
            || effectiveLineNumbers(input, encoding) !== requestedLineNumbers
          )
        })
        const eofLine = matchingInputs.reduce<number | undefined>((latest, input) => {
          const current = observationEofLine(input)
          return current === undefined ? latest : Math.max(latest ?? 0, current)
        }, undefined)
        if (eofLine !== undefined && requestedOffset > eofLine) return true
        const effectiveRequestedEnd = eofLine === undefined
          ? requestedEnd
          : Math.min(requestedEnd, eofLine)
        const intervals = matchingInputs.flatMap((input) => {
          const start = normalisePositiveInt(input.offset, 1)
          const end = start + normalisePositiveInt(input.limit, DEFAULT_LIMIT) - 1
          return [{ start, end }]
        }).sort((left, right) => left.start - right.start)

        let coveredThrough = requestedOffset - 1
        for (const interval of intervals) {
          if (interval.end < requestedOffset) continue
          if (interval.start > coveredThrough + 1) break
          coveredThrough = Math.max(coveredThrough, interval.end)
          if (coveredThrough >= effectiveRequestedEnd) return true
        }
        return false
      },
      uncoveredInputs: (observedInputs, requestedInput, context) => {
        const requestedPath = typeof requestedInput.path === 'string'
          ? resolveToolPath(requestedInput.path, context.cwd)
          : ''
        if (!requestedPath) return undefined
        const requestedEncoding = String(requestedInput.encoding ?? 'utf-8').toLowerCase()
        const requestedLineNumbers = effectiveLineNumbers(requestedInput, requestedEncoding)
        const requestedStart = normalisePositiveInt(requestedInput.offset, 1)
        const requestedEnd = requestedStart
          + normalisePositiveInt(requestedInput.limit, DEFAULT_LIMIT) - 1
        const matchingInputs = observedInputs.filter((input) => {
          const path = typeof input.path === 'string'
            ? resolveToolPath(input.path, context.cwd)
            : ''
          const encoding = String(input.encoding ?? 'utf-8').toLowerCase()
          return !(
            path !== requestedPath
            || encoding !== requestedEncoding
            || effectiveLineNumbers(input, encoding) !== requestedLineNumbers
          )
        })
        const eofLine = matchingInputs.reduce<number | undefined>((latest, input) => {
          const current = observationEofLine(input)
          return current === undefined ? latest : Math.max(latest ?? 0, current)
        }, undefined)
        if (eofLine !== undefined && requestedStart > eofLine) return []
        const effectiveRequestedEnd = eofLine === undefined
          ? requestedEnd
          : Math.min(requestedEnd, eofLine)
        const intervals = matchingInputs.flatMap((input) => {
          const start = Math.max(requestedStart, normalisePositiveInt(input.offset, 1))
          const end = Math.min(
            effectiveRequestedEnd,
            normalisePositiveInt(input.offset, 1)
              + normalisePositiveInt(input.limit, DEFAULT_LIMIT) - 1,
          )
          return start <= end ? [{ start, end }] : []
        }).sort((left, right) => left.start - right.start)
        if (intervals.length === 0 && eofLine === undefined) return undefined

        const missing: Array<{ start: number; end: number }> = []
        let cursor = requestedStart
        for (const interval of intervals) {
          if (interval.end < cursor) continue
          if (interval.start > cursor) {
            missing.push({ start: cursor, end: interval.start - 1 })
          }
          cursor = Math.max(cursor, interval.end + 1)
          if (cursor > effectiveRequestedEnd) break
        }
        if (cursor <= effectiveRequestedEnd) {
          missing.push({ start: cursor, end: effectiveRequestedEnd })
        }
        return missing.map(({ start, end }) => ({
          ...requestedInput,
          offset: start,
          limit: end - start + 1,
        }))
      },
    },
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description:
            'File path. Supports absolute paths, ~/ paths, and relative paths resolved against the active session cwd.',
        },
        encoding: {
          type: 'string',
          description:
            'File encoding (default: utf-8). Use base64 or hex only when you intentionally need encoded binary bytes.',
        },
        offset: {
          type: 'integer',
          minimum: 1,
          description:
            '1-based line number to start reading from. Use this to continue after a truncated read.',
        },
        limit: {
          type: 'integer',
          minimum: 1,
          description: `Maximum number of lines to return (default ${DEFAULT_LIMIT}).`,
        },
        lineNumbers: {
          type: 'boolean',
          description:
            'Prefix each output line with its 1-based file line number (default true). Line numbers are for reference only; never include the leading "NNN<tab>" prefix in fs.edit oldText or apply_patch context lines.',
        },
      },
      required: ['path'],
    },
    async execute(input: Record<string, unknown>, context): Promise<ToolResult> {
      const start = Date.now()
      const rawPath = input.path
      if (typeof rawPath !== 'string' || rawPath.trim() === '') {
        return {
          output: 'fs.read requires a non-empty "path" argument',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      const path = resolveToolPath(rawPath, context?.cwd)
      const encoding = (input.encoding as BufferEncoding) ?? 'utf-8'
      const offset = normalisePositiveInt(input.offset, 1)
      const limit = normalisePositiveInt(input.limit, DEFAULT_LIMIT)
      const lineNumbers = Buffer.isEncoding(encoding)
        && isTextEncoding(encoding)
        && input.lineNumbers !== false
        && process.env.SEPILOTD_FS_READ_LINE_NUMBERS !== '0'
      try {
        throwIfAborted(context?.signal, `Read of ${path} aborted`)
        if (!Buffer.isEncoding(encoding)) {
          return {
            output: `Unsupported encoding for fs.read: ${String(encoding)}`,
            status: 'error',
            durationMs: Date.now() - start,
            code: 'INVALID_ENCODING_PERMANENT',
          }
        }

        const info = await stat(path)
        if (info.isDirectory()) {
          return {
            output: await formatDirectoryReadOutput(path),
            status: 'success',
            durationMs: Date.now() - start,
          }
        }

        if (
          isTextEncoding(encoding)
          && await fileLooksBinary(path, info.size, context?.signal)
        ) {
          return {
            output: [
              `[fs.read: ${path} appears to be binary or NUL-delimited content.]`,
              'Use media.inspect/media.extract_text for supported document formats, or call fs.read with encoding=base64/hex only when encoded bytes are required.',
            ].join('\n'),
            status: 'error',
            durationMs: Date.now() - start,
            code: 'BINARY_FILE_PERMANENT',
          }
        }

        const viewKey = JSON.stringify({
          encoding: encoding.toLowerCase(),
          offset,
          limit,
          lineNumbers,
        })
        const cached = info.size <= MAX_FULL_READ_BYTES
          ? await context?.workspaceMutation?.lookupReadObservation?.(path, viewKey)
          : undefined
        if (cached?.status === 'hit' && cached.observation) {
          return {
            output: cached.observation.output,
            status: 'success',
            durationMs: Date.now() - start,
            metadata: {
              observationCache: {
                status: 'hit',
                reason: cached.reason,
                evidenceId: cached.observation.evidenceId,
                contentHash: cached.observation.contentHash,
              },
            },
          }
        }

        if (info.size > MAX_FULL_READ_BYTES) {
          if (!isTextEncoding(encoding)) {
            return {
              output: [
                `[fs.read: ${path} is ${formatBytes(info.size)}, which exceeds the ${formatBytes(MAX_FULL_READ_BYTES)} full-read limit.]`,
                'For large binary payloads, use a purpose-built media tool or a targeted terminal command that writes output to a file instead of returning it to the agent context.',
              ].join('\n'),
              status: 'error',
              durationMs: Date.now() - start,
              code: 'FILE_TOO_LARGE_PERMANENT',
            }
          }
          const output = await streamPaginatedTextFile(
            path,
            encoding,
            offset,
            limit,
            info.size,
            lineNumbers,
            context?.signal,
          )
          await context?.workspaceMutation?.recordRead(path)
          return {
            output,
            status: 'success',
            durationMs: Date.now() - start,
          }
        }

        const content = await readFile(path, { encoding })
        const output = paginateOutput(content.toString(), offset, limit, lineNumbers)
        const observation = context?.workspaceMutation?.recordReadObservation
          ? await context.workspaceMutation.recordReadObservation(path, viewKey, output)
          : null
        if (!context?.workspaceMutation?.recordReadObservation) {
          await context?.workspaceMutation?.recordRead(path)
        }
        return {
          output,
          status: 'success',
          durationMs: Date.now() - start,
          ...(cached || observation
            ? {
                metadata: {
                  observationCache: {
                    status: cached?.status ?? 'miss',
                    reason: cached?.reason ?? 'not-cached',
                    ...(observation
                      ? {
                          evidenceId: observation.evidenceId,
                          contentHash: observation.contentHash,
                        }
                      : {}),
                  },
                },
              }
            : {}),
        }
      } catch (err: unknown) {
        if ((err as { code?: string }).code === 'EISDIR') {
          return {
            output: await formatDirectoryReadOutput(path),
            status: 'success',
            durationMs: Date.now() - start,
          }
        }
        const { message, code } = describeReadError(err, path)
        // A missing path is the model's guess, not a filesystem fault. Return
        // the surrounding evidence so the next call can be the right one
        // instead of another guess.
        const hints = code === 'ENOENT_PERMANENT'
          ? await describeMissingPath(path, {
              root: context?.workspaceRoot ?? context?.cwd,
              signal: context?.signal,
            })
          : []
        return {
          output: [message, ...hints].join('\n'),
          status: 'error',
          durationMs: Date.now() - start,
          code,
        }
      }
    },
  }
}

function effectiveLineNumbers(input: Record<string, unknown>, encoding: string): boolean {
  return Buffer.isEncoding(encoding)
    && isTextEncoding(encoding as BufferEncoding)
    && input.lineNumbers !== false
    && process.env.SEPILOTD_FS_READ_LINE_NUMBERS !== '0'
}

function normalisePositiveInt(raw: unknown, fallback: number): number {
  if (typeof raw === 'number' && Number.isFinite(raw) && raw >= 1) {
    return Math.floor(raw)
  }
  return fallback
}

function observationEofLine(input: Record<string, unknown>): number | undefined {
  const raw = input[FS_READ_OBSERVATION_EOF_LINE]
  return typeof raw === 'number' && Number.isSafeInteger(raw) && raw >= 0
    ? raw
    : undefined
}

function paginateOutput(
  content: string,
  offset: number,
  limit: number,
  lineNumbers: boolean,
): string {
  const lines = content.split('\n')
  const totalLines = lines.length
  const startIdx = offset - 1

  if (startIdx >= totalLines) {
    return `[fs.read: offset ${offset} is past end of file (${totalLines} line${totalLines === 1 ? '' : 's'} total)]`
  }

  const endIdx = Math.min(startIdx + limit, totalLines)
  const body = lines.slice(startIdx, endIdx)
  const slice = formatLineNumberedLines(body, startIdx + 1, String(endIdx).length, lineNumbers)
  const truncatedHead = startIdx > 0
  const truncatedTail = endIdx < totalLines

  if (!truncatedHead && !truncatedTail) {
    return slice
  }

  const notes: string[] = []
  if (truncatedHead) {
    notes.push(`[fs.read: skipped lines 1-${startIdx} (use offset=1 to read from the top)]`)
  }
  if (truncatedTail) {
    const remaining = totalLines - endIdx
    notes.push(
      `[fs.read: showing lines ${offset}-${endIdx} of ${totalLines}. ${remaining} more line${remaining === 1 ? '' : 's'} — call fs.read again with offset=${endIdx + 1} (and optional limit) to continue]`,
    )
  }

  return `${slice}\n${notes.join('\n')}`
}

function isTextEncoding(encoding: BufferEncoding): boolean {
  return !['base64', 'base64url', 'hex'].includes(encoding.toLowerCase())
}

async function fileLooksBinary(
  path: string,
  size: number,
  signal?: AbortSignal,
): Promise<boolean> {
  if (size === 0) return false
  throwIfAborted(signal, `Read of ${path} aborted`)
  const sample = await readFileRange(path, Math.min(size, BINARY_SAMPLE_BYTES), signal)
  return sample.includes(0)
}

async function readFileRange(
  path: string,
  bytes: number,
  signal?: AbortSignal,
): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    const stream = createReadStream(path, { start: 0, end: Math.max(0, bytes - 1) })
    const abort = (): void => {
      stream.destroy(new Error(`Read of ${path} aborted`))
    }
    stream.on('data', (chunk) => {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
    })
    stream.on('error', (err) => {
      signal?.removeEventListener('abort', abort)
      reject(err)
    })
    stream.on('end', () => {
      signal?.removeEventListener('abort', abort)
      resolve(Buffer.concat(chunks))
    })
    signal?.addEventListener('abort', abort, { once: true })
    if (signal?.aborted) {
      abort()
    }
  })
}

async function streamPaginatedTextFile(
  path: string,
  encoding: BufferEncoding,
  offset: number,
  limit: number,
  fileSize: number,
  lineNumbers: boolean,
  signal?: AbortSignal,
): Promise<string> {
  const startLine = offset
  const endLine = offset + limit - 1
  const selected: string[] = []
  let lineNumber = 0
  let truncatedTail = false

  throwIfAborted(signal, `Read of ${path} aborted`)
  const stream = createReadStream(path, { encoding })
  const rl = createInterface({ input: stream, crlfDelay: Infinity })
  const abort = (): void => {
    rl.close()
    stream.destroy(new Error(`Read of ${path} aborted`))
  }
  signal?.addEventListener('abort', abort, { once: true })
  if (signal?.aborted) abort()

  try {
    for await (const line of rl) {
      throwIfAborted(signal, `Read of ${path} aborted`)
      lineNumber += 1
      if (lineNumber >= startLine && lineNumber <= endLine) {
        selected.push(
          lineNumbers
            ? `${String(lineNumber).padStart(String(endLine).length)}\t${line}`
            : line,
        )
      } else if (lineNumber > endLine) {
        truncatedTail = true
        rl.close()
        break
      }
    }
  } finally {
    signal?.removeEventListener('abort', abort)
    stream.destroy()
  }

  if (selected.length === 0) {
    return [
      `[fs.read: offset ${offset} is past the scanned end of file (${lineNumber} line${lineNumber === 1 ? '' : 's'} seen)]`,
      `[fs.read: ${path} is ${formatBytes(fileSize)}, so it was streamed instead of loaded fully.]`,
    ].join('\n')
  }

  const notes: string[] = []
  if (startLine > 1) {
    notes.push(`[fs.read: skipped lines 1-${startLine - 1} (use offset=1 to read from the top)]`)
  }
  if (truncatedTail) {
    notes.push(
      `[fs.read: ${path} is ${formatBytes(fileSize)}, so it was streamed instead of loaded fully and total line count was omitted. Call fs.read again with offset=${endLine + 1} (and optional limit) to continue]`,
    )
  }

  return notes.length > 0 ? `${selected.join('\n')}\n${notes.join('\n')}` : selected.join('\n')
}

function formatLineNumberedLines(
  lines: string[],
  firstLine: number,
  width: number,
  lineNumbers: boolean,
): string {
  if (!lineNumbers) return lines.join('\n')
  return lines
    .map((line, idx) => `${String(firstLine + idx).padStart(width)}\t${line}`)
    .join('\n')
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`
}

async function formatDirectoryReadOutput(path: string): Promise<string> {
  let entries: Dirent[]
  try {
    entries = await readdir(path, { withFileTypes: true })
  } catch (err) {
    const message = (err as { message?: string }).message ?? String(err)
    return [
      `[fs.read: ${path} is a directory, not a file.]`,
      `Unable to list directory contents: ${message}`,
      'Use fs.glob/fs.search for discovery or fs.read on a known file path.',
    ].join('\n')
  }
  const sorted = entries
    .map((entry) => `${entry.name}${entry.isDirectory() ? '/' : ''}`)
    .sort((a, b) => a.localeCompare(b))
  const limit = 100
  const visible = sorted.slice(0, limit)
  const remaining = sorted.length - visible.length

  return [
    `[fs.read: ${path} is a directory, not a file.]`,
    'Use fs.read on one of the files below, or fs.glob/fs.search for recursive discovery.',
    ...visible,
    remaining > 0 ? `[fs.read: ${remaining} more entr${remaining === 1 ? 'y' : 'ies'} omitted]` : '',
  ].filter(Boolean).join('\n')
}

function describeReadError(
  err: unknown,
  resolvedPath: string,
): { message: string; code: string } {
  const errCode = (err as { code?: string }).code
  const message = (err as { message?: string }).message ?? String(err)
  switch (errCode) {
    case 'ENOENT':
      return { message: `File not found: ${resolvedPath}`, code: 'ENOENT_PERMANENT' }
    case 'EACCES':
    case 'EPERM':
      return { message: `Permission denied reading ${resolvedPath}`, code: 'EACCES_PERMANENT' }
    default:
      return { message, code: 'UNKNOWN_PERMANENT' }
  }
}
