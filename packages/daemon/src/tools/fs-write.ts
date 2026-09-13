import { readFile, stat, writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import { throwIfAborted } from '../abort.js'
import { buildUnifiedDiff } from './edit-diff.js'
import { buildFileToolPosture, type FileToolPostureOptions } from './file-tool-posture.js'
import { resolveToolPath } from './path-utils.js'
import { contentEncodingSchema, decodeFileContent, normalizeFileContent } from './file-content.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

const EVIDENCE_TRUNCATION_MIN_BYTES = 1000
const EVIDENCE_TRUNCATION_RATIO = 0.75

async function protectExistingFileFromAccidentalTruncation(
  path: string,
  nextContent: string,
  overwrite: boolean,
): Promise<ToolResult | null> {
  const normalized = path.replaceAll('\\', '/')
  const protectedEvidence = normalized.includes('/_evidence/')

  try {
    const currentContent = await readFile(path, 'utf-8')
    if (
      currentContent.length >= EVIDENCE_TRUNCATION_MIN_BYTES
      && nextContent.length < currentContent.length * EVIDENCE_TRUNCATION_RATIO
      && (protectedEvidence || !overwrite)
    ) {
      return {
        output: [
          protectedEvidence
            ? `Refusing to overwrite evidence file with much shorter content: ${path}.`
            : `Refusing to overwrite an existing file with much shorter content without explicit overwrite confirmation: ${path}.`,
          `Current length is ${currentContent.length} chars; proposed length is ${nextContent.length} chars.`,
          protectedEvidence
            ? 'Read the current file and merge/update it instead of replacing the accumulated evidence ledger.'
            : 'Use fs.edit/apply_patch for a partial change. Pass overwrite:true only for a deliberate complete-file replacement after accounting for all existing content.',
        ].join('\n'),
        status: 'error',
        durationMs: 0,
        code: protectedEvidence
          ? 'EVIDENCE_TRUNCATION_PERMANENT'
          : 'ACCIDENTAL_TRUNCATION_PERMANENT',
      }
    }
  } catch {
    return null
  }

  return null
}

export function createFsWriteTool(postureOptions?: FileToolPostureOptions): ToolDefinitionRuntime {
  return {
    name: 'fs.write',
    description:
      'Write `content` to `path`, creating parent directories as needed. fs.write OVERWRITES — use fs.edit (exact-match replace) or apply_patch for partial changes. A large existing file cannot be replaced with much shorter content unless overwrite=true explicitly confirms a deliberate complete-file replacement. Evidence ledgers remain protected even with confirmation. Supports absolute paths, ~/ paths, and relative paths resolved against the active session cwd. Will fail on policy-protected paths (~/.ssh/, /etc/, secrets, *.pem/*.key).',
    resumeSafety: 'replay-risky',
    normalizeInput: normalizeFileContent,
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path to write' },
        content: { type: 'string', description: 'Content to write' },
        contentEncoding: contentEncodingSchema,
        overwrite: {
          type: 'boolean',
          description: 'Confirm a deliberate complete replacement when an existing file would be substantially shortened. Not needed for new files or ordinary same-size rewrites.',
        },
        createOnly: {
          type: 'boolean',
          description: 'Create a new file exclusively. Use true when existing files must not be overwritten; atomically fails if the path already exists, including a concurrent creation.',
        },
      },
      required: ['path', 'content'],
    },
    async recoverInterruptedExecution(input, context): Promise<ToolResult | null> {
      // Matching bytes alone cannot prove who exclusively created the file.
      if (input.createOnly === true) return null
      const path = typeof input.path === 'string' ? resolveToolPath(input.path, context.cwd) : ''
      const content = decodeFileContent(input)
      if (!path || typeof content !== 'string') {
        return null
      }

      try {
        const [fileStat, currentContent] = await Promise.all([
          stat(path),
          readFile(path, 'utf-8'),
        ])
        if (
          currentContent === content
          && fileStat.mtime.getTime() >= new Date(context.startedAt).getTime()
        ) {
          return {
            output: `Wrote ${Buffer.byteLength(content, 'utf-8')} bytes to ${path}`,
            status: 'success',
            durationMs: 0,
          }
        }
      } catch {
        return null
      }

      return null
    },
    async execute(input: Record<string, unknown>, context): Promise<ToolResult> {
      const path = typeof input.path === 'string' ? resolveToolPath(input.path, context?.cwd) : ''
      const content = decodeFileContent(input)
      if (content === undefined) return { status: 'error', code: 'INVALID_CONTENT_PERMANENT', output: 'content must be literal UTF-8 text or canonical base64 of UTF-8 text', durationMs: 0 }
      const overwrite = input.overwrite === true
      const start = Date.now()
      try {
        throwIfAborted(context?.signal, `Write to ${path} aborted`)
        const stale = await context?.workspaceMutation?.detectStale(path)
        await context?.editCheckpoint?.recordPreEdit(path)
        await mkdir(dirname(path), { recursive: true })
        throwIfAborted(context?.signal, `Write to ${path} aborted`)
        const truncation = await protectExistingFileFromAccidentalTruncation(
          path,
          content,
          overwrite,
        )
        if (truncation) {
          return {
            ...truncation,
            durationMs: Date.now() - start,
          }
        }
        const previousContent = await readFile(path, 'utf-8').catch(() => '')
        await writeFile(path, content, { encoding: 'utf-8', flag: input.createOnly === true ? 'wx' : 'w' })
        await context?.workspaceMutation?.recordWrite(path)
        const warning = stale ? `[workspace warning] ${stale.description}\n` : ''
        const editDiff = buildUnifiedDiff(path, previousContent, content)
        const executionPosture = buildFileToolPosture(postureOptions, context?.cwd)
        return {
          output: `${warning}Wrote ${Buffer.byteLength(content, 'utf-8')} bytes to ${path}`,
          status: 'success',
          durationMs: Date.now() - start,
          ...(editDiff ? { metadata: { editDiff, editPath: path } } : {}),
          ...(executionPosture ? { executionPosture } : {}),
        }
      } catch (err: unknown) {
        const errCode = (err as { code?: string }).code
        const message = (err as { message?: string }).message ?? String(err)
        const code = errCode === 'EEXIST'
          ? 'FILE_EXISTS_PERMANENT'
          : errCode === 'EACCES' || errCode === 'EPERM'
          ? 'EACCES_PERMANENT'
          : errCode === 'ENOSPC'
            ? 'ENOSPC_PERMANENT'
            : errCode === 'EISDIR'
              ? 'EISDIR_PERMANENT'
              : 'UNKNOWN_PERMANENT'
        return {
          output: message,
          status: 'error',
          durationMs: Date.now() - start,
          code,
        }
      }
    },
  }
}
