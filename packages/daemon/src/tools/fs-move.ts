import { mkdir, rename, stat } from 'node:fs/promises'
import { dirname } from 'node:path'
import { throwIfAborted } from '../abort.js'
import { buildFileToolPosture, type FileToolPostureOptions } from './file-tool-posture.js'
import { resolveToolPath } from './path-utils.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

export function createFsMoveTool(postureOptions?: FileToolPostureOptions): ToolDefinitionRuntime {
  return {
    name: 'fs.move',
    description:
      'Move or rename a file or directory from `from` to `to`, creating the destination parent directories as needed. Use this to reorganise files instead of rewriting their content at a new path and deleting the original. Refuses to overwrite an existing destination unless `overwrite` is true. Supports absolute paths, ~/ paths, and relative paths resolved against the active session cwd. Will fail on policy-protected paths (~/.ssh/, /etc/, secrets, *.pem/*.key).',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        from: { type: 'string', description: 'Existing file or directory path' },
        to: { type: 'string', description: 'Destination path' },
        overwrite: {
          type: 'boolean',
          description: 'Replace the destination if it already exists (default false)',
        },
      },
      required: ['from', 'to'],
    },
    async recoverInterruptedExecution(input, context): Promise<ToolResult | null> {
      const from = typeof input.from === 'string' ? resolveToolPath(input.from, context.cwd) : ''
      const to = typeof input.to === 'string' ? resolveToolPath(input.to, context.cwd) : ''
      if (!from || !to) return null

      try {
        await stat(to)
      } catch {
        return null
      }
      try {
        await stat(from)
        return null
      } catch {
        return {
          output: `Moved ${from} to ${to}`,
          status: 'success',
          durationMs: 0,
        }
      }
    },
    async execute(input: Record<string, unknown>, context): Promise<ToolResult> {
      const from = typeof input.from === 'string' ? resolveToolPath(input.from, context?.cwd) : ''
      const to = typeof input.to === 'string' ? resolveToolPath(input.to, context?.cwd) : ''
      const overwrite = input.overwrite === true
      const start = Date.now()

      if (!from || !to) {
        return {
          output: 'fs.move requires both `from` and `to` paths.',
          status: 'error',
          durationMs: Date.now() - start,
          code: 'INVALID_INPUT_PERMANENT',
        }
      }

      try {
        throwIfAborted(context?.signal, `Move of ${from} aborted`)
        await stat(from)
      } catch (err) {
        const errCode = (err as { code?: string }).code
        if (errCode === 'ENOENT') {
          return {
            output: `Source does not exist: ${from}`,
            status: 'error',
            durationMs: Date.now() - start,
            code: 'ENOENT_PERMANENT',
          }
        }
        return {
          output: (err as { message?: string }).message ?? String(err),
          status: 'error',
          durationMs: Date.now() - start,
          code: 'UNKNOWN_PERMANENT',
        }
      }

      if (!overwrite) {
        try {
          await stat(to)
          return {
            output: [
              `Destination already exists: ${to}.`,
              'Pass overwrite:true to replace it, or move to a different path.',
            ].join(' '),
            status: 'error',
            durationMs: Date.now() - start,
            code: 'EEXIST_PERMANENT',
          }
        } catch {
        }
      }

      try {
        const stale = await context?.workspaceMutation?.detectStale(to)
        await context?.editCheckpoint?.recordPreEdit(from)
        await context?.editCheckpoint?.recordPreEdit(to)
        await mkdir(dirname(to), { recursive: true })
        throwIfAborted(context?.signal, `Move of ${from} aborted`)
        await rename(from, to)
        await context?.workspaceMutation?.recordWrite(to)
        const warning = stale ? `[workspace warning] ${stale.description}\n` : ''
        const executionPosture = buildFileToolPosture(postureOptions, context?.cwd)
        return {
          output: `${warning}Moved ${from} to ${to}`,
          status: 'success',
          durationMs: Date.now() - start,
          ...(executionPosture ? { executionPosture } : {}),
        }
      } catch (err: unknown) {
        const errCode = (err as { code?: string }).code
        const message = (err as { message?: string }).message ?? String(err)
        const code = errCode === 'EACCES' || errCode === 'EPERM'
          ? 'EACCES_PERMANENT'
          : errCode === 'ENOSPC'
            ? 'ENOSPC_PERMANENT'
            : errCode === 'EXDEV'
              ? 'EXDEV_PERMANENT'
              : 'UNKNOWN_PERMANENT'
        return {
          output: errCode === 'EXDEV'
            ? `Cannot move across filesystems in one step: ${message}`
            : message,
          status: 'error',
          durationMs: Date.now() - start,
          code,
        }
      }
    },
  }
}
