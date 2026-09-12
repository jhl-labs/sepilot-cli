import { appendFile, mkdir, readFile, stat } from 'node:fs/promises'
import { dirname } from 'node:path'
import { throwIfAborted } from '../abort.js'
import { buildFileToolPosture, type FileToolPostureOptions } from './file-tool-posture.js'
import { resolveToolPath } from './path-utils.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

export function createFsAppendTool(postureOptions?: FileToolPostureOptions): ToolDefinitionRuntime {
  return {
    name: 'fs.append',
    description:
      'Append `content` to `path`, creating parent directories as needed. Use this for growing large generated artifacts section-by-section instead of rewriting the whole file. Supports absolute paths, ~/ paths, and relative paths resolved against the active session cwd. Will fail on policy-protected paths (~/.ssh/, /etc/, secrets, *.pem/*.key).',
    resumeSafety: 'replay-risky',
    scheduling: {
      mode: 'parallel-safe',
      resource: 'filesystem',
      key: (input) => (typeof input.path === 'string' ? input.path : null),
    },
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path to append to' },
        content: { type: 'string', description: 'Content to append' },
      },
      required: ['path', 'content'],
    },
    async recoverInterruptedExecution(input, context): Promise<ToolResult | null> {
      const path = typeof input.path === 'string' ? resolveToolPath(input.path, context.cwd) : ''
      const content = typeof input.content === 'string' ? input.content : ''
      if (!path || !content) {
        return null
      }

      try {
        const [fileStat, currentContent] = await Promise.all([
          stat(path),
          readFile(path, 'utf-8'),
        ])
        if (
          currentContent.endsWith(content)
          && fileStat.mtime.getTime() >= new Date(context.startedAt).getTime()
        ) {
          return {
            output: `Appended ${content.length} bytes to ${path}`,
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
      const start = Date.now()
      const path = typeof input.path === 'string' ? resolveToolPath(input.path, context?.cwd) : ''
      const content = typeof input.content === 'string' ? input.content : ''
      if (!path) {
        return {
          output: 'path is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      if (!content) {
        return {
          output: 'content is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }

      try {
        throwIfAborted(context?.signal, `Append to ${path} aborted`)
        const stale = await context?.workspaceMutation?.detectStale(path)
        await context?.editCheckpoint?.recordPreEdit(path)
        await mkdir(dirname(path), { recursive: true })
        throwIfAborted(context?.signal, `Append to ${path} aborted`)
        await appendFile(path, content, 'utf-8')
        await context?.workspaceMutation?.recordWrite(path)
        const warning = stale ? `[workspace warning] ${stale.description}\n` : ''
        const executionPosture = buildFileToolPosture(postureOptions, context?.cwd)
        return {
          output: `${warning}Appended ${content.length} bytes to ${path}`,
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
