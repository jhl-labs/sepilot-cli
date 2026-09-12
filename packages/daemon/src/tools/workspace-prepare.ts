import { lstat, mkdir, realpath, stat } from 'node:fs/promises'
import { isAbsolute, join, relative } from 'node:path'
import { throwIfAborted } from '../abort.js'
import { resolveToolPath } from './path-utils.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

const PROJECT_DIRECTORY_PATTERN = /^[a-z0-9](?:[a-z0-9._-]{0,78}[a-z0-9])?$/i

function failure(output: string, startedAt: number, code: string): ToolResult {
  return {
    output,
    status: 'error',
    durationMs: Date.now() - startedAt,
    code,
  }
}

function isInside(root: string, candidate: string): boolean {
  const rel = relative(root, candidate)
  return rel === '' || (!isAbsolute(rel) && rel !== '..' && !rel.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`))
}

/**
 * Prepare one project directory under the immutable workspace collection root.
 * The input is deliberately a single portable path segment: callers cannot use
 * this convenience capability to select a broader filesystem root.
 */
export function createWorkspacePrepareTool(): ToolDefinitionRuntime {
  return {
    name: 'workspace.prepare',
    description:
      'Create one new project directory directly under the active workspace root before starting a new development project. `path` must be a portable single-segment name such as `status-dashboard`; absolute paths, separators, `.` and `..` are rejected. Defaults to refusing an existing path; set onExisting="reuse" only when the user asked to continue an existing project. Use the returned absolute path as cwd/path for subsequent file, Git, build, test, and process tools. Do not call this for information-only work or when the active workspace already is the requested project.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'Portable single-segment project directory name (1-80 characters)',
        },
        onExisting: {
          type: 'string',
          enum: ['error', 'reuse'],
          description: 'Whether an existing real directory may be reused (default: error)',
        },
      },
      required: ['path'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const requested = typeof input.path === 'string' ? input.path.trim() : ''
      if (!requested || !PROJECT_DIRECTORY_PATTERN.test(requested) || requested === '.' || requested === '..') {
        return failure(
          'workspace.prepare requires a portable single-segment project name using letters, numbers, dot, underscore, or hyphen.',
          startedAt,
          'INVALID_INPUT_PERMANENT',
        )
      }

      if (!context?.workspaceRoot) {
        return failure(
          'workspace.prepare requires an active workspaceRoot.',
          startedAt,
          'WORKSPACE_REQUIRED_USER',
        )
      }

      try {
        throwIfAborted(context.signal, 'Workspace preparation aborted')
        const root = await realpath(resolveToolPath(context.workspaceRoot))
        if (!(await stat(root)).isDirectory()) {
          return failure(`Workspace root is not a directory: ${root}`, startedAt, 'ENOTDIR_PERMANENT')
        }

        const candidate = join(root, requested)
        if (!isInside(root, candidate)) {
          return failure('Project path escapes the active workspace root.', startedAt, 'WORKSPACE_ESCAPE_PERMANENT')
        }

        let created = false
        try {
          await mkdir(candidate, { recursive: false })
          created = true
        } catch (error) {
          if ((error as { code?: string }).code !== 'EEXIST') throw error
          if (input.onExisting !== 'reuse') {
            return failure(
              `Project directory already exists: ${candidate}. Choose a different name or explicitly set onExisting="reuse".`,
              startedAt,
              'EEXIST_PERMANENT',
            )
          }
        }

        throwIfAborted(context.signal, 'Workspace preparation aborted')
        const entry = await lstat(candidate)
        if (entry.isSymbolicLink()) {
          return failure('Refusing to use a symbolic link as a project directory.', startedAt, 'SYMLINK_PERMANENT')
        }
        const prepared = await realpath(candidate)
        const info = await stat(prepared)
        if (!info.isDirectory() || !isInside(root, prepared)) {
          return failure('Prepared project path is not a directory inside the workspace root.', startedAt, 'WORKSPACE_ESCAPE_PERMANENT')
        }

        await context.workspaceMutation?.recordWrite(prepared)
        return {
          output: JSON.stringify({ path: prepared, created, reused: !created }),
          status: 'success',
          durationMs: Date.now() - startedAt,
        }
      } catch (error) {
        const code = (error as { code?: string }).code
        return failure(
          error instanceof Error ? error.message : String(error),
          startedAt,
          code === 'EACCES' || code === 'EPERM'
            ? 'EACCES_PERMANENT'
            : code === 'ENOSPC'
              ? 'ENOSPC_PERMANENT'
              : 'WORKSPACE_PREPARE_PERMANENT',
        )
      }
    },
  }
}
