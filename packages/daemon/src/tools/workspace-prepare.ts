import { lstat, mkdir, realpath, stat } from 'node:fs/promises'
import { isAbsolute, join, relative } from 'node:path'
import { throwIfAborted } from '../abort.js'
import { resolveToolPath } from './path-utils.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'
import type { TerminalRunner } from './terminal.js'
import { prepareReviewWorkspace, ReviewWorkspaceError } from './review-workspace.js'

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

export function createWorkspacePrepareTool(options: { reviewRunner?: TerminalRunner } = {}): ToolDefinitionRuntime {
  return {
    name: 'workspace.prepare',
    description:
      'Prepare a directory directly inside the active workspace root, or the explicitly selected cwd of a policy session when no strict root is set. mode="project" (default) creates a new project directory. mode="review" prepares an independent detached Git worktree for an exact locally available ref/commit inside an existing repository, without switching its original branch, discarding changes or fetching remotely. Review mode requires ref and an isolated network-disabled runner; fetch using a separately authorized tool first. path must be a portable single-segment name; absolute paths, separators, "." and ".." are rejected. Existing paths require onExisting="reuse"; review reuse additionally requires the same clean detached worktree and exact revision. Use the returned path for subsequent tools. Do not treat a project directory as a review checkout.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        mode: { type: 'string', enum: ['project', 'review'], description: 'Project directory creation or detached review checkout (default: project)' },
        ref: { type: 'string', description: 'Locally available Git ref or exact fetched commit revision; required only for review mode' },
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
      additionalProperties: false,
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (input.mode !== undefined && input.mode !== 'project' && input.mode !== 'review'
        || input.ref !== undefined && input.mode !== 'review'
        || input.onExisting !== undefined && input.onExisting !== 'error' && input.onExisting !== 'reuse') {
        return failure('Invalid mode/reuse option; ref is only valid for review mode.', startedAt, 'INVALID_INPUT_PERMANENT')
      }
      const requested = typeof input.path === 'string' ? input.path.trim() : ''
      if (!requested || !PROJECT_DIRECTORY_PATTERN.test(requested) || requested === '.' || requested === '..') {
        return failure(
          'workspace.prepare requires a portable single-segment project name using letters, numbers, dot, underscore, or hyphen.',
          startedAt,
          'INVALID_INPUT_PERMANENT',
        )
      }

      const preparationRoot = context?.workspaceRoot
        ?? (context?.workspaceIsolation === 'policy' && context.cwd && isAbsolute(context.cwd)
          ? context.cwd : undefined)
      if (!context || !preparationRoot) {
        return failure(
          'workspace.prepare requires a workspace root or an explicitly selected absolute cwd in a policy session.',
          startedAt,
          'WORKSPACE_REQUIRED_USER',
        )
      }

      try {
        throwIfAborted(context?.signal, 'Workspace preparation aborted')
        const root = await realpath(resolveToolPath(preparationRoot))
        if (!(await stat(root)).isDirectory()) {
          return failure(`Workspace root is not a directory: ${root}`, startedAt, 'ENOTDIR_PERMANENT')
        }

        const candidate = join(root, requested)
        if (!isInside(root, candidate)) {
          return failure('Project path escapes the active workspace root.', startedAt, 'WORKSPACE_ESCAPE_PERMANENT')
        }
        if (input.mode === 'review') {
          const prepared = await prepareReviewWorkspace(root, candidate, input, context, options.reviewRunner)
          return { output: JSON.stringify(prepared), status: 'success', durationMs: Date.now() - startedAt }
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

        throwIfAborted(context?.signal, 'Workspace preparation aborted')
        const entry = await lstat(candidate)
        if (entry.isSymbolicLink()) {
          return failure('Refusing to use a symbolic link as a project directory.', startedAt, 'SYMLINK_PERMANENT')
        }
        const prepared = await realpath(candidate)
        const info = await stat(prepared)
        if (!info.isDirectory() || !isInside(root, prepared)) {
          return failure('Prepared project path is not a directory inside the workspace root.', startedAt, 'WORKSPACE_ESCAPE_PERMANENT')
        }

        await context?.workspaceMutation?.recordWrite(prepared)
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
          error instanceof ReviewWorkspaceError ? error.code : code === 'EACCES' || code === 'EPERM'
            ? 'EACCES_PERMANENT'
            : code === 'ENOSPC'
              ? 'ENOSPC_PERMANENT'
              : 'WORKSPACE_PREPARE_PERMANENT',
        )
      }
    },
  }
}
