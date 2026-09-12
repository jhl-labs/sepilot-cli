import type { Dirent } from 'node:fs'
import { readdir } from 'node:fs/promises'
import { throwIfAborted } from '../abort.js'
import { describeMissingPath } from './missing-path-hint.js'
import { resolveToolCwd } from './path-utils.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

const DEFAULT_LIMIT = 200
const MAX_LIMIT = 1_000

function normalizeNonNegativeInteger(raw: unknown, fallback: number): number {
  const value = typeof raw === 'number'
    ? raw
    : typeof raw === 'string' && raw.trim()
      ? Number(raw)
      : fallback
  if (!Number.isFinite(value) || value < 0) return fallback
  return Math.floor(value)
}

function normalizeLimit(raw: unknown): number {
  const value = normalizeNonNegativeInteger(raw, DEFAULT_LIMIT)
  return Math.max(1, Math.min(MAX_LIMIT, value))
}

function safeEntryName(name: string): string {
  return name.replace(/[\u0000-\u001f\u007f]/g, (character) =>
    `\\u${character.charCodeAt(0).toString(16).padStart(4, '0')}`,
  )
}

function formatEntry(entry: Dirent): string {
  const name = safeEntryName(entry.name)
  if (entry.isDirectory()) return `${name}/`
  if (entry.isSymbolicLink()) return `${name}@`
  return name
}

function describeListError(error: unknown, directory: string): { message: string; code: string } {
  const code = (error as { code?: string }).code
  switch (code) {
    case 'ENOENT':
      return { message: `Directory not found: ${directory}`, code: 'ENOENT_PERMANENT' }
    case 'ENOTDIR':
      return { message: `Not a directory: ${directory}`, code: 'ENOTDIR_PERMANENT' }
    case 'EACCES':
    case 'EPERM':
      return { message: `Permission denied listing ${directory}`, code: 'EACCES_PERMANENT' }
    default:
      return {
        message: (error as { message?: string }).message ?? String(error),
        code: 'UNKNOWN_PERMANENT',
      }
  }
}

/**
 * Lists one directory level without invoking an OS shell. Keeping directory
 * inventory separate from fs.read/fs.glob gives models one portable operation
 * for `ls`, `dir`, and `Get-ChildItem` style requests.
 */
export function createFsListTool(): ToolDefinitionRuntime {
  return {
    name: 'fs.list',
    description:
      'List the immediate files and subdirectories in a directory without using an OS shell. Use this for requests such as "list files in the current folder". Omit cwd to list the active session cwd. Directory names end in "/" and symbolic links end in "@". Hidden entries are omitted by default but the result reports how many were omitted. Use fs.glob instead for recursive or pattern-based discovery.',
    resumeSafety: 'replay-safe',
    scheduling: {
      mode: 'parallel-safe',
      resource: 'filesystem',
      key: (input) => typeof input.cwd === 'string' ? input.cwd : '.',
    },
    observationCoverage: {
      covers: (observedInput, requestedInput, context) => {
        const observedDirectory = resolveToolCwd(observedInput.cwd, context.cwd)
        const requestedDirectory = resolveToolCwd(requestedInput.cwd, context.cwd)
        if (observedDirectory !== requestedDirectory) return false
        if ((observedInput.hidden === true) !== (requestedInput.hidden === true)) return false
        const observedOffset = normalizeNonNegativeInteger(observedInput.offset, 0)
        const requestedOffset = normalizeNonNegativeInteger(requestedInput.offset, 0)
        const observedEnd = observedOffset + normalizeLimit(observedInput.limit)
        const requestedEnd = requestedOffset + normalizeLimit(requestedInput.limit)
        return observedOffset <= requestedOffset && observedEnd >= requestedEnd
      },
    },
    inputSchema: {
      type: 'object',
      properties: {
        cwd: {
          type: 'string',
          description:
            'Directory to list. Supports absolute, ~/ and relative paths. Defaults to the active session cwd.',
        },
        hidden: {
          type: 'boolean',
          description: 'Include entries whose names begin with a dot (default false).',
        },
        limit: {
          type: 'integer',
          minimum: 1,
          maximum: MAX_LIMIT,
          description: `Maximum entries to return (default ${DEFAULT_LIMIT}, maximum ${MAX_LIMIT}).`,
        },
        offset: {
          type: 'integer',
          minimum: 0,
          description: 'Zero-based entry offset for paging through a large directory.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const directory = resolveToolCwd(input.cwd, context?.cwd)
      const limit = normalizeLimit(input.limit)
      const offset = normalizeNonNegativeInteger(input.offset, 0)

      try {
        throwIfAborted(context?.signal, `Directory listing of ${directory} aborted`)
        const entries = await readdir(directory, { withFileTypes: true })
        throwIfAborted(context?.signal, `Directory listing of ${directory} aborted`)
        const hiddenCount = input.hidden === true
          ? 0
          : entries.filter((entry) => entry.name.startsWith('.')).length
        const visible = entries
          .filter((entry) => input.hidden === true || !entry.name.startsWith('.'))
          .sort((left, right) => left.name.localeCompare(right.name))
        const page = visible.slice(offset, offset + limit).map(formatEntry)

        if (entries.length === 0) {
          return {
            output: '[empty directory]',
            status: 'success',
            durationMs: Date.now() - startedAt,
          }
        }

        if (page.length === 0) {
          const hiddenNote = hiddenCount > 0
            ? `${hiddenCount} hidden entr${hiddenCount === 1 ? 'y' : 'ies'} omitted; retry with hidden=true`
            : ''
          return {
            output: offset > 0
              ? `[no visible entries after offset ${offset}${hiddenNote ? `; ${hiddenNote}` : ''}]`
              : `[no visible entries; ${hiddenNote}]`,
            status: 'success',
            durationMs: Date.now() - startedAt,
          }
        }

        const nextOffset = offset + page.length
        const notes: string[] = []
        if (nextOffset < visible.length) {
          notes.push(`[truncated after ${page.length} entries; continue with offset ${nextOffset}]`)
        }
        if (hiddenCount > 0) {
          notes.push(
            `[${hiddenCount} hidden entr${hiddenCount === 1 ? 'y' : 'ies'} omitted; retry with hidden=true]`,
          )
        }
        const output = [...page, ...notes].join('\n')
        return {
          output,
          status: 'success',
          durationMs: Date.now() - startedAt,
        }
      } catch (error) {
        const described = describeListError(error, directory)
        const hints = described.code === 'ENOENT_PERMANENT'
          ? await describeMissingPath(directory, {
              root: context?.workspaceRoot ?? context?.cwd,
              signal: context?.signal,
            })
          : []
        return {
          output: [described.message, ...hints].join('\n'),
          status: 'error',
          code: described.code,
          durationMs: Date.now() - startedAt,
        }
      }
    },
  }
}
