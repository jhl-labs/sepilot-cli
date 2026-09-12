import { readFile, writeFile } from 'node:fs/promises'
import { throwIfAborted } from '../abort.js'
import { buildUnifiedDiff } from './edit-diff.js'
import { buildFileToolPosture, type FileToolPostureOptions } from './file-tool-posture.js'
import { resolveToolPath } from './path-utils.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

function editRecoveryHint(path: string): string {
  return [
    '[edit recovery]',
    `fs.edit could not apply an exact replacement in ${path}.`,
    'Next action: call fs.read on this path to inspect the current content, then retry with byte-exact unique oldText or use apply_patch for a structural edit.',
    'If oldText was copied from fs.read output, remove the leading "NNN<tab>" line-number prefixes first; they are display-only and never part of the file.',
    'Do not retry the same fs.edit arguments until the file has been read again.',
    '[/edit recovery]',
  ].join('\n')
}

function withEditRecovery(output: string, path: string): string {
  return `${output}\n${editRecoveryHint(path)}`
}

function workspaceStaleError(path: string): string {
  return `WORKSPACE_STALE_PERMANENT: ${path} changed on disk after your last read. Re-read the file and re-apply your edit against the current content.`
}

function matchLineNumbers(content: string, oldText: string): number[] {
  const lines: number[] = []
  let index = content.indexOf(oldText)
  while (index !== -1) {
    lines.push(content.slice(0, index).split('\n').length)
    index = content.indexOf(oldText, index + oldText.length)
  }
  return lines
}

export function createFsEditTool(postureOptions?: FileToolPostureOptions): ToolDefinitionRuntime {
  return {
    name: 'fs.edit',
    description:
      'Replace `oldText` with `newText` in `path`. `oldText` must match byte-for-byte including whitespace and indentation — read the file with fs.read first if you are not sure of the current content. Include enough surrounding context to make the match unique; otherwise prefer apply_patch for structural edits. Use `replaceAll: true` only when every occurrence should change.',
    scheduling: {
      mode: 'parallel-safe',
      resource: 'filesystem',
      key: (input) => (typeof input.path === 'string' ? input.path : null),
    },
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Absolute file path to edit.' },
        oldText: { type: 'string', description: 'Exact text to replace.' },
        newText: { type: 'string', description: 'Replacement text.' },
        replaceAll: {
          type: 'boolean',
          description: 'Replace all matches instead of the first match.',
        },
      },
      required: ['path', 'oldText', 'newText'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const path = typeof input.path === 'string' ? resolveToolPath(input.path, context?.cwd) : ''
      const oldText = typeof input.oldText === 'string' ? input.oldText : ''
      const newText = typeof input.newText === 'string' ? input.newText : ''
      if (!path) {
        return {
          output: 'path is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      if (!oldText) {
        return {
          output: 'oldText is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      if (oldText === newText) {
        return {
          output: withEditRecovery('oldText and newText must be different for fs.edit', path),
          status: 'error',
          durationMs: Date.now() - start,
        }
      }

      try {
        throwIfAborted(context?.signal, `Edit of ${path} aborted`)
        const content = await readFile(path, 'utf-8')
        if (!content.includes(oldText)) {
          return {
            output: withEditRecovery(`oldText not found in ${path}`, path),
            status: 'error',
            durationMs: Date.now() - start,
            code: 'EDIT_CONTEXT_MISMATCH_PERMANENT',
          }
        }

        const replaceAll = input.replaceAll === true
        const matchCount = content.split(oldText).length - 1
        if (!replaceAll && matchCount > 1) {
          return {
            output: withEditRecovery(
              `oldText matches ${matchCount} times in ${path} (lines ${matchLineNumbers(content, oldText).join(', ')}). Include more surrounding context to make the match unique, or pass replaceAll: true to change every occurrence.`,
              path,
            ),
            status: 'error',
            durationMs: Date.now() - start,
            code: 'EDIT_CONTEXT_AMBIGUOUS_PERMANENT',
          }
        }
        // split/join for BOTH paths: String.replace would interpret $-
        // substitution patterns ($&, $$, $') in newText and silently
        // corrupt the file. After the ambiguity gate above, the
        // non-replaceAll path is guaranteed exactly one match, so
        // split/join is an equivalent literal replacement.
        const nextContent = content.split(oldText).join(newText)
        const replacements = replaceAll ? matchCount : 1
        const stale = await context?.workspaceMutation?.detectStale(path)
        if (stale && process.env.SEPILOTD_WORKSPACE_STALE_SOFT !== '1') {
          return {
            output: workspaceStaleError(path),
            status: 'error',
            durationMs: Date.now() - start,
            code: 'WORKSPACE_STALE_PERMANENT',
          }
        }
        await context?.editCheckpoint?.recordPreEdit(path)
        await writeFile(path, nextContent, 'utf-8')
        await context?.workspaceMutation?.recordWrite(path)
        const warning = stale ? `[workspace warning] ${stale.description}\n` : ''
        const editDiff = buildUnifiedDiff(path, content, nextContent)
        const executionPosture = buildFileToolPosture(postureOptions, context?.cwd)
        return {
          output: `${warning}Replaced ${replacements} occurrence${replacements === 1 ? '' : 's'} in ${path}`,
          status: 'success',
          durationMs: Date.now() - start,
          ...(editDiff ? { metadata: { editDiff, editPath: path } } : {}),
          ...(executionPosture ? { executionPosture } : {}),
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return {
          output: message,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
    },
  }
}
