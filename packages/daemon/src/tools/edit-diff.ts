import { readFile } from 'node:fs/promises'
import { createTwoFilesPatch } from 'diff'
import type { ToolCall } from '@sepilotd/core'
import { resolveToolPath } from './path-utils.js'

const MAX_DIFF_CHARS = 32_768

/**
 * Unified diff for a single file edit, attached to tool results
 * (`metadata.editDiff`) and approval requests (`previewDiff`) so surfaces
 * can show *what changes* instead of a byte-count summary. Returns
 * undefined for identical content; oversized diffs are truncated with a
 * marker so a huge generated file cannot bloat session events.
 */
export function buildUnifiedDiff(path: string, before: string, after: string): string | undefined {
  if (before === after) return undefined
  const patch = createTwoFilesPatch(path, path, before, after, undefined, undefined, { context: 3 })
  // createTwoFilesPatch prepends an "Index:"-style header pair; keep it —
  // surfaces detect unified diffs by the ---/+++/@@ structure.
  if (patch.length > MAX_DIFF_CHARS) {
    return `${patch.slice(0, MAX_DIFF_CHARS)}\n[diff truncated at ${MAX_DIFF_CHARS} chars]`
  }
  return patch
}

export interface ApprovalPreviewOptions {
  /**
   * Read-policy gate evaluated before the preview reads the target file.
   * The preview runs pre-consent and its output is broadcast to every
   * connected surface, so a path the operator denied for fs.read must
   * not have its current content leaked through a write-approval diff.
   */
  canReadPath?: (path: string) => boolean
}

/**
 * Read-only preview of what a file-edit tool call would change, computed
 * before the user approves it. Returns undefined when the tool is not a
 * file editor, arguments are malformed, or the edit would not apply —
 * the approval flow then simply shows no diff, never an error.
 */
export async function buildApprovalPreviewDiff(
  toolCall: ToolCall,
  cwd?: string,
  options: ApprovalPreviewOptions = {},
): Promise<string | undefined> {
  try {
    const args = toolCall.arguments ?? {}
    const readGuarded = (path: string): boolean =>
      options.canReadPath ? options.canReadPath(path) : true
    switch (toolCall.name) {
      case 'fs.write': {
        const path = typeof args.path === 'string' ? resolveToolPath(args.path, cwd) : ''
        const content = typeof args.content === 'string' ? args.content : undefined
        if (!path || content === undefined || !readGuarded(path)) return undefined
        const before = await readFile(path, 'utf-8').catch(() => '')
        return buildUnifiedDiff(path, before, content)
      }
      case 'fs.append': {
        const path = typeof args.path === 'string' ? resolveToolPath(args.path, cwd) : ''
        const content = typeof args.content === 'string' ? args.content : undefined
        if (!path || content === undefined || !readGuarded(path)) return undefined
        const before = await readFile(path, 'utf-8').catch(() => '')
        return buildUnifiedDiff(path, before, before + content)
      }
      case 'fs.edit': {
        const path = typeof args.path === 'string' ? resolveToolPath(args.path, cwd) : ''
        const oldText = typeof args.oldText === 'string' ? args.oldText : ''
        const newText = typeof args.newText === 'string' ? args.newText : ''
        if (!path || !oldText || !readGuarded(path)) return undefined
        const before = await readFile(path, 'utf-8').catch(() => undefined)
        if (before === undefined || !before.includes(oldText)) return undefined
        // Mirror the execution semantics exactly: a multi-match edit
        // without replaceAll will be rejected by fs.edit, so previewing
        // a first-match-only diff would invite approval of a change
        // that never applies. split/join keeps $-patterns literal.
        const matchCount = before.split(oldText).length - 1
        if (args.replaceAll !== true && matchCount > 1) return undefined
        const after = before.split(oldText).join(newText)
        return buildUnifiedDiff(path, before, after)
      }
      case 'apply_patch': {
        // The patch argument already is the diff.
        return typeof args.patch === 'string' ? args.patch : undefined
      }
      default:
        return undefined
    }
  } catch {
    return undefined
  }
}
