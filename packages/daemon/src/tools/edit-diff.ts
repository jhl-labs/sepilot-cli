import { lstat, readFile } from 'node:fs/promises'
import { decodeFileContent } from './file-content.js'
import { createTwoFilesPatch } from 'diff'
import type { ToolCall } from '@sepilotd/core'
import { resolveToolPath } from './path-utils.js'

const MAX_DIFF_CHARS = 32_768

export function buildUnifiedDiff(path: string, before: string, after: string): string | undefined {
  if (before === after) return undefined
  const patch = createTwoFilesPatch(path, path, before, after, undefined, undefined, { context: 3 })
  if (patch.length > MAX_DIFF_CHARS) {
    return `${patch.slice(0, MAX_DIFF_CHARS)}\n[diff truncated at ${MAX_DIFF_CHARS} chars]`
  }
  return patch
}

export interface ApprovalPreviewOptions {
  canReadPath?: (path: string) => boolean
}

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
        const content = decodeFileContent(args)
        if (!path || content === undefined || !readGuarded(path)) return undefined
        if (args.createOnly === true) {
          try { await lstat(path); return undefined }
          catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') return undefined }
        }
        const before = await readFile(path, 'utf-8').catch(() => '')
        return buildUnifiedDiff(path, before, content)
      }
      case 'fs.append': {
        const path = typeof args.path === 'string' ? resolveToolPath(args.path, cwd) : ''
        const content = decodeFileContent(args)
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
        const matchCount = before.split(oldText).length - 1
        if (args.replaceAll !== true && matchCount > 1) return undefined
        const after = before.split(oldText).join(newText)
        return buildUnifiedDiff(path, before, after)
      }
      case 'apply_patch': {
        return typeof args.patch === 'string' ? args.patch : undefined
      }
      default:
        return undefined
    }
  } catch {
    return undefined
  }
}
