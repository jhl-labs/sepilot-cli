import { readFile, writeFile, unlink, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import { parseApplyPatch, type ParsedPatchFile, type PatchHunk } from './apply-patch-parser.js'
import { buildFileToolPosture, type FileToolPostureOptions } from './file-tool-posture.js'
import { resolveToolPath } from './path-utils.js'
import type { EditCheckpointHandle, ToolDefinitionRuntime, ToolResult } from './registry.js'

function workspaceStaleError(path: string): string {
  return `WORKSPACE_STALE_PERMANENT: ${path} changed on disk after your last read. Re-read the file and re-apply your edit against the current content.`
}

function patchContextRecovery(path: string): string {
  return [
    '[patch recovery]',
    `apply_patch could not match the current content in ${path}.`,
    'Read only the narrow line range around the intended change, then build a smaller patch from that exact current text.',
    'Do not retry the same patch and do not read the whole file unless the affected range is genuinely unknown.',
    '[/patch recovery]',
  ].join('\n')
}

export function createApplyPatchTool(postureOptions?: FileToolPostureOptions): ToolDefinitionRuntime {
  return {
    name: 'apply_patch',
    description:
      'Apply a multi-file patch in OpenAI apply-patch format. Supports Add/Update/Delete File. '
      + 'Wrap the patch in *** Begin Patch / *** End Patch. Add files use *** Add File: path followed by +lines. '
      + 'Updates use *** Update File: path and @@ hunks with space-prefixed context, -removed lines and +added lines. '
      + 'Every @@ hunk must contain an addition or removal: keep unchanged locator lines in the same hunk as the edit, '
      + 'rather than separate context-only hunks. Use exact current file content; read the affected range before retrying a mismatch. '
      + 'Hunks are matched by context; failures return a clear diagnostic.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: { patch: { type: 'string' } },
      required: ['patch'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      let parsed
      try {
        parsed = parseApplyPatch(String(input.patch ?? ''))
      } catch (e) {
        return {
          output: 'parse error: ' + (e as Error).message,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      if (parsed.files.length === 0) {
        return {
          output: 'parse error: patch did not contain any Add/Update/Delete File sections',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      const files = parsed.files.map((file: ParsedPatchFile) => ({
        ...file,
        path: resolveToolPath(file.path, context?.cwd),
      }))
      const touched: string[] = []
      const staleNotices: string[] = []
      try {
        for (const f of files) {
          if (f.op === 'update' || f.op === 'delete') {
            const stale = await context?.workspaceMutation?.detectStale(f.path)
            if (stale) {
              if (process.env.SEPILOTD_WORKSPACE_STALE_SOFT !== '1') {
                return {
                  output: workspaceStaleError(f.path),
                  status: 'error',
                  durationMs: Date.now() - start,
                  code: 'WORKSPACE_STALE_PERMANENT',
                }
              }
              staleNotices.push(stale.description)
            }
          }
        }
        for (const f of files) {
          await applyFile(f, touched, context?.editCheckpoint)
          await context?.workspaceMutation?.recordWrite(f.path)
        }
        const warning = staleNotices.length > 0
          ? `[workspace warning]\n${staleNotices.map((n) => `- ${n}`).join('\n')}\n`
          : ''
        const executionPosture = buildFileToolPosture(postureOptions, context?.cwd)
        return {
          output: warning + 'Applied patch to ' + touched.length + ' file(s): ' + touched.join(', '),
          status: 'success',
          durationMs: Date.now() - start,
          ...(executionPosture ? { executionPosture } : {}),
        }
      } catch (e) {
        const message = (e as Error).message
        const contextMiss = message.startsWith('context not found in ')
        const path = contextMiss ? message.slice('context not found in '.length) : ''
        return {
          output: contextMiss ? `${message}\n${patchContextRecovery(path)}` : message,
          status: 'error',
          durationMs: Date.now() - start,
          ...(contextMiss ? { code: 'PATCH_CONTEXT_MISMATCH_PERMANENT' } : {}),
        }
      }
    },
  }
}

async function applyFile(
  f: ParsedPatchFile,
  touched: string[],
  checkpoint: EditCheckpointHandle | undefined,
): Promise<void> {
  await checkpoint?.recordPreEdit(f.path)
  if (f.op === 'delete') {
    await unlink(f.path)
    touched.push(f.path)
    return
  }
  if (f.op === 'add') {
    await mkdir(dirname(f.path), { recursive: true })
    await writeFile(f.path, (f.addLines ?? []).join('\n') + '\n', 'utf8')
    touched.push(f.path)
    return
  }
  const current = await readFile(f.path, 'utf8')
  const hadTrailing = current.endsWith('\n')
  let lines = current.split('\n')
  if (hadTrailing) lines = lines.slice(0, -1)
  for (const h of f.hunks ?? []) lines = applyHunk(lines, h, f.path)
  const next = lines.join('\n') + (hadTrailing ? '\n' : '')
  if (next === current) {
    throw new Error(`patch produced no content change in ${f.path}`)
  }
  await writeFile(f.path, next, 'utf8')
  touched.push(f.path)
}

function applyHunk(lines: string[], h: PatchHunk, path: string): string[] {
  const orderedLines = h.lines.length > 0
    ? h.lines
    : [
        ...h.contextBefore.map((text) => ({ kind: 'context' as const, text })),
        ...h.remove.map((text) => ({ kind: 'remove' as const, text })),
        ...h.add.map((text) => ({ kind: 'add' as const, text })),
      ]
  const needle = orderedLines
    .filter((line) => line.kind !== 'add')
    .map((line) => line.text)
  const replacement = orderedLines
    .filter((line) => line.kind !== 'remove')
    .map((line) => line.text)
  if (needle.length === 0) {
    throw new Error('empty update hunk has no context in ' + path)
  }
  for (let i = 0; i <= lines.length - needle.length; i += 1) {
    let ok = true
    for (let j = 0; j < needle.length; j += 1) {
      if (lines[i + j] !== needle[j]) {
        ok = false
        break
      }
    }
    if (ok) {
      return [
        ...lines.slice(0, i),
        ...replacement,
        ...lines.slice(i + needle.length),
      ]
    }
  }
  throw new Error('context not found in ' + path)
}
