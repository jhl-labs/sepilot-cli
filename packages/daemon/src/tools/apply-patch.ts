import { readFile, writeFile, unlink, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import { parseApplyPatch, type ParsedPatchFile, type PatchHunk } from './apply-patch-parser.js'
import { buildFileToolPosture, type FileToolPostureOptions } from './file-tool-posture.js'
import { resolveToolPath } from './path-utils.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

interface PreparedPatchFile {
  file: ParsedPatchFile
  before: Buffer | null
  after: string | null
}

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
      + 'All file sections are checked before any writes. Hunks are matched by context; failures return a clear diagnostic. '
      + 'Filesystem commit failures may leave partial changes; inspect the reported applied paths before retrying.',
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
      let attemptedPath: string | undefined
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
        const prepared = await prepareFiles(files)
        for (const path of new Set(files.map((file) => file.path))) {
          await context?.editCheckpoint?.recordPreEdit(path)
        }
        for (const plan of prepared) {
          const f = plan.file
          const current = await readCurrentFile(f.path)
          if (current === null || plan.before === null
            ? current !== plan.before
            : !current.equals(plan.before)) {
            throw Object.assign(new Error(workspaceStaleError(f.path)), { code: 'WORKSPACE_STALE_PERMANENT' })
          }
          attemptedPath = f.path
          await applyPreparedFile(plan)
          touched.push(f.path)
          await context?.workspaceMutation?.recordWrite(f.path)
          attemptedPath = undefined
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
        const appliedPaths = [...new Set(touched)]
        const partial = appliedPaths.length > 0 || attemptedPath !== undefined
        const partialNotice = partial ? [
          '[partial patch]',
          `The patch failed during commit. Earlier changes were not rolled back. Applied paths: ${appliedPaths.slice(0, 20).join(', ') || '(none confirmed)'}.`,
          appliedPaths.length > 20 ? `${appliedPaths.length - 20} more applied paths are recorded in patchApplication metadata.` : '',
          attemptedPath ? `The attempted path ${attemptedPath} may also have changed; inspect its current bytes.` : '',
          'Read the affected files or inspect the diff before building a new patch. Do not replay already-applied edits.',
          '[/partial patch]',
        ].filter(Boolean).join('\n') : ''
        return {
          output: [contextMiss ? `${message}\n${patchContextRecovery(path)}` : message, partialNotice].filter(Boolean).join('\n'),
          status: 'error',
          durationMs: Date.now() - start,
          ...(contextMiss ? { code: 'PATCH_CONTEXT_MISMATCH_PERMANENT' } : {}),
          ...((e as NodeJS.ErrnoException).code === 'WORKSPACE_STALE_PERMANENT' ? { code: 'WORKSPACE_STALE_PERMANENT' } : {}),
          ...(partial ? { metadata: { patchApplication: { complete: false, appliedPaths, ...(attemptedPath ? { attemptedPath } : {}) } } } : {}),
        }
      }
    },
  }
}

async function readCurrentFile(path: string): Promise<Buffer | null> {
  try {
    return await readFile(path)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null
    throw error
  }
}

async function prepareFiles(files: ParsedPatchFile[]): Promise<PreparedPatchFile[]> {
  const seenPaths = new Set<string>()
  const prepared: PreparedPatchFile[] = []
  for (const file of files) {
    if (seenPaths.has(file.path)) throw new Error(`duplicate file section for ${file.path}`)
    seenPaths.add(file.path)
    const before = await readCurrentFile(file.path)
    if (file.op !== 'add' && before === null) {
      throw new Error(`ENOENT: no such file or directory, ${file.op} '${file.path}'`)
    }
    const after = file.op === 'delete' ? null
      : file.op === 'add' ? (file.addLines ?? []).join('\n') + '\n'
        : prepareUpdatedContent(file, before!.toString('utf8'))
    prepared.push({ file, before, after })
  }
  return prepared
}

async function applyPreparedFile({ file: f, after }: PreparedPatchFile): Promise<void> {
  if (f.op === 'delete') {
    await unlink(f.path)
    return
  }
  if (f.op === 'add') {
    await mkdir(dirname(f.path), { recursive: true })
  }
  await writeFile(f.path, after!, 'utf8')
}

function prepareUpdatedContent(f: ParsedPatchFile, current: string): string {
  const hadTrailing = current.endsWith('\n')
  let lines = current.split('\n')
  if (hadTrailing) lines = lines.slice(0, -1)
  for (const h of f.hunks ?? []) lines = applyHunk(lines, h, f.path)
  const next = lines.join('\n') + (hadTrailing ? '\n' : '')
  if (next === current) {
    throw new Error(`patch produced no content change in ${f.path}`)
  }
  return next
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
