export type PatchOp = 'add' | 'delete' | 'update'

export type PatchHunkLineKind = 'context' | 'remove' | 'add'

export interface PatchHunkLine {
  kind: PatchHunkLineKind
  text: string
}

export interface PatchHunk {
  contextBefore: string[]
  remove: string[]
  add: string[]
  lines: PatchHunkLine[]
}

export interface ParsedPatchFile {
  op: PatchOp
  path: string
  hunks?: PatchHunk[]
  addLines?: string[]
}

export interface ParsedPatch {
  files: ParsedPatchFile[]
}

export function parseApplyPatch(raw: string): ParsedPatch {
  const lines = raw.replace(/\r\n/g, '\n').split('\n')
  if (!lines.includes('*** Begin Patch')) throw new Error('missing *** Begin Patch')
  if (!lines.includes('*** End Patch')) throw new Error('missing *** End Patch')
  const files: ParsedPatchFile[] = []
  const seenPaths = new Set<string>()
  const requirePath = (path: string, directive: string): string => {
    if (!path) throw new Error(`${directive} requires a non-empty path`)
    if (seenPaths.has(path)) throw new Error(`duplicate file section for ${path}`)
    seenPaths.add(path)
    return path
  }
  let i = lines.indexOf('*** Begin Patch') + 1
  while (i < lines.length && lines[i] !== '*** End Patch') {
    const line = lines[i]
    if (line.trim() === '') {
      i += 1
      continue
    }
    if (line.startsWith('*** Add File: ')) {
      const path = requirePath(line.slice('*** Add File: '.length).trim(), 'Add File')
      const addLines: string[] = []
      i += 1
      while (i < lines.length && lines[i].startsWith('+')) {
        addLines.push(lines[i].slice(1))
        i += 1
      }
      files.push({ op: 'add', path, addLines })
    } else if (line.startsWith('*** Delete File: ')) {
      const path = requirePath(line.slice('*** Delete File: '.length).trim(), 'Delete File')
      files.push({ op: 'delete', path })
      i += 1
    } else if (line.startsWith('*** Update File: ')) {
      const path = requirePath(line.slice('*** Update File: '.length).trim(), 'Update File')
      const hunks: PatchHunk[] = []
      i += 1
      while (i < lines.length && !lines[i].startsWith('***')) {
        if (lines[i].startsWith('@@')) {
          i += 1
          const ctx: string[] = []
          const rem: string[] = []
          const add: string[] = []
          const hunkLines: PatchHunkLine[] = []
          let sawChange = false
          while (
            i < lines.length
            && !lines[i].startsWith('@@')
            && !lines[i].startsWith('***')
          ) {
            const l = lines[i]
            if (l.startsWith(' ')) {
              const text = l.slice(1)
              if (!sawChange) ctx.push(text)
              hunkLines.push({ kind: 'context', text })
            } else if (l.startsWith('-')) {
              const text = l.slice(1)
              rem.push(text)
              hunkLines.push({ kind: 'remove', text })
              sawChange = true
            } else if (l.startsWith('+')) {
              const text = l.slice(1)
              add.push(text)
              hunkLines.push({ kind: 'add', text })
              sawChange = true
            } else {
              throw new Error(`invalid update hunk line for ${path}: ${l || '(blank line)'}`)
            }
            i += 1
          }
          if (!sawChange) throw new Error(`update hunk for ${path} contains no changes`)
          hunks.push({ contextBefore: ctx, remove: rem, add, lines: hunkLines })
        } else {
          throw new Error(`unexpected patch content for ${path}: ${lines[i] || '(blank line)'}`)
        }
      }
      if (hunks.length === 0) throw new Error(`Update File ${path} requires at least one @@ hunk`)
      files.push({ op: 'update', path, hunks })
    } else {
      throw new Error(
        `unexpected patch directive: ${line}; expected an Add/Update/Delete File section with a path`,
      )
    }
  }
  return { files }
}
