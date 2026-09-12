import { readFile } from 'node:fs/promises'
import { isAbsolute, resolve } from 'node:path'
import { strictWorkspacePathViolation } from '../../security/policy-engine.js'
import type { CustomCommand } from './commands.js'
import { expandCommandTemplate } from '../slash-runtime.js'

export interface SlashExpandDeps {
  commands: readonly CustomCommand[]
  cwd?: string
  workspaceRoot?: string
}

export interface SlashExpandResult {
  matched: CustomCommand
  expanded: string
}

export class SlashWorkspaceBoundaryError extends Error {
  readonly code = 'WORKSPACE_BOUNDARY'

  constructor() {
    super('Slash command file references must stay inside the selected workspace')
    this.name = 'SlashWorkspaceBoundaryError'
  }
}

const SLASH_RE = /^\/([a-zA-Z][a-zA-Z0-9_\-]*)(?:\s+(.*))?$/s

export async function tryExpandSlashInput(
  input: string,
  deps: SlashExpandDeps,
): Promise<SlashExpandResult | null> {
  const match = SLASH_RE.exec(input.trim())
  if (!match) return null
  const [, id, argsRaw] = match
  const command = deps.commands?.find((c) => c.id === id)
  if (!command) return null
  const args = (argsRaw ?? '').trim()
  const cwd = deps.cwd ?? process.cwd()
  const expanded = await expandCommandTemplate(command.template, {
    args,
    runCommand: async () => {
      throw new Error('command-runner substitution is disabled for slash commands')
    },
    readFile: async (path) => {
      const full = isAbsolute(path) ? path : resolve(cwd, path)
      if (
        deps.workspaceRoot
        && strictWorkspacePathViolation(full, deps.workspaceRoot)
      ) {
        throw new SlashWorkspaceBoundaryError()
      }
      return (await readFile(full, 'utf8')).trim()
    },
  })
  return { matched: command, expanded }
}
