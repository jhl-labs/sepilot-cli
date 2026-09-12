import { spawn } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

export interface ExternalEditorOptions {
  env?: NodeJS.ProcessEnv
  setRawMode?: (enabled: boolean) => void
}

export function parseEditorCommand(value: string): string[] {
  const parts: string[] = []
  let current = ''
  let quote: '"' | "'" | null = null
  let escaped = false
  for (const character of value.trim()) {
    if (escaped) { current += character; escaped = false; continue }
    if (character === '\\' && quote !== "'") { escaped = true; continue }
    if (quote) {
      if (character === quote) quote = null
      else current += character
      continue
    }
    if (character === '"' || character === "'") { quote = character; continue }
    if (/\s/.test(character)) {
      if (current) { parts.push(current); current = '' }
      continue
    }
    current += character
  }
  if (escaped || quote) throw new Error(`Invalid editor command: ${value}`)
  if (current) parts.push(current)
  return parts
}

export async function editTextInExternalEditor(
  initialText: string,
  options: ExternalEditorOptions = {},
): Promise<string> {
  const env = options.env ?? process.env
  const editor = env.VISUAL?.trim() || env.EDITOR?.trim()
  if (!editor) throw new Error('Set $VISUAL or $EDITOR to use /editor.')
  const [command, ...editorArgs] = parseEditorCommand(editor)
  if (!command) throw new Error('Set $VISUAL or $EDITOR to use /editor.')

  const directory = await mkdtemp(join(tmpdir(), 'sepilot-editor-'))
  const path = join(directory, 'prompt.md')
  try {
    await writeFile(path, initialText, { encoding: 'utf8', mode: 0o600 })
    options.setRawMode?.(false)
    const exitCode = await new Promise<number | null>((resolve, reject) => {
      const child = spawn(command, [...editorArgs, path], { env, stdio: 'inherit' })
      child.once('error', reject)
      child.once('exit', resolve)
    })
    if (exitCode !== 0) throw new Error(`${command} exited with code ${exitCode ?? 'unknown'}.`)
    return await readFile(path, 'utf8')
  } finally {
    options.setRawMode?.(true)
    await rm(directory, { recursive: true, force: true })
  }
}
