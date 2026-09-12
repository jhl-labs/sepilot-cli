import { execFile } from 'node:child_process'
import { mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const MAX_CLIPBOARD_IMAGE_BYTES = 25 * 1024 * 1024
const POWERSHELL_CLIPBOARD_PNG = [
  'Add-Type -AssemblyName System.Windows.Forms',
  'Add-Type -AssemblyName System.Drawing',
  '$image = [Windows.Forms.Clipboard]::GetImage()',
  'if ($null -eq $image) { exit 2 }',
  '$stream = New-Object IO.MemoryStream',
  '$image.Save($stream, [Drawing.Imaging.ImageFormat]::Png)',
  '$bytes = $stream.ToArray()',
  '[Console]::OpenStandardOutput().Write($bytes, 0, $bytes.Length)',
].join('; ')

export interface ClipboardImageDeps {
  run(command: string, args: string[]): Promise<Buffer>
  makeTempDir(prefix: string): Promise<string>
  write(path: string, data: Buffer): Promise<void>
}

function runBinary(command: string, args: string[]): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    execFile(command, args, { encoding: 'buffer', maxBuffer: MAX_CLIPBOARD_IMAGE_BYTES }, (error, stdout) => {
      if (error) reject(error)
      else resolve(Buffer.isBuffer(stdout) ? stdout : Buffer.from(stdout))
    })
  })
}

const NODE_DEPS: ClipboardImageDeps = {
  run: runBinary,
  makeTempDir: (prefix) => mkdtemp(prefix),
  write: (path, data) => writeFile(path, data),
}

export async function pasteClipboardPng(deps: ClipboardImageDeps = NODE_DEPS): Promise<string> {
  const backends = [
    { command: 'wl-paste', args: ['--no-newline', '--type', 'image/png'] },
    { command: 'xclip', args: ['-selection', 'clipboard', '-t', 'image/png', '-o'] },
    { command: 'pngpaste', args: ['-'] },
    { command: 'powershell.exe', args: ['-NoProfile', '-NonInteractive', '-Command', POWERSHELL_CLIPBOARD_PNG] },
  ]
  const failures: string[] = []
  for (const backend of backends) {
    try {
      const data = await deps.run(backend.command, backend.args)
      if (data.length === 0) throw new Error('clipboard returned no image data')
      if (data.length > MAX_CLIPBOARD_IMAGE_BYTES) throw new Error('clipboard image exceeds 25 MiB')
      const directory = await deps.makeTempDir(join(tmpdir(), 'sepilotd-clipboard-'))
      const path = join(directory, 'clipboard.png')
      await deps.write(path, data)
      return path
    } catch (error) {
      failures.push(`${backend.command}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  throw new Error(`No PNG image could be read from the clipboard. ${failures.join(' · ')}`)
}
