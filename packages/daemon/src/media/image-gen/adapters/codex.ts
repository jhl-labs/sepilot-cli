import { spawn } from 'node:child_process'
import { lstat, mkdtemp, readFile, readdir, realpath, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { extname, join, relative, resolve, sep } from 'node:path'
import type { Provider } from '../adapter.js'

const DEFAULT_TIMEOUT_MS = 15 * 60_000
const MAX_TIMEOUT_MS = 60 * 60_000
const MAX_OUTPUT_BYTES = 25 * 1024 * 1024
const MAX_PROCESS_OUTPUT_BYTES = 2 * 1024 * 1024
const MAX_COUNT = 8

interface CodexImageGenOptions {
  command?: string
  env?: NodeJS.ProcessEnv
  runProcess?: typeof runProcess
  createWorkspace?: () => Promise<string>
  removeWorkspace?: (path: string) => Promise<void>
}

interface CodexImageParams {
  operation: 'text-to-image'
  width?: number
  height?: number
  count: number
  negativePrompt?: string
  timeoutMs: number
}

interface ProcessInput {
  cwd: string
  env: NodeJS.ProcessEnv
  stdin: string
  signal?: AbortSignal
  timeoutMs: number
}

interface ProcessResult {
  stdout: string
  stderr: string
}

function asObject(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

function finiteNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return undefined
}

function parseParams(raw: unknown): CodexImageParams {
  const params = asObject(raw)
  const operation = typeof params.operation === 'string' ? params.operation : 'text-to-image'
  if (operation !== 'text-to-image') {
    throw new Error('Codex image generation currently supports text-to-image only')
  }
  const count = Math.floor(finiteNumber(params.count) ?? finiteNumber(params.batchSize) ?? 1)
  if (count < 1 || count > MAX_COUNT) {
    throw new Error(`params.count must be between 1 and ${MAX_COUNT}`)
  }
  const timeoutMs = Math.floor(finiteNumber(params.timeoutMs) ?? DEFAULT_TIMEOUT_MS)
  if (timeoutMs < 1 || timeoutMs > MAX_TIMEOUT_MS) {
    throw new Error(`params.timeoutMs must be between 1 and ${MAX_TIMEOUT_MS}`)
  }
  const width = finiteNumber(params.width)
  const height = finiteNumber(params.height)
  const negativePrompt =
    typeof params.negativePrompt === 'string' && params.negativePrompt.trim()
      ? params.negativePrompt.trim()
      : undefined
  return {
    operation,
    count,
    timeoutMs,
    ...(width && width > 0 ? { width: Math.floor(width) } : {}),
    ...(height && height > 0 ? { height: Math.floor(height) } : {}),
    ...(negativePrompt ? { negativePrompt } : {}),
  }
}

function appendBounded(current: string, chunk: Buffer): string {
  const next = current + chunk.toString('utf8')
  return next.length <= MAX_PROCESS_OUTPUT_BYTES
    ? next
    : next.slice(next.length - MAX_PROCESS_OUTPUT_BYTES)
}

async function runProcess(
  command: string,
  args: string[],
  input: ProcessInput,
): Promise<ProcessResult> {
  input.signal?.throwIfAborted()
  return await new Promise<ProcessResult>((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd: input.cwd,
      env: input.env,
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    let settled = false
    const finish = (error?: Error) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      input.signal?.removeEventListener('abort', abort)
      if (error) reject(error)
      else resolvePromise({ stdout, stderr })
    }
    const abort = () => {
      child.kill('SIGTERM')
      finish(new Error('Codex image generation cancelled'))
    }
    const timer = setTimeout(() => {
      child.kill('SIGTERM')
      finish(new Error(`Codex image generation timed out after ${input.timeoutMs}ms`))
    }, input.timeoutMs)
    input.signal?.addEventListener('abort', abort, { once: true })
    child.stdout.on('data', (chunk: Buffer) => {
      stdout = appendBounded(stdout, chunk)
    })
    child.stderr.on('data', (chunk: Buffer) => {
      stderr = appendBounded(stderr, chunk)
    })
    child.once('error', (error) => finish(error))
    child.once('close', (code, signal) => {
      if (code === 0) finish()
      else {
        const detail = stderr.trim() || stdout.trim() || `signal ${signal ?? 'unknown'}`
        finish(new Error(`Codex image generation failed (exit ${code ?? 'unknown'}): ${detail}`))
      }
    })
    child.stdin.once('error', (error) => finish(error))
    child.stdin.end(input.stdin)
  })
}

function generationPrompt(prompt: string, outputName: string, params: CodexImageParams): string {
  const size = params.width && params.height
    ? `${params.width}x${params.height} target aspect ratio and resolution when supported.`
    : 'Use the image generator default size.'
  return [
    '$imagegen',
    'Generate exactly one new raster image with the built-in image generation tool.',
    `Primary request: ${prompt}`,
    `Output sizing: ${size}`,
    params.negativePrompt ? `Avoid: ${params.negativePrompt}` : '',
    'Do not use the OpenAI API or the fallback image generation script.',
    `After generation succeeds, copy the final generated image to the current workspace as exactly ${outputName}.`,
    'The built-in tool may initially save under CODEX_HOME/generated_images; copying that result is expected.',
    'Do not create any other image files and do not modify files outside the current workspace.',
  ].filter(Boolean).join('\n')
}

function mimeFromBytes(bytes: Buffer): string | null {
  if (bytes.length >= 8 && bytes.subarray(0, 8).equals(Buffer.from('89504e470d0a1a0a', 'hex'))) {
    return 'image/png'
  }
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return 'image/jpeg'
  }
  if (
    bytes.length >= 12 &&
    bytes.subarray(0, 4).toString('ascii') === 'RIFF' &&
    bytes.subarray(8, 12).toString('ascii') === 'WEBP'
  ) {
    return 'image/webp'
  }
  return null
}

async function imageFiles(root: string): Promise<string[]> {
  const canonicalRoot = await realpath(root)
  const found: string[] = []
  async function visit(directory: string): Promise<void> {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const candidate = join(directory, entry.name)
      if (entry.isSymbolicLink()) continue
      if (entry.isDirectory()) {
        await visit(candidate)
        continue
      }
      if (!entry.isFile() || !['.png', '.jpg', '.jpeg', '.webp'].includes(extname(entry.name).toLowerCase())) {
        continue
      }
      const canonical = await realpath(candidate)
      const rel = relative(canonicalRoot, canonical)
      if (!rel || rel === '..' || rel.startsWith(`..${sep}`) || resolve(canonical) === canonicalRoot) {
        continue
      }
      const info = await lstat(canonical)
      if (!info.isFile() || info.size < 1 || info.size > MAX_OUTPUT_BYTES) continue
      found.push(canonical)
    }
  }
  await visit(canonicalRoot)
  return found
}

export function createCodexImageGenProvider(options: CodexImageGenOptions = {}): Provider {
  const env = options.env ?? process.env
  const command = options.command ?? (env.SEPILOTD_CODEX_COMMAND?.trim() || 'codex')
  const execute = options.runProcess ?? runProcess
  const createWorkspace = options.createWorkspace ?? (() => mkdtemp(join(tmpdir(), 'sepilot-codex-image-')))
  const removeWorkspace = options.removeWorkspace ?? ((path) => rm(path, { recursive: true, force: true, maxRetries: 3 }))

  return {
    info: {
      id: 'codex',
      label: 'Codex',
      enabled: true,
      operations: ['text-to-image'],
      recommendedModels: [{
        id: 'codex-integrated',
        label: 'Codex integrated image generation',
        operation: 'text-to-image',
        modelId: 'default',
        notes: 'Uses the current Codex login and built-in image generation; no API key is required.',
        tags: ['Codex login', 'built-in'],
      }],
    },
    async run(input) {
      const params = parseParams(input.params)
      const workspace = await createWorkspace()
      const outputs: Awaited<ReturnType<Provider['run']>>['outputs'] = []
      const usedPaths = new Set<string>()
      try {
        for (let index = 0; index < params.count; index += 1) {
          input.signal?.throwIfAborted()
          const outputName = `generated-${index + 1}.png`
          input.onProgress(index / params.count)
          await execute(
            command,
            [
              'exec',
              '--skip-git-repo-check',
              '--ephemeral',
              '--sandbox',
              'workspace-write',
              '-c',
              'approval_policy="never"',
              '-c',
              'sandbox_workspace_write.exclude_tmpdir_env_var=true',
              '-c',
              'sandbox_workspace_write.exclude_slash_tmp=true',
              '--ignore-rules',
              '--color',
              'never',
              '-C',
              workspace,
              '-',
            ],
            {
              cwd: workspace,
              env,
              stdin: generationPrompt(input.prompt, outputName, params),
              signal: input.signal,
              timeoutMs: params.timeoutMs,
            },
          )
          const candidates = await imageFiles(workspace)
          const unused = candidates.filter((path) => !usedPaths.has(path))
          const expected = unused.find((path) => path.endsWith(outputName)) ?? unused[0]
          if (!expected) {
            throw new Error('Codex completed without copying a generated image into the isolated workspace')
          }
          const fileInfo = await stat(expected)
          if (!fileInfo.isFile() || fileInfo.size > MAX_OUTPUT_BYTES) {
            throw new Error(`Codex generated image exceeds the ${MAX_OUTPUT_BYTES}-byte limit`)
          }
          const bytes = await readFile(expected)
          const mime = mimeFromBytes(bytes)
          if (!mime) throw new Error('Codex output is not a supported PNG, JPEG, or WebP image')
          outputs.push({
            id: `${input.jobId}-${index}`,
            mime,
            bytes,
            kind: 'image',
          })
          usedPaths.add(expected)
          input.onProgress((index + 1) / params.count)
        }
        return { outputs }
      } finally {
        await removeWorkspace(workspace)
      }
    },
  }
}

export const codexImageGenProvider = createCodexImageGenProvider()
