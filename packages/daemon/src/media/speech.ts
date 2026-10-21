import { execFile, spawn } from 'node:child_process'
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, extname, join } from 'node:path'
import { promisify } from 'node:util'

interface ExecFileResult {
  stdout: string
  stderr: string
}

type ExecFileFn = (
  file: string,
  args: string[],
  options?: { maxBuffer?: number; timeout?: number; cwd?: string },
) => Promise<ExecFileResult>

type RunWithStdinFn = (
  file: string,
  args: string[],
  stdin: string,
  options?: { timeout?: number },
) => Promise<void>

export interface SpeechToolDeps {
  execFile?: ExecFileFn
  runWithStdin?: RunWithStdinFn
  env?: NodeJS.ProcessEnv
}

const execFileAsync = promisify(execFile)

function defaultRunWithStdin(
  file: string,
  args: string[],
  stdin: string,
  options?: { timeout?: number },
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const child = spawn(file, args, { stdio: ['pipe', 'ignore', 'pipe'] })
    let stderr = ''
    const timer = options?.timeout
      ? setTimeout(() => {
          child.kill('SIGKILL')
          reject(new Error(`${file} timed out after ${options.timeout}ms`))
        }, options.timeout)
      : null
    child.stderr?.on('data', (chunk) => {
      stderr += String(chunk)
      if (stderr.length > 64 * 1024) stderr = stderr.slice(-64 * 1024)
    })
    child.on('error', (err) => {
      if (timer) clearTimeout(timer)
      reject(err)
    })
    child.on('close', (code) => {
      if (timer) clearTimeout(timer)
      if (code === 0) resolve()
      else
        reject(new Error(`${file} exited with code ${code}${stderr ? `: ${stderr.trim()}` : ''}`))
    })
    child.stdin?.end(stdin, 'utf-8')
  })
}

function defaultDeps(deps?: SpeechToolDeps): Required<SpeechToolDeps> {
  return {
    execFile:
      deps?.execFile ??
      ((file, args, options) => execFileAsync(file, args, options) as Promise<ExecFileResult>),
    runWithStdin: deps?.runWithStdin ?? defaultRunWithStdin,
    env: deps?.env ?? process.env,
  }
}

class SpeechSemaphore {
  private max: number
  private active = 0
  private queue: Array<() => void> = []

  constructor(max: number) {
    this.max = Math.max(1, Math.floor(max))
  }

  setMax(max: number): void {
    this.max = Math.max(1, Math.floor(max))
    this.drain()
  }

  private drain(): void {
    while (this.active < this.max && this.queue.length > 0) {
      const next = this.queue.shift()
      if (next) {
        this.active += 1
        next()
      }
    }
  }

  async run<T>(fn: () => Promise<T>): Promise<T> {
    if (this.active >= this.max) {
      await new Promise<void>((resolve) => this.queue.push(resolve))
    } else {
      this.active += 1
    }
    try {
      return await fn()
    } finally {
      this.active -= 1
      this.drain()
    }
  }
}

const speechSemaphore = new SpeechSemaphore(
  Number(process.env.SEPILOTD_VOICE_MAX_CONCURRENT) > 0
    ? Number(process.env.SEPILOTD_VOICE_MAX_CONCURRENT)
    : 2,
)

export function configureSpeechConcurrency(max: number): void {
  speechSemaphore.setMax(max)
}

export function runSpeechJob<T>(fn: () => Promise<T>): Promise<T> {
  return speechSemaphore.run(fn)
}

export class SpeechBinaryMissingError extends Error {
  readonly code = 'SPEECH_BINARY_MISSING'
  constructor(message: string) {
    super(message)
    this.name = 'SpeechBinaryMissingError'
  }
}

export function isSpeechBinaryMissingError(error: unknown): error is SpeechBinaryMissingError {
  return (
    error instanceof SpeechBinaryMissingError ||
    (typeof error === 'object' &&
      error !== null &&
      (error as { code?: unknown }).code === 'SPEECH_BINARY_MISSING')
  )
}

function isMissingBinary(error: unknown): boolean {
  return (error as { code?: unknown })?.code === 'ENOENT'
}

export interface TranscribeOptions {
  audioPath: string
  binaryPath?: string
  language?: string
  model?: string
}

export interface TranscribeResult {
  text: string
  model: string
  language?: string
}

export async function transcribeAudio(
  options: TranscribeOptions,
  deps?: SpeechToolDeps,
): Promise<TranscribeResult> {
  const { execFile: run, env } = defaultDeps(deps)
  const audioPath = options.audioPath.trim()
  if (!audioPath) {
    throw new Error('audioPath is required')
  }

  const bin = options.binaryPath?.trim() || env.SEPILOTD_WHISPER_BIN?.trim() || 'whisper'
  const model = options.model?.trim() || env.SEPILOTD_WHISPER_MODEL?.trim() || 'base'
  const language = options.language?.trim() || env.SEPILOTD_WHISPER_LANG?.trim() || undefined

  const outDir = await mkdtemp(join(tmpdir(), 'sepilotd-stt-'))
  try {
    const args = [
      audioPath,
      '--model',
      model,
      '--output_format',
      'all',
      '--output_dir',
      outDir,
      '--verbose',
      'False',
    ]
    if (language) {
      args.push('--language', language)
    }

    try {
      await run(bin, args, { maxBuffer: 16 * 1024 * 1024, timeout: 10 * 60_000 })
    } catch (error) {
      if (isMissingBinary(error)) {
        throw new SpeechBinaryMissingError(
          `Transcription requires the "${bin}" binary (pip install openai-whisper). ` +
            'Set voice.transcription.whisperBin in settings or SEPILOTD_WHISPER_BIN to its path, ' +
            'or install it on this host. No external API is used.',
        )
      }
      throw error
    }

    const stem = basename(audioPath, extname(audioPath))
    const txtPath = join(outDir, `${stem}.txt`)
    const raw = await readFile(txtPath, 'utf-8').catch(() => {
      throw new Error(`whisper produced no transcript at ${txtPath}`)
    })
    let detectedLanguage = language
    try {
      const jsonRaw = await readFile(join(outDir, `${stem}.json`), 'utf-8')
      const parsed = JSON.parse(jsonRaw) as { language?: unknown }
      if (typeof parsed.language === 'string' && parsed.language.trim()) {
        detectedLanguage = normalizeSpeechLanguage(parsed.language.trim()) ?? parsed.language.trim()
      }
    } catch {
    }
    return { text: raw.trim(), model, language: detectedLanguage }
  } finally {
    await rm(outDir, { recursive: true, force: true }).catch(() => {})
  }
}

export interface SynthesizeOptions {
  text: string
  outputPath?: string
  binaryPath?: string
  modelPath?: string
  language?: string
}

export interface SynthesizeResult {
  audioPath: string
  temporary: boolean
  bytes: number
}

export function normalizeSpeechLanguage(value: string | undefined): string | undefined {
  const trimmed = value?.trim()
  if (!trimmed) return undefined
  const normalized = trimmed.toLowerCase().replace(/_/g, '-')
  if (normalized === 'auto' || normalized === 'detect' || normalized === 'same') return undefined
  return normalized
}

function piperModelEnvKey(language: string): string {
  return `SEPILOTD_PIPER_MODEL_${language.toUpperCase().replace(/[^A-Z0-9]/g, '_')}`
}

export function configuredPiperLanguages(env: NodeJS.ProcessEnv = process.env): string[] {
  const languages = new Set<string>()
  const defaultLanguage = normalizeSpeechLanguage(env.SEPILOTD_PIPER_LANG)
  if (defaultLanguage && env.SEPILOTD_PIPER_MODEL?.trim()) {
    languages.add(defaultLanguage)
  }
  for (const [key, value] of Object.entries(env)) {
    if (!value?.trim() || !key.startsWith('SEPILOTD_PIPER_MODEL_')) continue
    const suffix = key.slice('SEPILOTD_PIPER_MODEL_'.length)
    if (!suffix) continue
    languages.add(suffix.toLowerCase().replace(/_/g, '-'))
  }
  return [...languages].sort()
}

function resolvePiperModel(
  env: NodeJS.ProcessEnv,
  language: string | undefined,
): string | undefined {
  const defaultModel = env.SEPILOTD_PIPER_MODEL?.trim()
  if (!language) return defaultModel

  const exactModel = env[piperModelEnvKey(language)]?.trim()
  if (exactModel) return exactModel

  const baseLanguage = language.split('-')[0]
  const baseModel =
    baseLanguage === language ? undefined : env[piperModelEnvKey(baseLanguage)]?.trim()
  if (baseModel) return baseModel

  const defaultLanguage = normalizeSpeechLanguage(env.SEPILOTD_PIPER_LANG)
  if (
    defaultModel &&
    defaultLanguage &&
    (defaultLanguage === language || defaultLanguage === baseLanguage)
  ) {
    return defaultModel
  }
  return undefined
}

export async function synthesizeSpeech(
  options: SynthesizeOptions,
  deps?: SpeechToolDeps,
): Promise<SynthesizeResult> {
  const { runWithStdin: run, env } = defaultDeps(deps)
  const text = options.text.trim()
  if (!text) {
    throw new Error('text is required')
  }

  const bin = options.binaryPath?.trim() || env.SEPILOTD_PIPER_BIN?.trim() || 'piper'
  const language = normalizeSpeechLanguage(options.language)
  const modelPath = options.modelPath?.trim() || resolvePiperModel(env, language)
  if (!modelPath) {
    const languageMessage = language
      ? ` for language "${language}". Set ${piperModelEnvKey(language)} or set SEPILOTD_PIPER_LANG=${language} with SEPILOTD_PIPER_MODEL.`
      : '.'
    throw new SpeechBinaryMissingError(
      `Text-to-speech requires a Piper .onnx voice model${languageMessage} ` +
        'Download one from https://github.com/rhasspy/piper/releases and set voice.synthesis.model in settings or SEPILOTD_PIPER_MODEL. No external API is used.',
    )
  }

  let outputPath = options.outputPath?.trim() || ''
  let scratchDir: string | undefined
  let temporary = false
  if (!outputPath) {
    scratchDir = await mkdtemp(join(tmpdir(), 'sepilotd-tts-'))
    outputPath = join(scratchDir, 'speech.wav')
    temporary = true
  }

  try {
    try {
      await run(bin, ['--model', modelPath, '--output_file', outputPath], text, {
        timeout: 5 * 60_000,
      })
    } catch (error) {
      if (isMissingBinary(error)) {
        throw new SpeechBinaryMissingError(
          `Text-to-speech requires the "${bin}" binary (https://github.com/rhasspy/piper). ` +
            'Set voice.synthesis.piperBin in settings or SEPILOTD_PIPER_BIN to its path, or install it on this host. No external API is used.',
        )
      }
      throw error
    }
    const info = await stat(outputPath).catch(() => {
      throw new Error(`piper produced no audio at ${outputPath}`)
    })
    return { audioPath: outputPath, temporary, bytes: info.size }
  } catch (error) {
    if (scratchDir) await rm(scratchDir, { recursive: true, force: true }).catch(() => {})
    throw error
  }
}

function findRiffChunk(buf: Buffer, id: string): { offset: number; size: number } | null {
  let pos = 12
  while (pos + 8 <= buf.length) {
    const chunkId = buf.toString('ascii', pos, pos + 4)
    const size = buf.readUInt32LE(pos + 4)
    if (chunkId === id) return { offset: pos + 8, size }
    pos += 8 + size + (size % 2)
  }
  return null
}

export async function concatenateWavFiles(inputPaths: string[], outputPath: string): Promise<number> {
  if (inputPaths.length === 0) throw new Error('concatenateWavFiles: no input files')
  const buffers = await Promise.all(inputPaths.map((p) => readFile(p)))
  const first = buffers[0]
  const fmt = findRiffChunk(first, 'fmt ')
  if (!fmt) throw new Error('concatenateWavFiles: first WAV has no fmt chunk')
  const fmtWithHeader = first.subarray(fmt.offset - 8, fmt.offset + fmt.size)
  const dataParts: Buffer[] = []
  for (const buf of buffers) {
    const data = findRiffChunk(buf, 'data')
    if (data) dataParts.push(buf.subarray(data.offset, data.offset + data.size))
  }
  const pcm = Buffer.concat(dataParts)
  const riffHeader = Buffer.alloc(12)
  riffHeader.write('RIFF', 0, 'ascii')
  riffHeader.writeUInt32LE(4 + fmtWithHeader.length + 8 + pcm.length, 4)
  riffHeader.write('WAVE', 8, 'ascii')
  const dataHeader = Buffer.alloc(8)
  dataHeader.write('data', 0, 'ascii')
  dataHeader.writeUInt32LE(pcm.length, 4)
  const out = Buffer.concat([riffHeader, fmtWithHeader, dataHeader, pcm])
  await writeFile(outputPath, out)
  return out.length
}
