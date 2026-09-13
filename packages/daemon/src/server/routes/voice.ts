import { randomUUID } from 'node:crypto'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { mkdir, unlink } from 'node:fs/promises'
import type { AddressInfo } from 'node:net'
import { extname, join } from 'node:path'
import { createInterface } from 'node:readline'
import { streamDaemonChatEvents, type DaemonChatStreamPayload } from '@sepilotd/api-client'
import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { timingSafeTokenEqual } from '../auth.js'
import {
  concatenateWavFiles,
  configuredPiperLanguages,
  configureSpeechConcurrency,
  isSpeechBinaryMissingError,
  normalizeSpeechLanguage,
  runSpeechJob,
  synthesizeSpeech,
  transcribeAudio,
} from '../../media/speech.js'
import type { VoiceConfig } from '../../config/schema.js'
import { createJobsRepo } from '../../jobs/repo.js'
import { createJobRunner } from '../../jobs/runner.js'
import type { JobItem } from '../../jobs/types.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import { openApiJsonResponseRef, openApiSchemaRef, type OpenApiOverrideMap } from '../openapi.js'
import {
  configureVoiceRetention,
  getUploadedFile,
  storeUploadedFile,
  sweepOrphanVoiceFiles,
  type UploadedFile,
} from './file-registry.js'
import { getRuntimeDataDir, zodRequestValidation } from './utils.js'

const audioExtensions = new Set([
  '.aac',
  '.aiff',
  '.flac',
  '.m4a',
  '.mp3',
  '.oga',
  '.ogg',
  '.opus',
  '.wav',
  '.webm',
])

const speechTextHardLimitChars = 20_000
const speechTextSynthesisMaxChars = 6_000
const MEETING_VOICE_JOB_KIND = 'meeting_voice' as const
const MEETING_QUEUE_DEFAULT_CONCURRENCY = Math.max(
  1,
  Number.parseInt(process.env.SEPILOTD_MEETING_QUEUE_CONCURRENCY ?? '2', 10) || 2,
)
const MEETING_QUEUE_MAX_ATTEMPTS = Math.max(
  1,
  Number.parseInt(process.env.SEPILOTD_MEETING_QUEUE_MAX_ATTEMPTS ?? '3', 10) || 3,
)

export const voiceTurnRequestSchema = z
  .object({
    fileId: z.string().min(1).optional(),
    text: z.string().min(1).optional(),
    sessionId: z.string().min(1).optional(),
    provider: z.string().min(1).optional(),
    model: z.string().min(1).optional(),
    inputLanguage: z.string().min(1).optional(),
    outputLanguage: z.string().min(1).optional(),
    language: z.string().min(1).optional(),
    transcriptionModel: z.string().min(1).optional(),
    thinkingLevel: z.enum(['auto', 'off', 'low', 'medium', 'high', 'max']).optional(),
    purpose: z.enum(['conversation', 'meeting_notes']).optional(),
    enqueue: z.boolean().optional(),
    tts: z.boolean().optional(),
  })
  .refine((body) => body.fileId || body.text?.trim(), {
    message: 'fileId or text is required',
  })

const voiceSpeechRequestSchema = z.object({
  text: z.string().trim().min(1).max(speechTextHardLimitChars),
  language: z.string().min(1).optional(),
})

const voiceAudioSchema = z.object({
  fileId: z.string(),
  url: z.string(),
  mimeType: z.string(),
  size: z.number().int().nonnegative(),
})

const voiceRealtimeSchema = z.object({
  protocol: z.literal('sepilotd.voice.v1'),
  websocket: z.object({
    available: z.boolean(),
    path: z.string(),
  }),
  webrtc: z.object({
    signaling: z.boolean(),
    media: z.boolean(),
    reason: z.string().optional(),
  }),
})

const voiceStatusSchema = z.object({
  transport: z.literal('turn'),
  transcription: z.object({
    engine: z.literal('local-whisper'),
    configured: z.boolean(),
    defaultModel: z.string(),
    language: z.string().nullable(),
  }),
  synthesis: z.object({
    engine: z.literal('local-piper'),
    configured: z.boolean(),
    defaultLanguage: z.string().nullable(),
    availableLanguages: z.array(z.string()),
  }),
  realtime: voiceRealtimeSchema,
})

export const voiceStatusResponseSchema = z.object({
  data: voiceStatusSchema,
})

const voiceTurnResponseSchema = z.object({
  data: z.object({
    sessionId: z.string(),
    messageId: z.string(),
    content: z.string(),
    transcript: z.string(),
    transcription: z.object({
      model: z.string().nullable(),
      language: z.string().nullable(),
    }),
    speech: z.object({
      requested: z.boolean(),
      generated: z.boolean(),
      language: z.string().nullable(),
      error: z
        .object({
          code: z.string(),
          message: z.string(),
        })
        .optional(),
    }),
    audio: voiceAudioSchema.optional(),
  }),
})

const voiceMeetingQueueItemSchema = z.object({
  itemId: z.string(),
  jobId: z.string(),
  idx: z.number().int().nonnegative(),
  status: z.enum(['queued', 'running', 'succeeded', 'failed']),
  attempts: z.number().int().nonnegative(),
  maxAttempts: z.number().int().positive(),
  error: z.string().nullable(),
  createdAt: z.number().int().nonnegative(),
  startedAt: z.number().int().nullable(),
  finishedAt: z.number().int().nullable(),
})

const voiceMeetingQueueResponseSchema = z.object({
  data: z.object({
    itemId: z.string().optional(),
    sessionId: z.string(),
    position: z.number().int().positive().nullable(),
    status: z.enum(['queued', 'running', 'succeeded', 'failed']).optional(),
    queued: z.number().int().nonnegative().optional(),
    running: z.number().int().nonnegative().optional(),
    succeeded: z.number().int().nonnegative().optional(),
    failed: z.number().int().nonnegative().optional(),
    total: z.number().int().nonnegative().optional(),
    items: z.array(voiceMeetingQueueItemSchema).optional(),
  }),
})

const voiceSpeechResponseSchema = z.object({
  data: z.object({
    language: z.string().nullable(),
    truncated: z.boolean(),
    inputChars: z.number().int().nonnegative(),
    spokenChars: z.number().int().nonnegative(),
    audio: voiceAudioSchema,
  }),
})

export const voiceOpenApiComponents = openApiComponentsFromZod({
  schemas: {
    VoiceStatusResponse: voiceStatusResponseSchema,
    VoiceTurnRequest: voiceTurnRequestSchema,
    VoiceTurnResponse: voiceTurnResponseSchema,
    VoiceMeetingQueueResponse: voiceMeetingQueueResponseSchema,
    VoiceSpeechRequest: voiceSpeechRequestSchema,
    VoiceSpeechResponse: voiceSpeechResponseSchema,
  },
})

export const voiceOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/voice/status': {
    get: {
      summary: 'Get daemon voice backend status',
      tags: ['Voice'],
      responses: { 200: openApiJsonResponseRef('VoiceStatusResponse') },
    },
  },
  '/api/v1/voice/turn': {
    post: {
      summary: 'Run one voice chat turn',
      tags: ['Voice'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('VoiceTurnRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('VoiceTurnResponse'),
        202: openApiJsonResponseRef('VoiceMeetingQueueResponse'),
        400: { description: 'Invalid input or non-audio file' },
        404: { description: 'Uploaded file not found' },
        503: { description: 'Speech backend is not configured' },
      },
    },
  },
  '/api/v1/voice/meeting-queue/{sessionId}': {
    get: {
      summary: 'Get meeting capture queue status',
      tags: ['Voice'],
      parameters: [
        {
          name: 'sessionId',
          in: 'path',
          required: true,
          schema: { type: 'string' },
        },
      ],
      responses: {
        200: openApiJsonResponseRef('VoiceMeetingQueueResponse'),
      },
    },
  },
  '/api/v1/voice/speech': {
    post: {
      summary: 'Synthesize text to speech',
      tags: ['Voice'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('VoiceSpeechRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('VoiceSpeechResponse'),
        400: { description: 'Invalid input' },
        503: { description: 'Speech backend is not configured' },
      },
    },
  },
  '/api/v1/voice/realtime/ws': {
    get: {
      summary: 'Realtime voice WebSocket',
      tags: ['Voice'],
      responses: {
        101: { description: 'Realtime voice WebSocket upgrade' },
        401: { description: 'Unauthorized' },
      },
    },
  },
}

type ChatEnvelope = {
  data?: {
    sessionId: string
    messageId: string
    content: string
  }
  error?: {
    code?: string
    message?: string
  }
}

type VoiceTurnBody = z.infer<typeof voiceTurnRequestSchema>
type VoiceTurnEnvelope = z.infer<typeof voiceTurnResponseSchema>
type QueuedVoiceTurnRequest = VoiceTurnBody & { queuedFile?: UploadedFile }
type VoiceSpeechBody = z.infer<typeof voiceSpeechRequestSchema>
type VoiceSpeechEnvelope = z.infer<typeof voiceSpeechResponseSchema>
type VoicePurpose = NonNullable<VoiceTurnBody['purpose']>

type VoiceTurnResult =
  | { statusCode: 200; payload: VoiceTurnEnvelope }
  | { statusCode: number; payload: { error: { code: string; message: string } } | ChatEnvelope }

type VoiceSpeechResult =
  | { statusCode: 200; payload: VoiceSpeechEnvelope }
  | { statusCode: number; payload: { error: { code: string; message: string } } }

type PreparedVoiceTurn = {
  transcript: string
  transcription: { model: string | null; language: string | null }
  outputLanguage: string | undefined
  requestedSpeech: boolean
  purpose: VoicePurpose
  requestedSessionId: string | undefined
  chatHeaders: Record<string, string>
  chatPayload: {
    message: string
    sessionId?: string
    provider?: string
    model?: string
    mode: 'auto'
    thinkingLevel: NonNullable<VoiceTurnBody['thinkingLevel']>
    inputTrustLevel: 'trusted' | 'untrusted'
  }
}

type VoiceWebRtcBridge = {
  send(message: Record<string, unknown>): void
  close(): void
}

function voiceConfig(app?: FastifyInstance): VoiceConfig {
  return (
    app?.runtime?.config?.voice ?? {
      transcription: {},
      synthesis: {},
      fileRetentionMs: 60 * 60 * 1000,
      maxFiles: 200,
      maxConcurrent: 2,
    }
  )
}

function voiceEnvStatus(app?: FastifyInstance) {
  const config = voiceConfig(app)
  const transcription = config.transcription
  const synthesis = config.synthesis
  const whisperBin = transcription.whisperBin?.trim() || process.env.SEPILOTD_WHISPER_BIN?.trim()
  const whisperModel = transcription.model?.trim() || process.env.SEPILOTD_WHISPER_MODEL?.trim()
  const whisperLanguage =
    transcription.language?.trim() || process.env.SEPILOTD_WHISPER_LANG?.trim()
  const piperModel = synthesis.model?.trim() || process.env.SEPILOTD_PIPER_MODEL?.trim()
  const piperLanguage = synthesis.language?.trim() || process.env.SEPILOTD_PIPER_LANG?.trim()
  const piperLanguages = [
    ...new Set([
      ...configuredPiperLanguages(),
      ...(normalizeSpeechLanguage(synthesis.language) && synthesis.model?.trim()
        ? [normalizeSpeechLanguage(synthesis.language)!]
        : []),
    ]),
  ].sort()
  const webrtcMedia = Boolean(process.env.SEPILOTD_VOICE_WEBRTC_ADAPTER?.trim())
  return {
    transport: 'turn' as const,
    transcription: {
      engine: 'local-whisper' as const,
      configured: Boolean(whisperBin || whisperModel),
      defaultModel: whisperModel || 'base',
      language: whisperLanguage || null,
    },
    synthesis: {
      engine: 'local-piper' as const,
      configured: Boolean(piperModel || piperLanguages.length > 0),
      defaultLanguage: normalizeSpeechLanguage(piperLanguage) ?? null,
      availableLanguages: piperLanguages,
    },
    realtime: {
      protocol: 'sepilotd.voice.v1' as const,
      websocket: {
        available: true,
        path: '/api/v1/voice/realtime/ws',
      },
      webrtc: {
        signaling: true,
        media: webrtcMedia,
        ...(!webrtcMedia
          ? { reason: 'Set SEPILOTD_VOICE_WEBRTC_ADAPTER to enable a daemon WebRTC media adapter.' }
          : {}),
      },
    },
  }
}

function isAudioFile(filename: string, mimeType: string): boolean {
  return mimeType.startsWith('audio/') || audioExtensions.has(extname(filename).toLowerCase())
}

function normalizeThinkingLevel(value: unknown): VoiceTurnBody['thinkingLevel'] {
  if (
    value === 'off' ||
    value === 'low' ||
    value === 'medium' ||
    value === 'high' ||
    value === 'max'
  ) {
    return value
  }
  return undefined
}

function outputLanguageInstruction(language: string | undefined): string | null {
  if (!language) return null
  if (language.startsWith('ko')) {
    return 'Reply in natural spoken Korean. Use short conversational Korean sentences that sound natural when spoken by a Korean TTS voice.'
  }
  if (language.startsWith('en')) return 'Reply in natural spoken English.'
  if (language.startsWith('ja')) return 'Reply in natural spoken Japanese.'
  if (language.startsWith('zh')) return 'Reply in natural spoken Chinese.'
  if (language.startsWith('es')) return 'Reply in natural spoken Spanish.'
  if (language.startsWith('fr')) return 'Reply in natural spoken French.'
  if (language.startsWith('de')) return 'Reply in natural spoken German.'
  return `Reply in ${language}.`
}

function outputLanguageWritingInstruction(language: string | undefined): string | null {
  if (!language) return null
  if (language.startsWith('ko')) return 'Write the meeting notes in Korean.'
  if (language.startsWith('en')) return 'Write the meeting notes in English.'
  if (language.startsWith('ja')) return 'Write the meeting notes in Japanese.'
  if (language.startsWith('zh')) return 'Write the meeting notes in Chinese.'
  if (language.startsWith('es')) return 'Write the meeting notes in Spanish.'
  if (language.startsWith('fr')) return 'Write the meeting notes in French.'
  if (language.startsWith('de')) return 'Write the meeting notes in German.'
  return `Write the meeting notes in ${language}.`
}

function inferOutputLanguageFromText(text: string): string | undefined {
  return /[\u3131-\u318e\uac00-\ud7a3]/.test(text) ? 'ko' : undefined
}

function buildVoicePrompt(
  transcript: string,
  fromAudio: boolean,
  outputLanguage: string | undefined,
  thinkingLevel: VoiceTurnBody['thinkingLevel'],
  purpose: VoicePurpose,
): string {
  if (purpose === 'meeting_notes') {
    return [
      'Meeting audio transcript segment:',
      '',
      transcript,
      '',
      'Convert this segment into durable meeting minutes for the current session.',
      'Do not answer the speakers as a chat assistant.',
      'Extract only useful meeting content. Preserve concrete decisions, action items with owners or dates when present, open questions, and important context.',
      'Do not restate notes that were already captured in earlier session context unless this segment changes them.',
      'If this segment contains no useful meeting content, reply exactly: No new meeting notes.',
      outputLanguageWritingInstruction(outputLanguage),
    ]
      .filter(Boolean)
      .join('\n')
  }
  const instruction = outputLanguageInstruction(outputLanguage)
  const thinkingEnabled = thinkingLevel && thinkingLevel !== 'off'
  const voiceInstruction = [
    'Reply naturally and keep the answer concise enough for spoken playback.',
    thinkingEnabled
      ? 'If reasoning is shown, keep the final spoken answer concise.'
      : 'Do not include hidden reasoning, <think> tags, markdown tables, or code unless explicitly requested.',
  ].join(' ')
  if (!fromAudio) {
    return [transcript, voiceInstruction, instruction].filter(Boolean).join('\n\n')
  }
  return ['Voice message transcript:', '', transcript, '', voiceInstruction, instruction]
    .filter((line) => line !== null && line !== undefined)
    .join('\n')
}

function stripForSpeech(text: string): string {
  return text
    .replace(/<think\b[^>]*>[\s\S]*?<\/think>/gi, '')
    .replace(/<think\b[^>]*>[\s\S]*$/gi, '')
    .replace(/<\/?think\b[^>]*>/gi, '')
    .replace(/```[\s\S]*?```/g, 'code block omitted')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/<[^>]+>/g, '')
    .replace(/[`*_>#]/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

type PreparedSpeechText = {
  text: string
  inputChars: number
  spokenChars: number
  truncated: boolean
  // Sentence-boundary chunks covering the full text. Long replies are
  // synthesized as multiple clips and concatenated instead of silently dropping
  // everything past the single-clip cap.
  chunks: string[]
}

/**
 * Split speech text into <= maxChars chunks at sentence boundaries so long
 * replies can be synthesized as multiple clips rather than truncated. Falls back
 * to a hard cut only when no boundary exists within a window.
 */
function splitSpeechIntoChunks(text: string, maxChars: number): string[] {
  const trimmed = text.trim()
  if (!trimmed) return []
  if (Array.from(trimmed).length <= maxChars) return [trimmed]
  const chunks: string[] = []
  let remaining = trimmed
  const boundaryPattern = /(?:\n\n|\n|[.!?]\s|[。！？])/g
  while (Array.from(remaining).length > maxChars) {
    const window = Array.from(remaining).slice(0, maxChars).join('')
    const minBoundaryIndex = Math.floor(window.length * 0.5)
    let cut = -1
    for (const match of window.matchAll(boundaryPattern)) {
      const endIndex = (match.index ?? 0) + match[0].length
      if (endIndex >= minBoundaryIndex) cut = endIndex
    }
    if (cut <= 0) cut = window.length
    const piece = remaining.slice(0, cut).trim()
    if (piece) chunks.push(piece)
    remaining = remaining.slice(cut)
  }
  const tail = remaining.trim()
  if (tail) chunks.push(tail)
  return chunks
}

function countSpeechChars(text: string): number {
  return Array.from(text).length
}

function prepareSpeechText(text: string): PreparedSpeechText {
  const stripped = stripForSpeech(text)
  const inputChars = countSpeechChars(stripped)
  const chunks = splitSpeechIntoChunks(stripped, speechTextSynthesisMaxChars)
  // The full stripped text is spoken across the chunks; `text` keeps the
  // first-clip content for callers that synthesize a single WAV, while `chunks`
  // covers everything for the multi-clip path.
  const spoken = chunks.join(' ')
  const spokenChars = countSpeechChars(spoken)
  return {
    text: chunks[0] ?? stripped.slice(0, speechTextSynthesisMaxChars),
    inputChars,
    spokenChars,
    truncated: false,
    chunks,
  }
}

/**
 * Synthesize a (possibly multi-clip) reply into a single WAV at outputPath.
 * Long replies are split into sentence-boundary chunks, each synthesized under
 * the media.speech concurrency limit, then concatenated — no silent truncation.
 */
async function synthesizeReplyClips(
  chunks: string[],
  outputPath: string,
  synth: { binaryPath?: string; modelPath?: string; language?: string },
): Promise<{ audioPath: string; bytes: number }> {
  if (chunks.length <= 1) {
    const result = await runSpeechJob(() =>
      synthesizeSpeech({ text: chunks[0] ?? '', outputPath, ...synth }),
    )
    return { audioPath: result.audioPath, bytes: result.bytes }
  }
  const clipPaths: string[] = []
  for (let i = 0; i < chunks.length; i += 1) {
    const clipPath = `${outputPath}.part-${i}.wav`
    await runSpeechJob(() => synthesizeSpeech({ text: chunks[i], outputPath: clipPath, ...synth }))
    clipPaths.push(clipPath)
  }
  const bytes = await concatenateWavFiles(clipPaths, outputPath)
  await Promise.all(clipPaths.map((p) => unlink(p).catch(() => {})))
  return { audioPath: outputPath, bytes }
}

function errorPayload(code: string, message: string) {
  return { error: { code, message } }
}

function isVoiceTurnSuccess(
  result: VoiceTurnResult,
): result is { statusCode: 200; payload: VoiceTurnEnvelope } {
  if (result.statusCode !== 200 || !('data' in result.payload)) return false
  const payload = result.payload as { data?: { transcript?: unknown } }
  return typeof payload.data?.transcript === 'string'
}

function websocketTokenFromUrl(url: string): string | null {
  const parsed = new URL(url, 'http://localhost')
  return parsed.searchParams.get('token') || parsed.searchParams.get('access_token')
}

/**
 * Parse the `SEPILOTD_VOICE_WEBRTC_ADAPTER` env value into a command + args
 * pair. Two accepted forms:
 *
 *   1. `"/path/to/bin --flag value"` — plain shell-style string. Split on
 *      whitespace, but only if no shell metacharacters are present. This
 *      stays backward-compatible with existing operator configs.
 *   2. `'["/path with spaces/bin","--flag","value"]'` — JSON array form for
 *      paths/args that need quoting (whitespace, special characters).
 *
 * Strings that contain `<>|&;`$()\\` are refused outright — the previous
 * implementation passed `shell: true` to `spawn`, which interpreted those
 * metacharacters. Now that we run the binary directly, any operator who
 * relied on shell features must switch to the JSON form so the intent is
 * explicit instead of relying on an implicit shell interpreter.
 */
function parseAdapterCommand(envValue: string): { command: string; args: string[] } {
  const trimmed = envValue.trim()
  if (trimmed.startsWith('[')) {
    try {
      const parsed = JSON.parse(trimmed) as unknown
      if (
        Array.isArray(parsed) &&
        parsed.length > 0 &&
        parsed.every((p) => typeof p === 'string')
      ) {
        return { command: parsed[0] as string, args: (parsed as string[]).slice(1) }
      }
    } catch {
      // fall through to the error below
    }
    throw new Error('SEPILOTD_VOICE_WEBRTC_ADAPTER JSON form must be a non-empty string array')
  }
  if (/[<>|&;`$()\\]/.test(trimmed)) {
    throw new Error(
      'SEPILOTD_VOICE_WEBRTC_ADAPTER contains shell metacharacters; ' +
        'use the JSON form (["/path","--flag"]) for paths/args that need quoting',
    )
  }
  const parts = trimmed.split(/\s+/).filter(Boolean)
  if (parts.length === 0) {
    throw new Error('SEPILOTD_VOICE_WEBRTC_ADAPTER is empty')
  }
  return { command: parts[0]!, args: parts.slice(1) }
}

function createVoiceWebRtcBridge(options: {
  command: string
  onEvent: (event: Record<string, unknown>) => void
  onError: (message: string) => void
}): VoiceWebRtcBridge {
  // Run the adapter binary directly (no shell), parsing the env value into
  // a command + args pair. Dropping `shell: true` eliminates the implicit
  // shell-metacharacter interpretation that the previous version trusted.
  const { command, args } = parseAdapterCommand(options.command)
  const child: ChildProcessWithoutNullStreams = spawn(command, args, {
    stdio: ['pipe', 'pipe', 'pipe'],
  })
  const stdout = createInterface({ input: child.stdout })
  let stderr = ''
  let closed = false

  stdout.on('line', (line) => {
    const trimmed = line.trim()
    if (!trimmed) return
    try {
      const event = JSON.parse(trimmed) as Record<string, unknown>
      options.onEvent(event)
    } catch {
      options.onError(`WebRTC adapter emitted invalid JSON: ${trimmed.slice(0, 200)}`)
    }
  })

  child.stderr.on('data', (chunk) => {
    stderr += String(chunk)
    if (stderr.length > 8192) stderr = stderr.slice(-8192)
  })

  child.on('error', (error) => {
    options.onError(error.message)
  })

  child.on('close', (code) => {
    if (closed) return
    closed = true
    options.onEvent({
      type: 'webrtc.adapter.closed',
      code,
      stderr: stderr.trim() || null,
    })
  })

  return {
    send(message) {
      if (closed || child.stdin.destroyed) return
      child.stdin.write(`${JSON.stringify(message)}\n`)
    },
    close() {
      if (closed) return
      closed = true
      stdout.close()
      child.stdin.end()
      child.kill('SIGTERM')
    },
  }
}

function resolveAuthorizationHeader(
  headers: { authorization?: string | string[] },
  fallbackToken?: string | null,
): string | undefined {
  const authorization = headers.authorization
  if (typeof authorization === 'string' && authorization.trim()) {
    return authorization
  }
  if (Array.isArray(authorization)) {
    const value = authorization.find((item) => item.trim())
    if (value) return value
  }
  return fallbackToken ? `Bearer ${fallbackToken}` : undefined
}

async function prepareVoiceTurn(
  headers: { authorization?: string | string[] },
  body: VoiceTurnBody,
  options: { fallbackToken?: string | null; voiceConfig?: VoiceConfig } = {},
): Promise<{ ok: true; data: PreparedVoiceTurn } | { ok: false; result: VoiceTurnResult }> {
  const inputLanguage = normalizeSpeechLanguage(body.inputLanguage ?? body.language)
  let outputLanguage = normalizeSpeechLanguage(body.outputLanguage) ?? inputLanguage
  const transcriptionConfig = options.voiceConfig?.transcription
  const thinkingLevel = body.thinkingLevel ?? 'off'
  const purpose = body.purpose ?? 'conversation'
  let transcript = body.text?.trim() || ''
  let transcription: { model: string | null; language: string | null } = {
    model: null,
    language: null,
  }
  const audioDerivedInput = Boolean(body.fileId)

  if (body.fileId) {
    const file = getUploadedFile(body.fileId)
    if (!file) {
      return {
        ok: false,
        result: {
          statusCode: 404,
          payload: errorPayload('NOT_FOUND', 'Uploaded voice file not found'),
        },
      }
    }
    if (!isAudioFile(file.filename, file.mimeType)) {
      return {
        ok: false,
        result: {
          statusCode: 400,
          payload: errorPayload('INVALID_AUDIO_FILE', 'Uploaded file is not an audio file'),
        },
      }
    }

    try {
      const result = await runSpeechJob(() =>
        transcribeAudio({
          audioPath: file.path,
          binaryPath: transcriptionConfig?.whisperBin,
          language:
            inputLanguage ??
            normalizeSpeechLanguage(
              transcriptionConfig?.language ?? process.env.SEPILOTD_WHISPER_LANG,
            ),
          model: body.transcriptionModel ?? transcriptionConfig?.model,
        }),
      )
      transcript = result.text
      transcription = {
        model: result.model,
        language: result.language ?? null,
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      return {
        ok: false,
        result: {
          statusCode: isSpeechBinaryMissingError(error) ? 503 : 500,
          payload: errorPayload(
            isSpeechBinaryMissingError(error) ? 'SPEECH_BINARY_MISSING' : 'TRANSCRIPTION_FAILED',
            message,
          ),
        },
      }
    }
  }

  if (!transcript) {
    return {
      ok: false,
      result: {
        statusCode: 400,
        payload: errorPayload('EMPTY_TRANSCRIPT', 'Voice transcript is empty'),
      },
    }
  }
  outputLanguage ??=
    normalizeSpeechLanguage(transcription.language ?? undefined) ??
    inferOutputLanguageFromText(transcript)

  const chatHeaders: Record<string, string> = {
    'content-type': 'application/json',
  }
  const authorization = resolveAuthorizationHeader(headers, options.fallbackToken)
  if (authorization) {
    chatHeaders.authorization = authorization
  }

  return {
    ok: true,
    data: {
      transcript,
      transcription,
      outputLanguage,
      requestedSpeech: purpose === 'meeting_notes' ? false : body.tts !== false,
      purpose,
      requestedSessionId: body.sessionId,
      chatHeaders,
      chatPayload: {
        message: buildVoicePrompt(
          transcript,
          Boolean(body.fileId),
          outputLanguage,
          thinkingLevel,
          purpose,
        ),
        sessionId: body.sessionId,
        provider: body.provider,
        model: body.model,
        mode: 'auto',
        thinkingLevel,
        inputTrustLevel: audioDerivedInput ? 'untrusted' : 'trusted',
      },
    },
  }
}

async function updateMeetingSessionMeta(
  app: FastifyInstance,
  prepared: PreparedVoiceTurn,
  sessionId: string,
): Promise<void> {
  if (prepared.purpose !== 'meeting_notes' || prepared.requestedSessionId) return
  const runtime = app.runtime
  if (!runtime?.sessions?.updateMeta || !runtime.sessions.get) return
  const session = await runtime.sessions.get(sessionId).catch(() => null)
  if (!session) return
  const generatedFromPrompt =
    session.title.startsWith('Meeting audio transcript segment') ||
    session.title.startsWith('Voice message transcript') ||
    session.title.trim().length === 0
  if (!generatedFromPrompt) return
  const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ')
  await runtime.sessions.updateMeta(sessionId, {
    title: `Meeting notes ${stamp}`,
  })
}

async function finalizeVoiceTurn(
  app: FastifyInstance,
  prepared: PreparedVoiceTurn,
  chat: { sessionId: string; messageId: string; content: string },
): Promise<VoiceTurnResult> {
  await updateMeetingSessionMeta(app, prepared, chat.sessionId)
  let audio:
    | {
        fileId: string
        url: string
        mimeType: string
        size: number
      }
    | undefined
  let speechError: { code: string; message: string } | undefined
  if (prepared.requestedSpeech) {
    const runtime = app.runtime
    const synthesisConfig = voiceConfig(app).synthesis
    if (!runtime) {
      speechError = {
        code: 'SERVICE_UNAVAILABLE',
        message: 'Runtime is required to store synthesized voice replies',
      }
    } else {
      try {
        const speechText = prepareSpeechText(chat.content)
        if (!speechText.text) {
          speechError = {
            code: 'EMPTY_SPEECH_TEXT',
            message: 'Reply did not contain text suitable for speech synthesis',
          }
        } else {
          const id = randomUUID()
          const voiceDir = join(getRuntimeDataDir(runtime), 'voice')
          await mkdir(voiceDir, { recursive: true })
          const outputPath = join(voiceDir, `${id}.wav`)
          const result = await synthesizeReplyClips(speechText.chunks, outputPath, {
            binaryPath: synthesisConfig.piperBin,
            modelPath: synthesisConfig.model,
            language:
              prepared.outputLanguage ??
              normalizeSpeechLanguage(synthesisConfig.language ?? process.env.SEPILOTD_PIPER_LANG),
          })
          storeUploadedFile({
            id,
            filename: 'voice-reply.wav',
            mimeType: 'audio/wav',
            size: result.bytes,
            path: result.audioPath,
            uploadedAt: new Date().toISOString(),
            kind: 'voice',
          })
          audio = {
            fileId: id,
            url: `/api/v1/files/${encodeURIComponent(id)}/download`,
            mimeType: 'audio/wav',
            size: result.bytes,
          }
        }
      } catch (error) {
        speechError = {
          code: isSpeechBinaryMissingError(error)
            ? 'SPEECH_BINARY_MISSING'
            : 'SPEECH_SYNTHESIS_FAILED',
          message: error instanceof Error ? error.message : String(error),
        }
      }
    }
  }

  return {
    statusCode: 200,
    payload: {
      data: {
        sessionId: chat.sessionId,
        messageId: chat.messageId,
        content: chat.content,
        transcript: prepared.transcript,
        transcription: prepared.transcription,
        speech: {
          requested: prepared.requestedSpeech,
          generated: Boolean(audio),
          language: prepared.outputLanguage ?? null,
          ...(speechError ? { error: speechError } : {}),
        },
        ...(audio ? { audio } : {}),
      },
    },
  }
}

async function executeVoiceSpeech(
  app: FastifyInstance,
  body: VoiceSpeechBody,
): Promise<VoiceSpeechResult> {
  const runtime = app.runtime
  if (!runtime) {
    return {
      statusCode: 503,
      payload: errorPayload(
        'SERVICE_UNAVAILABLE',
        'Runtime is required to store synthesized speech',
      ),
    }
  }

  const speechText = prepareSpeechText(body.text)
  if (!speechText.text) {
    return {
      statusCode: 400,
      payload: errorPayload(
        'EMPTY_SPEECH_TEXT',
        'Text did not contain content suitable for speech synthesis',
      ),
    }
  }

  const language =
    normalizeSpeechLanguage(body.language) ??
    normalizeSpeechLanguage(
      voiceConfig(app).synthesis.language ?? process.env.SEPILOTD_PIPER_LANG,
    ) ??
    inferOutputLanguageFromText(speechText.text)
  try {
    const id = randomUUID()
    const voiceDir = join(getRuntimeDataDir(runtime), 'voice')
    await mkdir(voiceDir, { recursive: true })
    const outputPath = join(voiceDir, `${id}.wav`)
    const result = await synthesizeReplyClips(speechText.chunks, outputPath, {
      binaryPath: voiceConfig(app).synthesis.piperBin,
      modelPath: voiceConfig(app).synthesis.model,
      language,
    })
    storeUploadedFile({
      id,
      filename: 'speech.wav',
      mimeType: 'audio/wav',
      size: result.bytes,
      path: result.audioPath,
      uploadedAt: new Date().toISOString(),
      kind: 'voice',
    })
    return {
      statusCode: 200,
      payload: {
        data: {
          language: language ?? null,
          truncated: speechText.truncated,
          inputChars: speechText.inputChars,
          spokenChars: speechText.spokenChars,
          audio: {
            fileId: id,
            url: `/api/v1/files/${encodeURIComponent(id)}/download`,
            mimeType: 'audio/wav',
            size: result.bytes,
          },
        },
      },
    }
  } catch (error) {
    return {
      statusCode: isSpeechBinaryMissingError(error) ? 503 : 500,
      payload: errorPayload(
        isSpeechBinaryMissingError(error) ? 'SPEECH_BINARY_MISSING' : 'SPEECH_SYNTHESIS_FAILED',
        error instanceof Error ? error.message : String(error),
      ),
    }
  }
}

async function executeVoiceTurn(
  app: FastifyInstance,
  headers: { authorization?: string | string[] },
  body: VoiceTurnBody,
): Promise<VoiceTurnResult> {
  const preparedResult = await prepareVoiceTurn(headers, body, {
    voiceConfig: voiceConfig(app),
  })
  if (!preparedResult.ok) return preparedResult.result
  const prepared = preparedResult.data

  const chatResponse = await app.inject({
    method: 'POST',
    url: '/api/v1/chat',
    headers: prepared.chatHeaders,
    payload: prepared.chatPayload,
  })

  let chat: ChatEnvelope
  try {
    chat = JSON.parse(chatResponse.body) as ChatEnvelope
  } catch {
    return {
      statusCode: 502,
      payload: errorPayload('CHAT_FAILED', chatResponse.body),
    }
  }

  if (chatResponse.statusCode >= 400 || !chat.data) {
    return { statusCode: chatResponse.statusCode, payload: chat }
  }

  return finalizeVoiceTurn(app, prepared, chat.data)
}

function queuedVoiceTurnBody(body: VoiceTurnBody, sessionId: string): QueuedVoiceTurnRequest {
  const queued: QueuedVoiceTurnRequest = {
    ...body,
    sessionId,
    enqueue: false,
    tts: false,
  }
  if (body.fileId) {
    const file = getUploadedFile(body.fileId)
    if (file) queued.queuedFile = file
  }
  return queued
}

function validateQueuedVoiceFile(body: VoiceTurnBody): VoiceTurnResult | null {
  if (!body.fileId) return null
  const file = getUploadedFile(body.fileId)
  if (!file) {
    return {
      statusCode: 404,
      payload: errorPayload('NOT_FOUND', 'Uploaded voice file not found'),
    }
  }
  if (!isAudioFile(file.filename, file.mimeType)) {
    return {
      statusCode: 400,
      payload: errorPayload('INVALID_AUDIO_FILE', 'Uploaded file is not an audio file'),
    }
  }
  return null
}

function restoreQueuedVoiceFile(request: QueuedVoiceTurnRequest): void {
  if (!request.queuedFile) return
  if (getUploadedFile(request.queuedFile.id)) return
  storeUploadedFile(request.queuedFile)
}

function stripQueuedVoiceMetadata(request: QueuedVoiceTurnRequest): VoiceTurnBody {
  const { queuedFile: _queuedFile, ...body } = request
  return { ...body, enqueue: false }
}

function voiceTurnErrorMessage(result: VoiceTurnResult): string {
  const payload = result.payload
  if ('error' in payload && payload.error?.message) return payload.error.message
  return `voice turn failed (${result.statusCode})`
}

function internalVoiceQueueHeaders(app: FastifyInstance): { authorization?: string } {
  return app.authToken ? { authorization: `Bearer ${app.authToken}` } : {}
}

function resolveInternalChatStreamUrl(
  app: FastifyInstance,
  host: string | undefined,
): string | null {
  const address = app.server.address() as AddressInfo | string | null
  if (address && typeof address === 'object' && Number.isFinite(address.port)) {
    return `http://127.0.0.1:${address.port}/api/v1/chat/stream`
  }
  if (host?.trim()) {
    return `http://${host.trim()}/api/v1/chat/stream`
  }
  return null
}

async function chatStreamErrorPayload(response: Response): Promise<VoiceTurnResult> {
  const body = await response.text().catch(() => '')
  try {
    return {
      statusCode: response.status,
      payload: JSON.parse(body) as ChatEnvelope,
    }
  } catch {
    return {
      statusCode: response.status,
      payload: errorPayload('CHAT_STREAM_FAILED', body || response.statusText),
    }
  }
}

function streamPayloadType(payload: unknown): string | undefined {
  if (!payload || typeof payload !== 'object') return undefined
  const type = (payload as { type?: unknown }).type
  return typeof type === 'string' ? type : undefined
}

async function executeVoiceTurnStreaming(
  app: FastifyInstance,
  headers: { authorization?: string | string[]; host?: string | string[] },
  body: VoiceTurnBody,
  options: {
    fallbackToken?: string | null
    onTranscript: (prepared: PreparedVoiceTurn) => void
    onAssistantDelta: (event: { sessionId: string; text: string }) => void
    // Barge-in: aborting this signal cancels the in-flight chat stream fetch so
    // an interrupt (or a closed socket) stops the turn instead of running the
    // full LLM turn + TTS to completion.
    signal?: AbortSignal
  },
): Promise<VoiceTurnResult> {
  const preparedResult = await prepareVoiceTurn(headers, body, {
    fallbackToken: options.fallbackToken,
    voiceConfig: voiceConfig(app),
  })
  if (!preparedResult.ok) return preparedResult.result
  const prepared = preparedResult.data
  options.onTranscript(prepared)

  const host = Array.isArray(headers.host) ? headers.host[0] : headers.host
  const streamUrl = resolveInternalChatStreamUrl(app, host)
  if (!streamUrl) {
    return executeVoiceTurn(app, headers, body)
  }

  const response = await fetch(streamUrl, {
    method: 'POST',
    headers: prepared.chatHeaders,
    body: JSON.stringify(prepared.chatPayload),
    signal: options.signal,
  })
  if (!response.ok || !response.body) {
    return chatStreamErrorPayload(response)
  }

  let sessionId = prepared.chatPayload.sessionId ?? ''
  let content = ''
  let sawTextDelta = false
  let streamError: { code?: string; message?: string } | null = null

  for await (const payload of streamDaemonChatEvents<DaemonChatStreamPayload>(response)) {
    if (
      payload &&
      typeof payload === 'object' &&
      'sessionId' in payload &&
      typeof payload.sessionId === 'string'
    ) {
      sessionId = payload.sessionId
    }
    const type = streamPayloadType(payload)
    if (type === 'text_delta') {
      const text = (payload as { text?: unknown }).text
      if (typeof text === 'string' && text) {
        sawTextDelta = true
        content += text
        options.onAssistantDelta({ sessionId, text })
      }
      continue
    }
    if (type === 'message') {
      const message = (payload as { content?: unknown }).content
      if (typeof message === 'string') {
        content = message
        if (!sawTextDelta && message) {
          options.onAssistantDelta({ sessionId, text: message })
        }
      }
      continue
    }
    if (type === 'error') {
      const error = (payload as { error?: unknown }).error
      streamError =
        error && typeof error === 'object'
          ? (error as { code?: string; message?: string })
          : { code: 'CHAT_STREAM_FAILED', message: 'Voice chat stream failed' }
    }
  }

  if (streamError) {
    return {
      statusCode: streamError.code === 'SERVICE_UNAVAILABLE' ? 503 : 500,
      payload: errorPayload(
        streamError.code ?? 'CHAT_STREAM_FAILED',
        streamError.message ?? 'Voice chat stream failed',
      ),
    }
  }

  const finalContent = content.trim()
  if (!finalContent) {
    return {
      statusCode: 502,
      payload: errorPayload('EMPTY_CHAT_RESPONSE', 'Voice chat stream produced no assistant text'),
    }
  }

  return finalizeVoiceTurn(app, prepared, {
    sessionId: sessionId || prepared.chatPayload.sessionId || randomUUID(),
    messageId: randomUUID(),
    content: finalContent,
  })
}

export const __voiceRouteTesting = {
  voiceEnvStatus,
  isAudioFile,
  normalizeThinkingLevel,
  outputLanguageInstruction,
  outputLanguageWritingInstruction,
  inferOutputLanguageFromText,
  buildVoicePrompt,
  stripForSpeech,
  prepareSpeechText,
  splitSpeechIntoChunks,
  errorPayload,
  isVoiceTurnSuccess,
  websocketTokenFromUrl,
  parseAdapterCommand,
  createVoiceWebRtcBridge,
  resolveAuthorizationHeader,
  prepareVoiceTurn,
  finalizeVoiceTurn,
  executeVoiceSpeech,
  executeVoiceTurn,
  resolveInternalChatStreamUrl,
  chatStreamErrorPayload,
  streamPayloadType,
  executeVoiceTurnStreaming,
}

export async function voiceRoutes(app: FastifyInstance) {
  // Apply configured retention and reap orphaned WAVs left by a prior process.
  // The in-memory registry does not survive restart, so every WAV on disk from
  // a previous run is orphaned (its fileId would 404). Best-effort, non-blocking.
  const voiceCfg = voiceConfig(app)
  configureVoiceRetention({
    ttlMs: voiceCfg.fileRetentionMs,
    maxEntries: voiceCfg.maxFiles,
  })
  configureSpeechConcurrency(voiceCfg.maxConcurrent)
  try {
    if (app.runtime) {
      const voiceDir = join(getRuntimeDataDir(app.runtime), 'voice')
      void sweepOrphanVoiceFiles(voiceDir)
    }
  } catch {
    /* best-effort boot sweep */
  }

  const meetingQueueRepo = createJobsRepo()
  const recoveredMeetingItems =
    meetingQueueRepo.requeueInterruptedPartitioned(MEETING_VOICE_JOB_KIND)
  if (recoveredMeetingItems > 0) {
    app.log.warn(
      { recovered: recoveredMeetingItems },
      'requeued interrupted meeting voice items on startup',
    )
  }
  const meetingQueueRunner = createJobRunner({
    repo: meetingQueueRepo,
    globalMax: Math.max(
      1,
      Number.parseInt(process.env.SEPILOTD_MEETING_QUEUE_GLOBAL_MAX ?? '4', 10) || 4,
    ),
  })
  const runMeetingQueue = (jobId: string, concurrency = MEETING_QUEUE_DEFAULT_CONCURRENCY) => {
    if (meetingQueueRunner.isRunning(jobId)) return
    setImmediate(() => {
      void meetingQueueRunner
        .runPartitioned({
          jobId,
          concurrency,
          execute: async (item: JobItem) => {
            const queued = item.request as QueuedVoiceTurnRequest
            restoreQueuedVoiceFile(queued)
            const result = await executeVoiceTurn(
              app,
              internalVoiceQueueHeaders(app),
              stripQueuedVoiceMetadata(queued),
            )
            if (!isVoiceTurnSuccess(result)) throw new Error(voiceTurnErrorMessage(result))
            return result.payload
          },
        })
        .catch((error) => {
          app.log.warn(
            { jobId, err: error instanceof Error ? error.message : String(error) },
            'meeting voice queue runner failed',
          )
        })
    })
  }
  for (const job of meetingQueueRepo
    .listInProgress()
    .filter((candidate) => candidate.kind === MEETING_VOICE_JOB_KIND)) {
    runMeetingQueue(job.id, job.concurrency)
  }

  app.get('/voice/status', async () => ({ data: voiceEnvStatus(app) }))

  app.get('/voice/realtime/ws', { websocket: true }, (socket, request) => {
    let closed = false
    let busy = false
    let webrtcBridge: VoiceWebRtcBridge | null = null
    // Barge-in: the controller for the currently running turn, so an interrupt
    // or a socket close aborts the in-flight chat stream fetch.
    let currentTurnAbort: AbortController | null = null
    const token = websocketTokenFromUrl(request.url)
    if (
      (app.authTokenRequired || app.authToken)
      && request.authContext?.kind !== 'master'
      && !timingSafeTokenEqual(token, app.authToken)
    ) {
      socket.send(
        JSON.stringify({
          type: 'voice.error',
          error: { code: 'UNAUTHORIZED', message: 'Realtime voice authentication required' },
        }),
      )
      socket.close(1008, 'Unauthorized')
      return
    }

    const safeSend = (payload: unknown) => {
      try {
        if (!closed && socket.readyState === 1) socket.send(JSON.stringify(payload))
      } catch {
        /* socket may have closed */
      }
    }

    const closeWebRtcBridge = () => {
      webrtcBridge?.close()
      webrtcBridge = null
    }

    const ensureWebRtcBridge = (): VoiceWebRtcBridge | null => {
      if (webrtcBridge) return webrtcBridge
      const command = process.env.SEPILOTD_VOICE_WEBRTC_ADAPTER?.trim()
      if (!command) return null
      webrtcBridge = createVoiceWebRtcBridge({
        command,
        onEvent: (event) => safeSend(event),
        onError: (message) =>
          safeSend({
            type: 'voice.error',
            error: { code: 'WEBRTC_ADAPTER_ERROR', message },
          }),
      })
      return webrtcBridge
    }

    safeSend({
      type: 'voice.ready',
      protocol: 'sepilotd.voice.v1',
      status: voiceEnvStatus(app),
    })

    socket.on('close', () => {
      closed = true
      // Cancel any in-flight turn so a client that hangs up does not leave the
      // full LLM turn + TTS running (previously the fetch was never cancelled).
      currentTurnAbort?.abort()
      closeWebRtcBridge()
    })

    socket.on('message', async (raw) => {
      let message: Record<string, unknown>
      try {
        message = JSON.parse(raw.toString()) as Record<string, unknown>
      } catch {
        safeSend({
          type: 'voice.error',
          error: { code: 'INVALID_JSON', message: 'Message must be JSON' },
        })
        return
      }

      if (message.type === 'ping') {
        safeSend({ type: 'pong', at: new Date().toISOString() })
        return
      }

      if (message.type === 'session.start') {
        safeSend({
          type: 'session.started',
          sessionId: typeof message.sessionId === 'string' ? message.sessionId : null,
          protocol: 'sepilotd.voice.v1',
        })
        return
      }

      if (message.type === 'session.stop') {
        safeSend({ type: 'session.stopped' })
        socket.close(1000, 'Voice session stopped')
        return
      }

      if (message.type === 'playback.interrupt') {
        const turnId = typeof message.turnId === 'string' ? message.turnId : null
        // Abort the in-flight turn so an interrupt actually stops the running
        // LLM turn + TTS instead of only echoing an acknowledgement.
        currentTurnAbort?.abort()
        if (webrtcBridge) {
          webrtcBridge.send({ type: 'playback.interrupt', turnId })
          return
        }
        safeSend({
          type: 'voice.interrupted',
          turnId,
        })
        return
      }

      if (message.type === 'webrtc.offer' || message.type === 'webrtc.ice') {
        const bridge = ensureWebRtcBridge()
        if (bridge) {
          bridge.send({
            ...message,
            sessionId: typeof message.sessionId === 'string' ? message.sessionId : undefined,
          })
          return
        }
        safeSend({
          type: 'webrtc.unavailable',
          signaling: true,
          media: false,
          reason:
            'Daemon WebRTC media adapter is not configured. Set SEPILOTD_VOICE_WEBRTC_ADAPTER when a media bridge is installed.',
        })
        return
      }

      if (message.type !== 'voice.turn') {
        safeSend({
          type: 'voice.error',
          error: { code: 'UNKNOWN_MESSAGE', message: 'Unsupported realtime voice message type' },
        })
        return
      }

      if (busy) {
        safeSend({
          type: 'voice.error',
          error: { code: 'BUSY', message: 'A realtime voice turn is already running' },
        })
        return
      }

      busy = true
      const turnId = typeof message.turnId === 'string' ? message.turnId : randomUUID()
      const turnAbort = new AbortController()
      currentTurnAbort = turnAbort
      safeSend({ type: 'voice.state', turnId, state: 'thinking' })
      try {
        const voiceTurnBody: VoiceTurnBody = {
          fileId: typeof message.fileId === 'string' ? message.fileId : undefined,
          text: typeof message.text === 'string' ? message.text : undefined,
          sessionId: typeof message.sessionId === 'string' ? message.sessionId : undefined,
          provider: typeof message.provider === 'string' ? message.provider : undefined,
          model: typeof message.model === 'string' ? message.model : undefined,
          inputLanguage:
            typeof message.inputLanguage === 'string' ? message.inputLanguage : undefined,
          outputLanguage:
            typeof message.outputLanguage === 'string' ? message.outputLanguage : undefined,
          language: typeof message.language === 'string' ? message.language : undefined,
          thinkingLevel: normalizeThinkingLevel(message.thinkingLevel),
          purpose:
            message.purpose === 'meeting_notes' || message.purpose === 'conversation'
              ? message.purpose
              : undefined,
          transcriptionModel:
            typeof message.transcriptionModel === 'string' ? message.transcriptionModel : undefined,
          tts: typeof message.tts === 'boolean' ? message.tts : undefined,
        }
        const result = await executeVoiceTurnStreaming(app, request.headers, voiceTurnBody, {
          fallbackToken: token,
          signal: turnAbort.signal,
          onTranscript: (prepared) => {
            safeSend({
              type: 'voice.transcript.final',
              turnId,
              sessionId: prepared.chatPayload.sessionId ?? '',
              text: prepared.transcript,
              transcription: prepared.transcription,
            })
          },
          onAssistantDelta: (event) => {
            safeSend({
              type: 'voice.assistant.delta',
              turnId,
              sessionId: event.sessionId,
              text: event.text,
            })
          },
        })

        if (isVoiceTurnSuccess(result)) {
          const data = result.payload.data
          safeSend({
            type: 'voice.assistant.final',
            turnId,
            sessionId: data.sessionId,
            messageId: data.messageId,
            content: data.content,
          })
          if (data.audio) {
            safeSend({
              type: 'voice.audio',
              turnId,
              sessionId: data.sessionId,
              audio: data.audio,
            })
          }
          if (data.speech.error) {
            safeSend({
              type: 'voice.warning',
              turnId,
              warning: data.speech.error,
            })
          }
          safeSend({ type: 'voice.turn.done', turnId, result: data })
        } else {
          safeSend({
            type: 'voice.error',
            turnId,
            statusCode: result.statusCode,
            ...result.payload,
          })
        }
      } catch (error) {
        // A barge-in / socket close aborts the fetch; surface that as an
        // interruption rather than a failure.
        if (turnAbort.signal.aborted || (error instanceof Error && error.name === 'AbortError')) {
          safeSend({ type: 'voice.interrupted', turnId })
        } else {
          safeSend({
            type: 'voice.error',
            turnId,
            error: {
              code: 'VOICE_TURN_FAILED',
              message: error instanceof Error ? error.message : String(error),
            },
          })
        }
      } finally {
        busy = false
        if (currentTurnAbort === turnAbort) currentTurnAbort = null
      }
    })
  })

  app.post<{ Body: VoiceTurnBody }>(
    '/voice/turn',
    {
      preValidation: zodRequestValidation({
        body: {
          schema: voiceTurnRequestSchema,
          message: 'Invalid voice turn request body',
        },
      }),
    },
    async (request, reply) => {
      const body = request.body
      if (body.purpose === 'meeting_notes' && body.enqueue === true) {
        const invalidFile = validateQueuedVoiceFile(body)
        if (invalidFile) return reply.status(invalidFile.statusCode).send(invalidFile.payload)

        const sessionId = body.sessionId ?? randomUUID()
        const job = meetingQueueRepo.getOrCreateOpenJob({
          kind: MEETING_VOICE_JOB_KIND,
          concurrency: MEETING_QUEUE_DEFAULT_CONCURRENCY,
        })
        const item = meetingQueueRepo.appendItem(job.id, {
          request: queuedVoiceTurnBody(body, sessionId),
          sessionId,
          maxAttempts: MEETING_QUEUE_MAX_ATTEMPTS,
        })
        const position =
          meetingQueueRepo.positionOfItem(
            MEETING_VOICE_JOB_KIND,
            sessionId,
            item.jobId,
            item.idx,
          ) ?? 1
        runMeetingQueue(job.id, job.concurrency)
        return reply.status(202).send({
          data: {
            itemId: `${item.jobId}:${item.idx}`,
            sessionId,
            position,
            status: item.status,
          },
        })
      }
      const result = await executeVoiceTurn(app, request.headers, request.body)
      return reply.status(result.statusCode).send(result.payload)
    },
  )

  app.get<{ Params: { sessionId: string } }>(
    '/voice/meeting-queue/:sessionId',
    async (request) => {
      const params = request.params
      return {
        data: meetingQueueRepo.partitionedQueueStatus(
          MEETING_VOICE_JOB_KIND,
          params.sessionId,
        ),
      }
    },
  )

  app.post<{ Body: VoiceSpeechBody }>(
    '/voice/speech',
    {
      preValidation: zodRequestValidation({
        body: {
          schema: voiceSpeechRequestSchema,
          message: 'Invalid voice speech request body',
        },
      }),
    },
    async (request, reply) => {
      const result = await executeVoiceSpeech(app, request.body)
      return reply.status(result.statusCode).send(result.payload)
    },
  )
}
