import { stat } from 'node:fs/promises'
import { extname } from 'node:path'
import {
  FileExtractionError,
  isOcrImageFilePath,
  isPdfFilePath,
  isPptxFilePath,
  isTextFilePath,
  OCR_DEFAULT_MAX_PAGES,
  readExtractableFileContent,
} from '../media/pipeline.js'
import {
  isSpeechBinaryMissingError,
  synthesizeSpeech,
  transcribeAudio,
  type SpeechToolDeps,
} from '../media/speech.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'
import { resolveToolPath } from './path-utils.js'

function truncateText(value: string, max: number): string {
  return value.length > max
    ? `${value.slice(0, max - 1)}…`
    : value
}

// Always cap the number of PDF pages OCR'd, independent of caller input, so the
// tool never fans out an unbounded number of rasterization/OCR subprocesses.
function getOcrMaxPages(): number {
  const raw = Number(process.env.SEPILOTD_OCR_MAX_PAGES)
  return Number.isFinite(raw) && raw >= 1 ? Math.min(200, Math.trunc(raw)) : OCR_DEFAULT_MAX_PAGES
}

function extractionErrorCode(error: unknown): string | undefined {
  if (!(error instanceof FileExtractionError)) return undefined
  switch (error.code) {
    case 'FILE_TOO_LARGE':
      return 'FILE_TOO_LARGE_PERMANENT'
    case 'BINARY_CONTENT':
    case 'UNSUPPORTED_FILE_TYPE':
    case 'INVALID_DOCUMENT':
    case 'ARCHIVE_LIMIT_EXCEEDED':
      return 'INVALID_INPUT_PERMANENT'
    default:
      return undefined
  }
}

function classifyMediaKind(path: string): 'text' | 'pdf' | 'presentation' | 'image' | 'binary' {
  if (isTextFilePath(path)) {
    return 'text'
  }
  if (isPdfFilePath(path)) {
    return 'pdf'
  }
  if (isPptxFilePath(path)) {
    return 'presentation'
  }
  if (isOcrImageFilePath(path)) {
    return 'image'
  }
  return 'binary'
}

export function createMediaInspectTool(): ToolDefinitionRuntime {
  return {
    name: 'media.inspect',
    description:
      'Inspect a file and report whether it is text, PDF, PowerPoint, OCR-capable image, or generic binary.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'filesystem' },
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path to inspect.' },
      },
      required: ['path'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const path = typeof input.path === 'string' && input.path.trim()
        ? resolveToolPath(input.path.trim(), context?.cwd)
        : ''
      if (!path) {
        return {
          output: 'path is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }

      try {
        const info = await stat(path)
        const kind = classifyMediaKind(path)
        return {
          output: JSON.stringify({
            path,
            extension: extname(path).toLowerCase(),
            sizeBytes: info.size,
            kind,
            textExtractable:
              kind === 'text' || kind === 'pdf' || kind === 'presentation' || kind === 'image',
            requiresOcr: kind === 'image',
          }, null, 2),
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return {
          output: message,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
    },
  }
}

export function createMediaTranscribeTool(deps?: SpeechToolDeps): ToolDefinitionRuntime {
  return {
    name: 'media.transcribe',
    description:
      'Transcribe a local audio file (voice note, recording, etc.) to text using a locally-installed whisper CLI '
      + '(pip install openai-whisper) — no external paid API is called. Use when the user references an audio file '
      + 'or when a voice message needs to become text. Pass `language` (ISO 639-1 like "ko"/"en") to skip auto-detect. '
      + 'If the whisper binary is not installed it returns an error explaining how to set SEPILOTD_WHISPER_BIN.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'media.speech' },
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Path to the audio file to transcribe.' },
        language: { type: 'string', description: 'Optional ISO 639-1 language hint (e.g. "ko", "en").' },
        model: { type: 'string', description: 'Optional whisper model name (e.g. "base", "small", "medium"). Defaults to env or "base".' },
      },
      required: ['path'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const path = typeof input.path === 'string' && input.path.trim()
        ? resolveToolPath(input.path.trim(), context?.cwd)
        : ''
      if (!path) {
        return { output: 'path is required', status: 'error', code: 'INVALID_INPUT_PERMANENT', durationMs: Date.now() - start }
      }
      try {
        await stat(path)
      } catch {
        return { output: `audio file not found: ${path}`, status: 'error', code: 'ENOENT_PERMANENT', durationMs: Date.now() - start }
      }
      try {
        const result = await transcribeAudio(
          {
            audioPath: path,
            language: typeof input.language === 'string' ? input.language : undefined,
            model: typeof input.model === 'string' ? input.model : undefined,
          },
          deps,
        )
        return {
          output: JSON.stringify({
            path,
            model: result.model,
            language: result.language ?? null,
            chars: result.text.length,
            text: result.text,
          }),
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return {
          output: message,
          status: 'error',
          // Missing-binary is a permanent config issue, not a transient failure.
          code: isSpeechBinaryMissingError(error) ? 'SPEECH_BINARY_MISSING_PERMANENT' : undefined,
          durationMs: Date.now() - start,
        }
      }
    },
  }
}

export function createMediaSpeakTool(deps?: SpeechToolDeps): ToolDefinitionRuntime {
  return {
    name: 'media.speak',
    description:
      'Synthesize speech from text and write it to a local WAV file using a locally-installed `piper` CLI '
      + '(https://github.com/rhasspy/piper) — no external paid API. Use when the user asks you to read text aloud, '
      + 'produce a voice clip, or generate narration. Pass `outputPath` to choose the destination, or omit it to get '
      + 'a temp file path back. Requires SEPILOTD_PIPER_MODEL (a .onnx voice model); if it is unset the tool returns '
      + 'an error explaining how to set it.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'media.speech' },
    inputSchema: {
      type: 'object',
      properties: {
        text: { type: 'string', description: 'The text to speak.' },
        outputPath: { type: 'string', description: 'Optional destination .wav path. A temp file is used when omitted.' },
      },
      required: ['text'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const text = typeof input.text === 'string' ? input.text.trim() : ''
      if (!text) {
        return { output: 'text is required', status: 'error', code: 'INVALID_INPUT_PERMANENT', durationMs: Date.now() - start }
      }
      try {
        const result = await synthesizeSpeech(
          {
            text,
            outputPath:
              typeof input.outputPath === 'string' && input.outputPath.trim()
                ? resolveToolPath(input.outputPath, context?.cwd)
                : undefined,
          },
          deps,
        )
        return {
          output: JSON.stringify({
            audioPath: result.audioPath,
            bytes: result.bytes,
            temporary: result.temporary,
          }),
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return {
          output: message,
          status: 'error',
          code: isSpeechBinaryMissingError(error) ? 'SPEECH_BINARY_MISSING_PERMANENT' : undefined,
          durationMs: Date.now() - start,
        }
      }
    },
  }
}

export function createMediaExtractTextTool(): ToolDefinitionRuntime {
  return {
    name: 'media.extract_text',
    description: 'Extract readable text from text files, PDFs, and OCR-capable images.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'filesystem' },
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path to extract text from.' },
        allowOcr: { type: 'boolean', description: 'Allow OCR fallback for images and scanned PDFs. Defaults to true.' },
        maxChars: { type: 'number', description: 'Maximum number of characters to return. Defaults to 12000.' },
      },
      required: ['path'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const path = typeof input.path === 'string' && input.path.trim()
        ? resolveToolPath(input.path.trim(), context?.cwd)
        : ''
      if (!path) {
        return {
          output: 'path is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }

      const maxChars = typeof input.maxChars === 'number' && Number.isFinite(input.maxChars)
        ? Math.max(200, Math.min(200_000, Math.trunc(input.maxChars)))
        : 12_000

      try {
        const text = await readExtractableFileContent(path, {
          allowOcr: input.allowOcr !== false,
          ocrMaxPages: getOcrMaxPages(),
        })
        const normalized = text.trim()
        const truncated = truncateText(normalized, maxChars)
        return {
          output: truncated.length === normalized.length
            ? truncated
            : `${truncated}\n\n[truncated to ${maxChars} chars]`,
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return {
          output: message,
          status: 'error',
          code: extractionErrorCode(error),
          durationMs: Date.now() - start,
        }
      }
    },
  }
}
