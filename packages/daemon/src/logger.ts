import { appendFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import {
  DEFAULT_LOG_ROTATION,
  normalizeLogRotationOptions,
  rotateLogFileIfNeeded,
  type LogRotationOptions,
} from './utils/log-rotation.js'
import {
  createTraceRedactionContext,
  isTraceRedactionEnabled,
  redactSensitiveText,
} from './observability/trace-redaction.js'

export type LogLevel = 'debug' | 'info' | 'warn' | 'error'

const LEVEL_ORDER: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 }

// Check size every N writes instead of every write to keep the hot path
// cheap. With ~10 MiB rotation limit, the worst-case overshoot at this rate
// is well under a megabyte even for large log lines.
const ROTATION_CHECK_INTERVAL = 64

let currentLevel: LogLevel = 'info'
let logFilePath: string | null = null
let writeQueue: Promise<void> = Promise.resolve()
let jsonFormat = false
let rotationOptions: LogRotationOptions = { ...DEFAULT_LOG_ROTATION }
let writesSinceCheck = 0

export function setLogLevel(level: LogLevel): void {
  currentLevel = level
}

export function setJsonFormat(enabled: boolean): void {
  jsonFormat = enabled
}

export async function setLogFile(
  path: string,
  options?: Partial<LogRotationOptions>,
): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  logFilePath = path
  rotationOptions = normalizeLogRotationOptions(options, DEFAULT_LOG_ROTATION)
  writesSinceCheck = 0
}

export function setLogRotation(options: Partial<LogRotationOptions>): void {
  rotationOptions = normalizeLogRotationOptions(options, rotationOptions)
}

export function getLogRotation(): LogRotationOptions {
  return { ...rotationOptions }
}

function shouldLog(level: LogLevel): boolean {
  return LEVEL_ORDER[level] >= LEVEL_ORDER[currentLevel]
}

function formatMessage(level: LogLevel, component: string, message: string, data?: Record<string, unknown>): string {
  if (jsonFormat) {
    return JSON.stringify({ ts: new Date().toISOString(), level, component, msg: message, ...data })
  }
  const ts = new Date().toISOString()
  const dataStr = data ? ' ' + JSON.stringify(data) : ''
  return `${ts} [${level.toUpperCase().padEnd(5)}] [${component}] ${message}${dataStr}`
}

function writeToFile(line: string): void {
  if (!logFilePath) return
  const path = logFilePath
  const fileLine = isTraceRedactionEnabled()
    ? redactSensitiveText(line, createTraceRedactionContext(), {
        maxStringLength: Number.MAX_SAFE_INTEGER,
      })
    : line
  const shouldCheckRotation =
    rotationOptions.enabled && ++writesSinceCheck >= ROTATION_CHECK_INTERVAL
  if (shouldCheckRotation) writesSinceCheck = 0

  writeQueue = writeQueue.then(async () => {
    try {
      if (shouldCheckRotation) {
        await rotateLogFileIfNeeded(path, rotationOptions).catch(() => {})
      }
      await appendFile(path, fileLine + '\n', 'utf-8')
    } catch {
      // Logging must never throw into the calling code path.
    }
  })
}

export function createLogger(component: string) {
  return {
    debug: (msg: string, data?: Record<string, unknown>) => {
      if (!shouldLog('debug')) return
      const line = formatMessage('debug', component, msg, data)
      console.log(line)
      writeToFile(line)
    },
    info: (msg: string, data?: Record<string, unknown>) => {
      if (!shouldLog('info')) return
      const line = formatMessage('info', component, msg, data)
      console.log(line)
      writeToFile(line)
    },
    warn: (msg: string, data?: Record<string, unknown>) => {
      if (!shouldLog('warn')) return
      const line = formatMessage('warn', component, msg, data)
      console.warn(line)
      writeToFile(line)
    },
    error: (msg: string, data?: Record<string, unknown>) => {
      if (!shouldLog('error')) return
      const line = formatMessage('error', component, msg, data)
      console.error(line)
      writeToFile(line)
    },
  }
}
