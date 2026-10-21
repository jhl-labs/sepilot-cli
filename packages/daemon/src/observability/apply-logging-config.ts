import type { LoggingConfig } from '../config/schema.js'
import {
  setJsonFormat,
  setLogLevel,
  setLogRotation,
  type LogLevel,
} from '../logger.js'
import { setAgentTraceRotation } from './agent-trace.js'

const ALLOWED_LEVELS: ReadonlyArray<LogLevel> = ['debug', 'info', 'warn', 'error']

export function applyLoggingConfig(
  config: LoggingConfig,
  options: { envLogLevel?: string } = {},
): void {
  const envLevel = options.envLogLevel
  if (envLevel && (ALLOWED_LEVELS as ReadonlyArray<string>).includes(envLevel)) {
    setLogLevel(envLevel as LogLevel)
  } else {
    setLogLevel(config.level)
  }
  setJsonFormat(config.jsonFormat)
  setLogRotation(config.app)
  setAgentTraceRotation(config.trace)
}
