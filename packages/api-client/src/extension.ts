import { readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { extname, join } from 'node:path'
import {
  DEFAULT_CHAT_OPTION_DEFAULTS,
  type ChatStreamOptions,
  DaemonClient,
  sanitizeChatOptions,
} from './daemon/http.js'

export interface ExtensionRuntimeConfig {
  baseUrl: string
  token: string | null
}

export async function createExtensionDaemonClient(): Promise<DaemonClient> {
  return new DaemonClient(await resolveExtensionRuntimeConfig())
}

export function buildExtensionStreamChatOptions(
  fileIds?: string[],
): ChatStreamOptions | undefined {
  return sanitizeChatOptions(
    {
      mode: DEFAULT_CHAT_OPTION_DEFAULTS.mode,
      fileIds,
    },
    DEFAULT_CHAT_OPTION_DEFAULTS,
  )
}

export async function resolveExtensionRuntimeConfig(): Promise<ExtensionRuntimeConfig> {
  await loadManagedExtensionEnv()
  return {
    baseUrl: process.env.SEPILOT_DAEMON_URL ?? 'http://127.0.0.1:17600',
    token: await loadExtensionDaemonToken(),
  }
}

export async function loadManagedExtensionEnv(): Promise<void> {
  const envPath = join(process.cwd(), '.sepilot', 'extension.env')

  try {
    const raw = await readFile(envPath, 'utf-8')
    for (const line of raw.split(/\r?\n/)) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith('#')) {
        continue
      }

      const separator = trimmed.indexOf('=')
      if (separator <= 0) {
        continue
      }

      const key = trimmed.slice(0, separator).trim()
      const value = trimmed.slice(separator + 1)
      if (key && !process.env[key]) {
        process.env[key] = value
      }
    }
  } catch {
    // Managed env file is optional.
  }
}

export async function loadExtensionDaemonToken(): Promise<string | null> {
  if (process.env.SEPILOT_DAEMON_TOKEN) {
    return process.env.SEPILOT_DAEMON_TOKEN
  }

  const tokenFile = process.env.SEPILOT_DAEMON_TOKEN_FILE
    ?? join(homedir(), '.sepilotd', 'security', 'daemon.token')

  try {
    return (await readFile(tokenFile, 'utf-8')).trim()
  } catch {
    return null
  }
}

export function guessExtensionMimeType(path: string): string {
  switch (extname(path).toLowerCase()) {
    case '.md':
      return 'text/markdown'
    case '.txt':
    case '.ts':
    case '.tsx':
    case '.js':
    case '.jsx':
    case '.json':
    case '.yml':
    case '.yaml':
    case '.css':
    case '.html':
    case '.xml':
    case '.sh':
    case '.sql':
    case '.toml':
    case '.py':
    case '.go':
    case '.rs':
    case '.java':
    case '.c':
    case '.cpp':
    case '.h':
    case '.rb':
    case '.php':
    case '.svg':
      return 'text/plain'
    case '.png':
      return 'image/png'
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg'
    case '.gif':
      return 'image/gif'
    case '.pdf':
      return 'application/pdf'
    default:
      return 'application/octet-stream'
  }
}
