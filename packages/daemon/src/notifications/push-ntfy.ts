import { createLogger } from '../logger.js'

const logger = createLogger('ntfy-push')

export interface NtfyRelayOptions {
  url: string
  topic: string
  token?: string
  fetchImpl?: typeof fetch
}

export interface NtfyNotification {
  title: string
  body?: string
  url?: string | null
}

function stripTrailingSlash(value: string): string {
  return value.replace(/\/+$/, '')
}

function isAsciiHeaderValue(value: string): boolean {
  return /^[\x20-\x7e]*$/.test(value)
}

function bodyForNtfy(input: NtfyNotification): string {
  const body = input.body?.trim() ?? ''
  if (isAsciiHeaderValue(input.title)) return body || input.title
  return [input.title, body].filter(Boolean).join('\n\n')
}

export class NtfyRelay {
  private readonly url: string
  private readonly topic: string
  private readonly token?: string
  private readonly fetchImpl: typeof fetch

  constructor(options: NtfyRelayOptions) {
    this.url = stripTrailingSlash(options.url)
    this.topic = options.topic
    this.token = options.token
    this.fetchImpl = options.fetchImpl ?? fetch
  }

  endpoint(): string {
    return `${this.url}/${encodeURIComponent(this.topic)}`
  }

  async publish(input: NtfyNotification): Promise<void> {
    const headers = new Headers()
    if (isAsciiHeaderValue(input.title)) headers.set('Title', input.title)
    if (input.url && isAsciiHeaderValue(input.url)) headers.set('Click', input.url)
    if (this.token) headers.set('Authorization', `Bearer ${this.token}`)

    try {
      const response = await this.fetchImpl(this.endpoint(), {
        method: 'POST',
        headers,
        body: bodyForNtfy(input),
      })
      if (!response.ok) {
        logger.warn('ntfy relay rejected notification', { status: response.status })
      }
    } catch (error) {
      logger.warn('ntfy relay unreachable', {
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }
}

export function ntfyRelayFromEnv(env: NodeJS.ProcessEnv): NtfyRelay | null {
  const url = env.SEPILOTD_NTFY_URL?.trim()
  const topic = env.SEPILOTD_NTFY_TOPIC?.trim()
  if (!url || !topic) return null
  return new NtfyRelay({
    url,
    topic,
    token: env.SEPILOTD_NTFY_TOKEN?.trim() || undefined,
  })
}
