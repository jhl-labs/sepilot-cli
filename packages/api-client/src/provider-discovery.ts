export const DEFAULT_OPENAI_COMPATIBLE_BASE_URL = 'http://localhost:11434/v1'

function openAiCompatibleModelsUrl(baseUrl: string): string {
  return `${baseUrl.trim().replace(/\/+$/, '')}/models`
}

function parseOpenAiCompatibleModels(payload: unknown): string[] {
  if (
    typeof payload !== 'object'
    || payload === null
    || !('data' in payload)
    || !Array.isArray((payload as { data?: unknown }).data)
  ) {
    return []
  }

  return Array.from(new Set(
    (payload as { data: unknown[] }).data
      .map((entry) => (
        typeof entry === 'object'
        && entry !== null
        && typeof (entry as { id?: unknown }).id === 'string'
          ? (entry as { id: string }).id.trim()
          : ''
      ))
      .filter(Boolean),
  ))
}

export async function discoverOpenAiCompatibleModels(
  baseUrl: string,
  options: {
    apiKey?: string | null
    headers?: Record<string, string> | null
    timeoutMs?: number
  } = {},
): Promise<string[]> {
  const url = openAiCompatibleModelsUrl(baseUrl)
  const headers: Record<string, string> = { ...(options.headers ?? {}) }
  const headerNames = new Set(Object.keys(headers).map((name) => name.toLowerCase()))
  if (!headerNames.has('accept')) {
    headers.accept = 'application/json'
  }
  const apiKey = options.apiKey?.trim()
  if (apiKey && !headerNames.has('authorization')) {
    headers.authorization = `Bearer ${apiKey}`
  }

  try {
    const response = await fetch(url, {
      headers,
      signal: AbortSignal.timeout(options.timeoutMs ?? 2500),
    })
    if (!response.ok) {
      return []
    }
    return parseOpenAiCompatibleModels(await response.json())
  } catch {
    return []
  }
}
