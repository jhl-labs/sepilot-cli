const HTTP_HEADER_NAME = /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/

export function parseProviderHeadersInput(value: string): Record<string, string> {
  const trimmed = value.trim()
  if (!trimmed) return {}

  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch {
    throw new Error(
      'Custom headers must be a JSON object, for example {"X-API-Key":"${MY_API_KEY}"}.',
    )
  }

  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('Custom headers must be a JSON object with string values.')
  }

  const headers: Record<string, string> = {}
  for (const [rawName, rawValue] of Object.entries(parsed)) {
    const name = rawName.trim()
    if (!name || !HTTP_HEADER_NAME.test(name)) {
      throw new Error(`Invalid HTTP header name: ${rawName || '(empty)'}.`)
    }
    if (typeof rawValue !== 'string') {
      throw new Error(`Custom header ${name} must have a string value.`)
    }
    if (/[\r\n]/.test(rawValue)) {
      throw new Error(`Custom header ${name} must not contain line breaks.`)
    }
    headers[name] = rawValue
  }

  return headers
}

export function formatProviderHeadersInput(
  headers: Record<string, string> | null | undefined,
): string {
  const entries = Object.entries(headers ?? {})
  return entries.length > 0 ? JSON.stringify(Object.fromEntries(entries)) : ''
}

export function resolveProviderHeaderEnvReferences(
  headers: Record<string, string>,
  env: Record<string, string | undefined>,
): Record<string, string> {
  return Object.fromEntries(
    Object.entries(headers).map(([name, value]) => [
      name,
      value.replace(/\$\{([^}]+)\}/g, (match, variableName: string) => env[variableName] ?? match),
    ]),
  )
}
