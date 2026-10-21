const SENSITIVE_QUERY_KEY = /(token|ticket|secret|password|passwd|signature|sig|key|auth|code|credential)/i
const PATH_CAPABILITY_SEGMENT = /^(\/(?:api\/v1\/)?share\/)[^/?#]+/

export function redactUrlForLog(url: string): string {
  const queryStart = url.indexOf('?')
  const path = queryStart >= 0 ? url.slice(0, queryStart) : url
  const redactedPath = path.replace(PATH_CAPABILITY_SEGMENT, '$1[redacted]')
  if (queryStart < 0) return redactedPath
  const params = url.slice(queryStart + 1).split('&').map((pair) => {
    const separator = pair.indexOf('=')
    if (separator < 0) return pair
    const rawKey = pair.slice(0, separator)
    let key = rawKey
    try { key = decodeURIComponent(rawKey) } catch {}
    return SENSITIVE_QUERY_KEY.test(key) ? `${rawKey}=[redacted]` : pair
  })
  return `${redactedPath}?${params.join('&')}`
}
