export function isSameCredentialOrigin(
  existingEndpoint: string | undefined,
  nextEndpoint: string | undefined,
): boolean {
  const existing = existingEndpoint?.trim() || undefined
  const next = nextEndpoint?.trim() || undefined
  if (existing === undefined || next === undefined) return existing === next
  try {
    const existingOrigin = new URL(existing).origin
    if (existingOrigin === 'null') return existing === next
    return existingOrigin === new URL(next).origin
  } catch {
    return existing === next
  }
}
