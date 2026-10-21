export function isNodeFsError(err: unknown, code: string): boolean {
  return (
    typeof err === 'object'
    && err !== null
    && 'code' in err
    && (err as { code?: unknown }).code === code
  )
}
