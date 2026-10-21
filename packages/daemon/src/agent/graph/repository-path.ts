export function normalizeRepositoryPath(path: string): string {
  return path
    .trim()
    .replace(/\\/g, '/')
    .replace(/^\/testbed\//, '')
    .replace(/^\.\//, '')
    .replace(/^\/+/, '')
}

export function searchPathMatchesReadPath(searchPath: string, readPath: string): boolean {
  const normalizedSearchPath = normalizeRepositoryPath(searchPath)
  const normalizedReadPath = normalizeRepositoryPath(readPath)
  return normalizedSearchPath === normalizedReadPath
    || normalizedReadPath.endsWith(`/${normalizedSearchPath}`)
    || normalizedSearchPath.endsWith(`/${normalizedReadPath}`)
}
