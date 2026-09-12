// Pure repository-path utilities pulled out of `graph/nodes.ts`.
// Some harnesses mount the target repository at `/testbed/<path>`, while
// agent tool inputs may come back as absolute or `./relative` paths and
// search outputs are usually pinned to the repo root. These helpers
// normalize those shapes into one comparable form so
// "did we already read this file?" / "does this search hit refer
// to that file?" don't false-negative on cosmetic prefix
// differences.

/**
 * Normalize a path string into the repository-relative form the
 * agent compares against: backslash → forward slash, `/testbed/`
 * prefix dropped (common harness mount), `./` prefix dropped,
 * leading slashes dropped.
 */
export function normalizeRepositoryPath(path: string): string {
  return path
    .trim()
    .replace(/\\/g, '/')
    .replace(/^\/testbed\//, '')
    .replace(/^\.\//, '')
    .replace(/^\/+/, '')
}

/**
 * True when two paths likely refer to the same file in the
 * repository. Either side may be longer (e.g. fully qualified
 * package path vs bare filename); we accept matches in either
 * direction with a `/` boundary so `models/foo.py` matches
 * `pkg/models/foo.py` but not `unrelated_models/foo.py`.
 */
export function searchPathMatchesReadPath(searchPath: string, readPath: string): boolean {
  const normalizedSearchPath = normalizeRepositoryPath(searchPath)
  const normalizedReadPath = normalizeRepositoryPath(readPath)
  return normalizedSearchPath === normalizedReadPath
    || normalizedReadPath.endsWith(`/${normalizedSearchPath}`)
    || normalizedSearchPath.endsWith(`/${normalizedReadPath}`)
}

/**
 * Extract an explicit `path.py:line` reference from a task
 * description, or a Python module path of the form
 * `package.module.submodule:line`. Returns null when neither shape
 * is present.
 *
 * The module-path branch lower-cases the leading segment because
 * Bug traces often capitalize the project root while the actual file
 * lives under a lower-cased directory.
 */
