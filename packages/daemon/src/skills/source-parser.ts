import type { SkillRef } from './sources/types.js'

export function parseSkillSource(input: string): SkillRef {
  const trimmed = input.trim()
  if (!trimmed) throw new Error('empty source')

  if (/^(git|http):\/\//i.test(trimmed) && !/^https:\/\//i.test(trimmed)) {
    throw new Error('Only https:// URLs are supported')
  }

  if (/\.md$/i.test(trimmed) && /^https:\/\//i.test(trimmed)) {
    return { type: 'url', url: trimmed }
  }

  const treeMatch = trimmed.match(
    /^https:\/\/github\.com\/([^/]+)\/([^/]+)\/tree\/([^/]+)\/(.+)$/i,
  )
  if (treeMatch) {
    const [, owner, repo, branch, path] = treeMatch
    return {
      type: 'git',
      repo: `https://github.com/${owner}/${repo}`,
      branch,
      path,
    }
  }

  if (
    /^https:\/\/.+\.git$/i.test(trimmed) ||
    /^https:\/\/github\.com\/[^/]+\/[^/]+\/?$/i.test(trimmed) ||
    /^git@/i.test(trimmed)
  ) {
    return { type: 'git', repo: trimmed, branch: null, path: null }
  }

  const qualifiedMatch = trimmed.match(/^([a-z0-9_-]+)\/([a-z0-9_.-]+)$/i)
  if (qualifiedMatch) {
    return { type: 'marketplace', marketplace: qualifiedMatch[1], name: qualifiedMatch[2] }
  }

  if (/^[a-z0-9_.-]+$/i.test(trimmed)) {
    return { type: 'marketplace', marketplace: null, name: trimmed }
  }

  throw new Error(`Unrecognised source: ${trimmed}`)
}
