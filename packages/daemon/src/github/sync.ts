import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { sepilotdHome } from '../storage/home.js'
import { secureDir, secureFile } from '../utils/secure-file.js'

export interface SyncRepo {
  fullName: string
  enabled: boolean
  lastSyncedAt: number | null
  lastSyncStatus?: 'success' | 'error' | null
  lastSyncError?: string | null
  pullRequests?: number
  issues?: number
  releases?: number
}

export interface SyncPolicy {
  intervalMin: number
  pullRequests: boolean
  issues: boolean
  releases: boolean
}

export interface DiscoveredRepo {
  fullName: string
  private: boolean
  defaultBranch: string | null
  updatedAt: string | null
  htmlUrl: string | null
}

export interface GitHubSyncResult {
  startedAt: number
  finishedAt: number
  total: number
  succeeded: number
  failed: number
  repos: SyncRepo[]
}

export interface GitHubRepoInspection {
  ok: true
  source: 'github-api'
  repo: {
    provider: 'GitHub'
    owner: string
    repo: string
    url: string
    webUrl: string
    pagesUrl: string
    description: string
    defaultBranch: string
    visibility: 'private' | 'public'
    stars: number
    forks: number
    openIssues: number
    pushedAt: string
    updatedAt: string
    links: {
      commits: string
      tags: string
      releases: string
      pages: string
    }
  }
  commits: Array<{
    sha: string
    message: string
    date: string
    url: string
  }>
  tags: Array<{
    name: string
    sha: string
    url: string
  }>
  releases: Array<{
    name: string
    tagName: string
    url: string
    publishedAt: string
    draft: boolean
    prerelease: boolean
  }>
}

interface GitHubRepoPayload {
  full_name?: string
  private?: boolean
  default_branch?: string | null
  description?: string | null
  stargazers_count?: number
  forks_count?: number
  open_issues_count?: number
  pushed_at?: string | null
  updated_at?: string | null
  html_url?: string | null
}

interface GitHubCommitPayload {
  sha?: string
  html_url?: string
  commit?: {
    message?: string
    author?: {
      date?: string
    } | null
  } | null
}

interface GitHubTagPayload {
  name?: string
  commit?: {
    sha?: string
  } | null
}

interface GitHubIssuePayload {
  number?: number
  title?: string
  state?: string
  html_url?: string
  updated_at?: string
  pull_request?: unknown
}

interface GitHubPullPayload {
  number?: number
  title?: string
  state?: string
  html_url?: string
  updated_at?: string
}

interface GitHubReleasePayload {
  id?: number
  tag_name?: string
  name?: string | null
  html_url?: string
  published_at?: string | null
  draft?: boolean
  prerelease?: boolean
}

function dir(): string {
  const d = join(sepilotdHome(), 'github')
  mkdirSync(d, { recursive: true, mode: 0o700 })
  secureDir(d)
  return d
}

function load<T>(name: string, fallback: T): T {
  const p = join(dir(), name)
  if (!existsSync(p)) return fallback
  try {
    return JSON.parse(readFileSync(p, 'utf-8')) as T
  } catch {
    return fallback
  }
}

function save<T>(name: string, value: T): void {
  const file = join(dir(), name)
  writeFileSync(file, JSON.stringify(value, null, 2), 'utf-8')
  secureFile(file)
}

function syncDataDir(): string {
  const d = join(dir(), 'sync-data')
  mkdirSync(d, { recursive: true, mode: 0o700 })
  secureDir(d)
  return d
}

function repoKey(fullName: string): string {
  return fullName.replace(/[^A-Za-z0-9_.-]+/g, '__')
}

function githubHeaders(token: string): HeadersInit {
  return {
    accept: 'application/vnd.github+json',
    authorization: `Bearer ${token}`,
    'user-agent': 'sepilotd',
    'x-github-api-version': '2022-11-28',
  }
}

function cleanText(value: unknown, maxLength = 4000): string {
  return typeof value === 'string' ? value.trim().slice(0, maxLength) : ''
}

async function githubJson<T>(token: string, path: string): Promise<T> {
  const response = await fetch(`https://api.github.com${path}`, {
    headers: githubHeaders(token),
  })
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(`GitHub request failed (${response.status}): ${text || response.statusText}`)
  }
  return response.json() as Promise<T>
}

function githubPagesUrl(owner: string, repo: string): string {
  const ownerHost = owner.toLowerCase()
  if (repo.toLowerCase() === `${ownerHost}.github.io`) {
    return `https://${ownerHost}.github.io/`
  }
  return `https://${ownerHost}.github.io/${encodeURIComponent(repo)}/`
}

function parseGitHubRepoRef(input: unknown): {
  owner: string
  repo: string
  webUrl: string
  pagesUrl: string
} {
  const record = input && typeof input === 'object' ? (input as Record<string, unknown>) : {}
  const explicitOwner = cleanText(record.owner, 200)
  const explicitRepo = cleanText(record.repo, 200)
  if (explicitOwner && explicitRepo) {
    return {
      owner: explicitOwner,
      repo: explicitRepo,
      webUrl: `https://github.com/${explicitOwner}/${explicitRepo}`,
      pagesUrl: githubPagesUrl(explicitOwner, explicitRepo),
    }
  }

  const rawUrl = cleanText(record.url ?? input, 2048)
  if (!rawUrl) throw new Error('GitHub repository URL is required.')

  const ssh = /^git@([^:]+):(.+)$/iu.exec(rawUrl)
  let host = ''
  let parts: string[] = []
  if (ssh) {
    host = ssh[1] ?? ''
    parts = (ssh[2] ?? '')
      .replace(/\.git$/iu, '')
      .split('/')
      .filter(Boolean)
  } else {
    const parsed = new URL(rawUrl)
    host = parsed.hostname
    parts = parsed.pathname
      .replace(/^\//u, '')
      .replace(/\.git$/iu, '')
      .split('/')
      .filter(Boolean)
  }

  if (!host.toLowerCase().includes('github')) {
    throw new Error('GitHub repository URL must use github.com.')
  }
  const owner = parts[0] ?? ''
  const repo = parts[1] ?? ''
  if (!owner || !repo) throw new Error('GitHub repository URL must include owner and repo.')
  return {
    owner,
    repo,
    webUrl: `https://github.com/${owner}/${repo}`,
    pagesUrl: githubPagesUrl(owner, repo),
  }
}

function githubRepoApiPath(owner: string, repo: string, suffix = ''): string {
  return `/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}${suffix}`
}

function normalizeRepo(input: SyncRepo): SyncRepo {
  return {
    fullName: input.fullName,
    enabled: input.enabled,
    lastSyncedAt: input.lastSyncedAt ?? null,
    lastSyncStatus: input.lastSyncStatus ?? null,
    lastSyncError: input.lastSyncError ?? null,
    pullRequests: input.pullRequests ?? 0,
    issues: input.issues ?? 0,
    releases: input.releases ?? 0,
  }
}

function summarizeIssue(issue: GitHubIssuePayload) {
  return {
    number: issue.number ?? null,
    title: issue.title ?? '',
    state: issue.state ?? '',
    htmlUrl: issue.html_url ?? '',
    updatedAt: issue.updated_at ?? null,
  }
}

function summarizePull(pull: GitHubPullPayload) {
  return {
    number: pull.number ?? null,
    title: pull.title ?? '',
    state: pull.state ?? '',
    htmlUrl: pull.html_url ?? '',
    updatedAt: pull.updated_at ?? null,
  }
}

function summarizeRelease(release: GitHubReleasePayload) {
  return {
    id: release.id ?? null,
    tagName: release.tag_name ?? '',
    name: release.name ?? release.tag_name ?? '',
    htmlUrl: release.html_url ?? '',
    publishedAt: release.published_at ?? null,
  }
}

function persistRepoData(fullName: string, payload: unknown): void {
  const file = join(syncDataDir(), `${repoKey(fullName)}.json`)
  writeFileSync(file, JSON.stringify(payload, null, 2), 'utf-8')
  secureFile(file)
}

export function listRepos(): SyncRepo[] {
  return load<SyncRepo[]>('repos.json', []).map(normalizeRepo)
}

export function setRepos(next: SyncRepo[]): void {
  save('repos.json', next.map(normalizeRepo))
}

export function getPolicy(): SyncPolicy {
  return load<SyncPolicy>('policy.json', {
    intervalMin: 60,
    pullRequests: true,
    issues: false,
    releases: false,
  })
}

export function setPolicy(next: SyncPolicy): void {
  save('policy.json', next)
}

export async function discoverRepos(token: string): Promise<DiscoveredRepo[]> {
  const repos = await githubJson<GitHubRepoPayload[]>(
    token,
    '/user/repos?per_page=100&sort=updated&affiliation=owner,collaborator,organization_member',
  )
  return repos
    .filter((repo) => Boolean(repo.full_name))
    .map((repo) => ({
      fullName: repo.full_name as string,
      private: Boolean(repo.private),
      defaultBranch: repo.default_branch ?? null,
      updatedAt: repo.updated_at ?? null,
      htmlUrl: repo.html_url ?? null,
    }))
}

export async function inspectRepo(token: string, input: unknown): Promise<GitHubRepoInspection> {
  const ref = parseGitHubRepoRef(input)
  const [repo, commits, tags, releases] = await Promise.all([
    githubJson<GitHubRepoPayload>(token, githubRepoApiPath(ref.owner, ref.repo)),
    githubJson<GitHubCommitPayload[]>(
      token,
      githubRepoApiPath(ref.owner, ref.repo, '/commits?per_page=5'),
    ),
    githubJson<GitHubTagPayload[]>(
      token,
      githubRepoApiPath(ref.owner, ref.repo, '/tags?per_page=5'),
    ),
    githubJson<GitHubReleasePayload[]>(
      token,
      githubRepoApiPath(ref.owner, ref.repo, '/releases?per_page=5'),
    ),
  ])

  return {
    ok: true,
    source: 'github-api',
    repo: {
      provider: 'GitHub',
      owner: ref.owner,
      repo: ref.repo,
      url: ref.webUrl,
      webUrl: ref.webUrl,
      pagesUrl: ref.pagesUrl,
      description: cleanText(repo.description, 500),
      defaultBranch: cleanText(repo.default_branch, 120),
      visibility: repo.private === true ? 'private' : 'public',
      stars: Number(repo.stargazers_count ?? 0),
      forks: Number(repo.forks_count ?? 0),
      openIssues: Number(repo.open_issues_count ?? 0),
      pushedAt: cleanText(repo.pushed_at, 80),
      updatedAt: cleanText(repo.updated_at, 80),
      links: {
        commits: `${ref.webUrl}/commits`,
        tags: `${ref.webUrl}/tags`,
        releases: `${ref.webUrl}/releases`,
        pages: ref.pagesUrl,
      },
    },
    commits: commits.map((commit) => ({
      sha: cleanText(commit.sha, 80),
      message: cleanText(commit.commit?.message, 500).split(/\r?\n/u)[0] ?? '',
      date: cleanText(commit.commit?.author?.date, 80),
      url: cleanText(commit.html_url, 2048),
    })),
    tags: tags.map((tag) => ({
      name: cleanText(tag.name, 200),
      sha: cleanText(tag.commit?.sha, 80),
      url: `${ref.webUrl}/releases/tag/${encodeURIComponent(cleanText(tag.name, 200))}`,
    })),
    releases: releases.map((release) => ({
      name: cleanText(release.name, 300),
      tagName: cleanText(release.tag_name, 200),
      url: cleanText(release.html_url, 2048),
      publishedAt: cleanText(release.published_at, 80),
      draft: release.draft === true,
      prerelease: release.prerelease === true,
    })),
  }
}

export async function runSync(token: string): Promise<GitHubSyncResult> {
  const startedAt = Date.now()
  const policy = getPolicy()
  const repos = listRepos()
  const nextRepos: SyncRepo[] = []
  let succeeded = 0
  let failed = 0

  for (const repo of repos) {
    if (!repo.enabled) {
      nextRepos.push(repo)
      continue
    }
    const data: {
      fullName: string
      syncedAt: number
      policy: SyncPolicy
      pullRequests: ReturnType<typeof summarizePull>[]
      issues: ReturnType<typeof summarizeIssue>[]
      releases: ReturnType<typeof summarizeRelease>[]
    } = {
      fullName: repo.fullName,
      syncedAt: Date.now(),
      policy,
      pullRequests: [],
      issues: [],
      releases: [],
    }
    try {
      if (policy.pullRequests) {
        const pulls = await githubJson<GitHubPullPayload[]>(
          token,
          `/repos/${repo.fullName}/pulls?state=open&per_page=50`,
        )
        data.pullRequests = pulls.map(summarizePull)
      }
      if (policy.issues) {
        const issues = await githubJson<GitHubIssuePayload[]>(
          token,
          `/repos/${repo.fullName}/issues?state=open&per_page=50`,
        )
        data.issues = issues.filter((issue) => !issue.pull_request).map(summarizeIssue)
      }
      if (policy.releases) {
        const releases = await githubJson<GitHubReleasePayload[]>(
          token,
          `/repos/${repo.fullName}/releases?per_page=20`,
        )
        data.releases = releases.map(summarizeRelease)
      }
      persistRepoData(repo.fullName, data)
      succeeded += 1
      nextRepos.push({
        ...repo,
        lastSyncedAt: data.syncedAt,
        lastSyncStatus: 'success',
        lastSyncError: null,
        pullRequests: data.pullRequests.length,
        issues: data.issues.length,
        releases: data.releases.length,
      })
    } catch (error) {
      failed += 1
      nextRepos.push({
        ...repo,
        lastSyncedAt: Date.now(),
        lastSyncStatus: 'error',
        lastSyncError: error instanceof Error ? error.message : 'GitHub sync failed',
      })
    }
  }

  setRepos(nextRepos)
  return {
    startedAt,
    finishedAt: Date.now(),
    total: repos.filter((repo) => repo.enabled).length,
    succeeded,
    failed,
    repos: nextRepos,
  }
}
