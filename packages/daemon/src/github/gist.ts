import type { Snippet, SnippetGistLink } from '../snippets/schema.js'

const GITHUB_API = 'https://api.github.com'

interface GitHubGistFile {
  filename: string
  language?: string | null
  content?: string | null
}

interface GitHubGist {
  id: string
  html_url: string | null
  updated_at: string | null
  files: Record<string, GitHubGistFile>
}

export interface PulledGistFile {
  link: SnippetGistLink
  title: string
  language: string
  body: string
}

export class GitHubGistError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'GitHubGistError'
  }
}

function headers(token: string): HeadersInit {
  return {
    accept: 'application/vnd.github+json',
    authorization: `Bearer ${token}`,
    'content-type': 'application/json',
    'user-agent': 'sepilotd',
    'x-github-api-version': '2022-11-28',
  }
}

async function githubJson<T>(
  token: string,
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${GITHUB_API}${path}`, {
    ...init,
    headers: {
      ...headers(token),
      ...(init.headers ?? {}),
    },
  })
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    let message = text.trim()
    try {
      const parsed = JSON.parse(text) as { message?: string }
      message = parsed.message ?? message
    } catch {
      /* keep raw text */
    }
    throw new GitHubGistError(message || `GitHub Gist request failed (${response.status})`, response.status)
  }
  return response.json() as Promise<T>
}

function normalizeLanguage(language: string): string {
  return language.trim().toLowerCase() || 'plaintext'
}

function extensionForLanguage(language: string): string {
  switch (normalizeLanguage(language)) {
    case 'bash':
    case 'shell':
    case 'sh':
      return 'sh'
    case 'powershell':
    case 'pwsh':
    case 'ps1':
      return 'ps1'
    case 'javascript':
    case 'js':
      return 'js'
    case 'typescript':
    case 'ts':
      return 'ts'
    case 'tsx':
      return 'tsx'
    case 'jsx':
      return 'jsx'
    case 'python':
    case 'py':
      return 'py'
    case 'dockerfile':
      return 'Dockerfile'
    case 'markdown':
    case 'md':
      return 'md'
    case 'yaml':
    case 'yml':
      return 'yaml'
    case 'json':
      return 'json'
    case 'sql':
      return 'sql'
    case 'html':
      return 'html'
    case 'css':
      return 'css'
    default:
      return 'txt'
  }
}

function languageFromFile(file: GitHubGistFile): string {
  const filename = file.filename.toLowerCase()
  const apiLanguage = file.language?.toLowerCase()
  if (apiLanguage === 'shell') return 'bash'
  if (apiLanguage === 'javascript') return 'js'
  if (apiLanguage === 'typescript') return filename.endsWith('.tsx') ? 'tsx' : 'ts'
  if (apiLanguage === 'python') return 'python'
  if (filename.endsWith('.ps1')) return 'powershell'
  if (filename.endsWith('.sh')) return 'bash'
  if (filename.endsWith('.tsx')) return 'tsx'
  if (filename.endsWith('.ts')) return 'ts'
  if (filename.endsWith('.jsx')) return 'jsx'
  if (filename.endsWith('.js')) return 'js'
  if (filename.endsWith('.md')) return 'markdown'
  if (filename.endsWith('.yml') || filename.endsWith('.yaml')) return 'yaml'
  if (filename.endsWith('.py')) return 'python'
  if (filename.endsWith('.sql')) return 'sql'
  if (filename.endsWith('.html')) return 'html'
  if (filename.endsWith('.css')) return 'css'
  if (filename.endsWith('.json')) return 'json'
  return apiLanguage ?? 'plaintext'
}

function titleFromFilename(filename: string): string {
  return filename.replace(/\.[^.]+$/, '').replace(/[-_]+/g, ' ').trim() || filename
}

function filenameForSnippet(snippet: Snippet): string {
  const ext = extensionForLanguage(snippet.language)
  if (snippet.gist?.file) return snippet.gist.file
  if (ext === 'Dockerfile') return 'Dockerfile'
  const base = snippet.title
    .trim()
    .replace(/\.[A-Za-z0-9]+$/, '')
    .replace(/[^A-Za-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    || 'snippet'
  return `${base}.${ext}`
}

function pickFile(gist: GitHubGist, preferredFile?: string | null): GitHubGistFile {
  if (preferredFile && gist.files[preferredFile]) return gist.files[preferredFile]
  const first = Object.values(gist.files)[0]
  if (!first) throw new GitHubGistError('Gist has no files.', 422)
  return first
}

function linkFromGist(gist: GitHubGist, file: GitHubGistFile): SnippetGistLink {
  return {
    id: gist.id,
    file: file.filename,
    htmlUrl: gist.html_url,
    updatedAt: gist.updated_at,
    syncedAt: Date.now(),
  }
}

export async function pushSnippetToGist(
  token: string,
  snippet: Snippet,
): Promise<SnippetGistLink> {
  const filename = filenameForSnippet(snippet)
  const body = {
    description: `sepilotd snippet: ${snippet.title}`,
    public: false,
    files: {
      [filename]: {
        content: snippet.body,
      },
    },
  }
  const gist = snippet.gist?.id
    ? await githubJson<GitHubGist>(
        token,
        `/gists/${encodeURIComponent(snippet.gist.id)}`,
        { method: 'PATCH', body: JSON.stringify(body) },
      )
    : await githubJson<GitHubGist>(
        token,
        '/gists',
        { method: 'POST', body: JSON.stringify(body) },
      )
  return linkFromGist(gist, pickFile(gist, filename))
}

export async function pullGistFile(
  token: string,
  gistId: string,
  preferredFile?: string | null,
): Promise<PulledGistFile> {
  const gist = await githubJson<GitHubGist>(
    token,
    `/gists/${encodeURIComponent(gistId)}`,
    { method: 'GET' },
  )
  const file = pickFile(gist, preferredFile)
  if (file.content == null) {
    throw new GitHubGistError(`Gist file has no content: ${file.filename}`, 422)
  }
  return {
    link: linkFromGist(gist, file),
    title: titleFromFilename(file.filename),
    language: languageFromFile(file),
    body: file.content,
  }
}
