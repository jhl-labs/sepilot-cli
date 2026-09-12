import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export interface PagesToolResult {
  output: string
  status: 'success' | 'error'
  durationMs: number
  code?: string
}

type PagesClient = Pick<DaemonClient, 'request'>

interface PagesResponse {
  data: PagesToolResult
}

export interface PagesInitOptions {
  url?: string
  repoPath?: string
  sitePath?: string
  name?: string
  siteName?: string
  branch?: string
  defaultBranch?: string
  workflow?: boolean
  includeWorkflow?: boolean
  overwrite?: boolean
}

export interface PagesScanOptions {
  url?: string
  maxFiles?: string
  maxBytesPerFile?: string
}

export interface PagesStatusOptions {
  url?: string
  repo?: string
  repoPath?: string
  workflowFile?: string
  branch?: string
}

function printToolResult(result: PagesToolResult): void {
  output(result, (item) => item.output)
  if (result.status === 'error') {
    process.exitCode = 1
  }
}

function parsePositiveInteger(name: string, value: string | undefined): number | undefined {
  if (value == null || value === '') return undefined
  const parsed = Number.parseInt(value, 10)
  if (!Number.isFinite(parsed) || parsed < 1 || String(parsed) !== value.trim()) {
    throw new Error(`${name} must be a positive integer`)
  }
  return parsed
}

async function postPagesTool(
  client: PagesClient,
  endpoint: 'scaffold' | 'scan',
  body: Record<string, unknown>,
): Promise<PagesToolResult> {
  const response = await client.request<PagesResponse>(`/api/v1/pages/${endpoint}`, {
    method: 'POST',
    body,
  })
  return response.data
}

async function getPagesStatus(
  client: PagesClient,
  params: Record<string, string | undefined>,
): Promise<PagesToolResult> {
  const query = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value) query.set(key, value)
  }
  const suffix = query.toString()
  const response = await client.request<PagesResponse>(
    suffix ? `/api/v1/pages/status?${suffix}` : '/api/v1/pages/status',
    { method: 'GET' },
  )
  return response.data
}

export async function pagesInitCommandImpl(
  client: PagesClient,
  options: PagesInitOptions = {},
): Promise<PagesToolResult> {
  const result = await postPagesTool(client, 'scaffold', {
    repoPath: options.repoPath,
    sitePath: options.sitePath,
    siteName: options.siteName ?? options.name,
    template: 'astro-mdx',
    defaultBranch: options.defaultBranch ?? options.branch,
    includeWorkflow: options.includeWorkflow ?? options.workflow ?? true,
    overwrite: options.overwrite === true,
    cwd: process.cwd(),
  })
  printToolResult(result)
  return result
}

export async function pagesScanCommandImpl(
  path: string | undefined,
  client: PagesClient,
  options: PagesScanOptions = {},
): Promise<PagesToolResult> {
  const result = await postPagesTool(client, 'scan', {
    path: path ?? '.',
    maxFiles: parsePositiveInteger('--max-files', options.maxFiles),
    maxBytesPerFile: parsePositiveInteger('--max-bytes-per-file', options.maxBytesPerFile),
    cwd: process.cwd(),
  })
  printToolResult(result)
  return result
}

export async function pagesStatusCommandImpl(
  client: PagesClient,
  options: PagesStatusOptions = {},
): Promise<PagesToolResult> {
  const result = await getPagesStatus(client, {
    repo: options.repo,
    repoPath: options.repoPath,
    workflowFile: options.workflowFile,
    branch: options.branch,
    cwd: process.cwd(),
  })
  printToolResult(result)
  return result
}

export async function pagesInitCommand(options: PagesInitOptions = {}): Promise<void> {
  await pagesInitCommandImpl(new DaemonClient(options.url), options)
}

export async function pagesScanCommand(path?: string, options: PagesScanOptions = {}): Promise<void> {
  await pagesScanCommandImpl(path, new DaemonClient(options.url), options)
}

export async function pagesStatusCommand(options: PagesStatusOptions = {}): Promise<void> {
  await pagesStatusCommandImpl(new DaemonClient(options.url), options)
}
