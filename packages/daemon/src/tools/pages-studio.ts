import { execFile } from 'node:child_process'
import { mkdir, readdir, readFile, stat, writeFile } from 'node:fs/promises'
import { dirname, relative, resolve, sep } from 'node:path'
import { promisify } from 'node:util'
import { throwIfAborted } from '../abort.js'
import {
  assertPagesScaffoldExecutionBoundary,
  resolvePagesScaffoldPathPlan,
} from './pages-scaffold-plan.js'
import { resolveToolPath } from './path-utils.js'
import type { ToolDefinitionRuntime, ToolExecutionContext, ToolResult } from './registry.js'

type CommandRunner = (
  executable: string,
  args: string[],
  options: { cwd?: string; signal?: AbortSignal },
) => Promise<{ stdout: string; stderr: string }>

export interface PagesStatusToolDeps {
  runCommand?: CommandRunner
}

interface PlannedFile {
  absolutePath: string
  relativePath: string
  content: string
}

interface Finding {
  path: string
  line?: number
  reason: string
}

const MAX_SCAN_FILES_DEFAULT = 2000
const MAX_SCAN_BYTES_DEFAULT = 512 * 1024
const MAX_FINDINGS = 100

const SKIPPED_SCAN_DIRS = new Set([
  '.astro',
  '.git',
  '.next',
  '.pnpm-store',
  '.turbo',
  'coverage',
  'node_modules',
  'out',
])

const BLOCKED_FILE_PATTERNS: Array<{ reason: string; pattern: RegExp }> = [
  { reason: 'environment file must not be published', pattern: /^\.env(?:\..*)?$/u },
  { reason: 'npm credentials file must not be published', pattern: /^\.npmrc$/u },
  { reason: 'yarn credentials/config file must not be published', pattern: /^\.yarnrc(?:\.yml)?$/u },
  { reason: 'private key material must not be published', pattern: /(?:^id_(?:rsa|dsa|ecdsa|ed25519)$|\.(?:pem|key|p12|pfx)$)/iu },
  { reason: 'service account JSON must not be published', pattern: /service[-_ ]?account.*\.json$/iu },
]

const SECRET_PATTERNS: Array<{ reason: string; pattern: RegExp }> = [
  {
    reason: 'private key block',
    pattern: /-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----/u,
  },
  {
    reason: 'AWS access key id',
    pattern: /\bAKIA[0-9A-Z]{16}\b/u,
  },
  {
    reason: 'GitHub token',
    pattern: /\bgh[pousr]_[A-Za-z0-9_]{30,}\b/u,
  },
  {
    reason: 'OpenAI-style API key',
    pattern: /\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b/u,
  },
  {
    reason: 'JWT-like bearer token',
    pattern: /\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/u,
  },
  {
    reason: 'credential-like assignment',
    pattern:
      /\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|bearer[_-]?token|client[_-]?secret|password|passwd)\b\s*[:=]\s*['"]?(?!(?:test-token|example|example\.invalid|placeholder|changeme|replace_me|your_|xxx)\b)[A-Za-z0-9_./+=-]{24,}/iu,
  },
]

const execFileAsync = promisify(execFile)

export function createPagesScaffoldTool(): ToolDefinitionRuntime {
  return {
    name: 'pages.scaffold',
    description:
      'Create a GitHub Pages Studio static-site scaffold. Generates an Astro + MDX site, a Pages GitHub Actions workflow, and publish-safe defaults. Refuses to overwrite existing files unless overwrite is true.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        repoPath: { type: 'string', description: 'Repository root. Defaults to the active session cwd.' },
        sitePath: { type: 'string', description: 'Relative path for the static site. Defaults to "site". Use "." for a dedicated Pages repository.' },
        siteName: { type: 'string', description: 'Human-readable site name.' },
        template: { type: 'string', enum: ['astro-mdx'], description: 'Template to create. Only astro-mdx is supported initially.' },
        defaultBranch: { type: 'string', description: 'Branch that triggers Pages deploys. Defaults to main.' },
        includeWorkflow: { type: 'boolean', description: 'Whether to create .github/workflows/pages-studio.yml. Defaults to true.' },
        overwrite: { type: 'boolean', description: 'Overwrite existing scaffold files. Defaults to false.' },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      try {
        throwIfAborted(context?.signal, 'Pages scaffold aborted')
        const plan = resolvePagesScaffoldPathPlan(input, context?.cwd)
        const templateFiles = astroMdxTemplate(plan.siteName)
        const planned: PlannedFile[] = plan.targets.map((target) => ({
          absolutePath: target.absolutePath,
          relativePath: target.relativePath,
          content: target.templatePath
            ? templateFiles[target.templatePath]
            : pagesWorkflow({ sitePath: plan.sitePath, defaultBranch: plan.defaultBranch }),
        }))

        if (context?.workspaceRoot) {
          for (const target of [plan.repoRoot, plan.siteRoot, ...planned.map((file) => file.absolutePath)]) {
            await assertPagesScaffoldExecutionBoundary(context.workspaceRoot, target)
          }
        }

        const conflicts = await existingFiles(planned)
        if (conflicts.length > 0 && !plan.overwrite) {
          return {
            status: 'error',
            output: [
              `Pages scaffold refused to overwrite ${conflicts.length} existing file(s).`,
              ...conflicts.slice(0, 20).map((file) => `- ${file}`),
              conflicts.length > 20 ? `- ... ${conflicts.length - 20} more` : '',
              'Pass overwrite=true only after reviewing the existing files.',
            ].filter(Boolean).join('\n'),
            durationMs: Date.now() - start,
            code: 'PAGES_SCAFFOLD_CONFLICT_USER',
          }
        }

        for (const file of planned) {
          throwIfAborted(context?.signal, 'Pages scaffold aborted')
          if (context?.workspaceRoot) {
            await assertPagesScaffoldExecutionBoundary(context.workspaceRoot, file.absolutePath)
          }
          await context?.editCheckpoint?.recordPreEdit(file.absolutePath)
          if (context?.workspaceRoot) {
            await assertPagesScaffoldExecutionBoundary(context.workspaceRoot, file.absolutePath)
          }
          await mkdir(dirname(file.absolutePath), { recursive: true })
          if (context?.workspaceRoot) {
            await assertPagesScaffoldExecutionBoundary(context.workspaceRoot, file.absolutePath)
          }
          await writeFile(file.absolutePath, file.content, 'utf-8')
          await context?.workspaceMutation?.recordWrite(file.absolutePath)
        }

        return {
          status: 'success',
          output: [
            `Created ${plan.template} Pages Studio scaffold at ${plan.sitePath}.`,
            'Files written:',
            ...planned.map((file) => `- ${file.relativePath}`),
            '',
            'Next steps:',
            `1. Run ${plan.sitePath === '.' ? 'pnpm install && pnpm build' : `cd ${plan.sitePath} && pnpm install && pnpm build`}.`,
            `2. Run pages.scan on ${plan.sitePath} before publishing.`,
            '3. Review git diff, then commit and push so GitHub Actions can deploy Pages.',
          ].join('\n'),
          durationMs: Date.now() - start,
        }
      } catch (error) {
        return {
          status: 'error',
          output: error instanceof Error ? error.message : String(error),
          durationMs: Date.now() - start,
          code: 'PAGES_SCAFFOLD_PERMANENT',
        }
      }
    },
  }
}

export function createPagesStatusTool(deps: PagesStatusToolDeps = {}): ToolDefinitionRuntime {
  const runCommand = deps.runCommand ?? defaultRunCommand
  return {
    name: 'pages.status',
    description:
      'Return GitHub Pages and latest Pages Studio deployment status for a repository. Uses the authenticated gh CLI; does not read or print GitHub tokens.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'pages-status' },
    inputSchema: {
      type: 'object',
      properties: {
        repoPath: { type: 'string', description: 'Local repository path used to infer owner/repo from origin. Defaults to active session cwd.' },
        repo: { type: 'string', description: 'GitHub repository in owner/name form. Overrides repoPath inference.' },
        workflowFile: { type: 'string', description: 'Pages workflow filename. Defaults to pages-studio.yml.' },
        branch: { type: 'string', description: 'Optional branch filter for the latest workflow run.' },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      try {
        const repoRoot = resolveRepoRoot(input, context)
        const repo = normalizeRepoFullName(input.repo)
          ?? await inferGitHubRepo(repoRoot, runCommand, context?.signal)
        const workflowFile = normalizeWorkflowFile(input.workflowFile)
        const branch = typeof input.branch === 'string' && input.branch.trim()
          ? input.branch.trim()
          : undefined
        const [owner, name] = splitRepoFullName(repo)

        const pages = await readGitHubPagesStatus({ owner, repo: name, runCommand, signal: context?.signal })
        const workflow = await readLatestWorkflowRun({
          owner,
          repo: name,
          workflowFile,
          branch,
          runCommand,
          signal: context?.signal,
        })

        return {
          status: 'success',
          output: JSON.stringify({
            repo,
            pages,
            workflow,
          }, null, 2),
          durationMs: Date.now() - start,
        }
      } catch (error) {
        return {
          status: 'error',
          output: error instanceof Error ? error.message : String(error),
          durationMs: Date.now() - start,
          code: 'PAGES_STATUS_PERMANENT',
        }
      }
    },
  }
}

export function createPagesScanTool(): ToolDefinitionRuntime {
  return {
    name: 'pages.scan',
    description:
      'Scan a static Pages source or artifact tree for publish blockers such as .env files, credential stores, private keys, service account JSON, and common token patterns. Does not print matched secret values.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'pages-scan' },
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Directory to scan. Defaults to the active session cwd.' },
        maxFiles: { type: 'number', description: 'Maximum files to inspect. Defaults to 2000.' },
        maxBytesPerFile: { type: 'number', description: 'Maximum bytes to read per file. Defaults to 512 KiB.' },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      try {
        const root = resolveToolPath(typeof input.path === 'string' && input.path.trim() ? input.path : '.', context?.cwd)
        const maxFiles = normalizeLimit(input.maxFiles, MAX_SCAN_FILES_DEFAULT, 1, 10000)
        const maxBytesPerFile = normalizeLimit(input.maxBytesPerFile, MAX_SCAN_BYTES_DEFAULT, 1024, 5 * 1024 * 1024)
        const result = await scanTree(root, {
          context,
          maxFiles,
          maxBytesPerFile,
        })

        if (result.truncated) {
          result.findings.push({
            path: '.',
            reason: `scan stopped after ${maxFiles} files; narrow the scan or increase maxFiles before publishing`,
          })
        }

        if (result.findings.length > 0) {
          return {
            status: 'error',
            output: formatScanFailure(root, result.findings),
            durationMs: Date.now() - start,
            code: 'PAGES_PUBLISH_BLOCKED',
          }
        }

        return {
          status: 'success',
          output: `No Pages publish blockers found. Scanned ${result.filesScanned} file(s) under ${root}.`,
          durationMs: Date.now() - start,
        }
      } catch (error) {
        return {
          status: 'error',
          output: error instanceof Error ? error.message : String(error),
          durationMs: Date.now() - start,
          code: 'PAGES_SCAN_PERMANENT',
        }
      }
    },
  }
}

async function defaultRunCommand(
  executable: string,
  args: string[],
  options: { cwd?: string; signal?: AbortSignal },
): Promise<{ stdout: string; stderr: string }> {
  const { stdout, stderr } = await execFileAsync(executable, args, {
    cwd: options.cwd,
    signal: options.signal,
    maxBuffer: 10 * 1024 * 1024,
  })
  return { stdout, stderr }
}

function resolveRepoRoot(input: Record<string, unknown>, context: ToolExecutionContext | undefined): string {
  return resolveToolPath(
    typeof input.repoPath === 'string' && input.repoPath.trim() ? input.repoPath : '.',
    context?.cwd,
  )
}

function normalizeRepoFullName(raw: unknown): string | undefined {
  if (typeof raw !== 'string' || !raw.trim()) return undefined
  const value = raw.trim()
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/u.test(value)) {
    throw new Error('repo must be in owner/name form')
  }
  return value.replace(/\.git$/u, '')
}

function splitRepoFullName(fullName: string): [string, string] {
  const [owner, repo] = fullName.split('/')
  if (!owner || !repo) throw new Error('repo must be in owner/name form')
  return [owner, repo]
}

function normalizeWorkflowFile(raw: unknown): string {
  if (typeof raw !== 'string' || !raw.trim()) return 'pages-studio.yml'
  const value = raw.trim()
  if (value.includes('/') || value.includes('\\') || value.includes('..') || !/\.ya?ml$/u.test(value)) {
    throw new Error('workflowFile must be a workflow YAML filename such as pages-studio.yml')
  }
  return value
}

async function inferGitHubRepo(
  repoRoot: string,
  runCommand: CommandRunner,
  signal?: AbortSignal,
): Promise<string> {
  try {
    const { stdout } = await runCommand('git', ['config', '--get', 'remote.origin.url'], { cwd: repoRoot, signal })
    const parsed = parseGitHubRemote(stdout.trim())
    if (!parsed) throw new Error('origin remote is not a GitHub repository')
    return parsed
  } catch (error) {
    const message = commandErrorMessage(error)
    throw new Error(`Could not infer GitHub repo from ${repoRoot}. Pass repo="owner/name". ${message}`)
  }
}

function parseGitHubRemote(remote: string): string | null {
  const ssh = /^git@github\.com:([^/]+)\/(.+?)(?:\.git)?$/u.exec(remote)
  if (ssh?.[1] && ssh[2]) return `${ssh[1]}/${ssh[2].replace(/\.git$/u, '')}`

  const sshUrl = /^ssh:\/\/git@github\.com\/([^/]+)\/(.+?)(?:\.git)?$/u.exec(remote)
  if (sshUrl?.[1] && sshUrl[2]) return `${sshUrl[1]}/${sshUrl[2].replace(/\.git$/u, '')}`

  try {
    const url = new URL(remote)
    if (url.hostname !== 'github.com') return null
    const parts = url.pathname.replace(/^\/+/u, '').split('/')
    if (parts.length < 2 || !parts[0] || !parts[1]) return null
    return `${parts[0]}/${parts[1].replace(/\.git$/u, '')}`
  } catch {
    return null
  }
}

async function readGitHubPagesStatus(input: {
  owner: string
  repo: string
  runCommand: CommandRunner
  signal?: AbortSignal
}): Promise<Record<string, unknown>> {
  try {
    const data = await ghApi(input.runCommand, `repos/${encodeURIComponent(input.owner)}/${encodeURIComponent(input.repo)}/pages`, input.signal)
    const page = asRecord(data)
    return {
      configured: true,
      status: stringProp(page, 'status'),
      url: stringProp(page, 'html_url'),
      cname: stringProp(page, 'cname'),
      protectedDomainState: stringProp(page, 'protected_domain_state'),
      source: page.source ?? null,
    }
  } catch (error) {
    return {
      configured: false,
      error: conciseGhError(error),
    }
  }
}

async function readLatestWorkflowRun(input: {
  owner: string
  repo: string
  workflowFile: string
  branch?: string
  runCommand: CommandRunner
  signal?: AbortSignal
}): Promise<Record<string, unknown>> {
  const params = new URLSearchParams({ per_page: '1' })
  if (input.branch) params.set('branch', input.branch)
  try {
    const data = await ghApi(
      input.runCommand,
      `repos/${encodeURIComponent(input.owner)}/${encodeURIComponent(input.repo)}/actions/workflows/${encodeURIComponent(input.workflowFile)}/runs?${params.toString()}`,
      input.signal,
    )
    const record = asRecord(data)
    const runs = Array.isArray(record.workflow_runs) ? record.workflow_runs : []
    const run = runs[0]
    if (!run || typeof run !== 'object' || Array.isArray(run)) {
      return {
        workflowFile: input.workflowFile,
        branch: input.branch ?? null,
        latestRun: null,
      }
    }
    const latest = asRecord(run)
    return {
      workflowFile: input.workflowFile,
      branch: input.branch ?? null,
      latestRun: {
        id: numberOrStringProp(latest, 'id'),
        name: stringProp(latest, 'name'),
        status: stringProp(latest, 'status'),
        conclusion: stringProp(latest, 'conclusion'),
        event: stringProp(latest, 'event'),
        headBranch: stringProp(latest, 'head_branch'),
        headSha: stringProp(latest, 'head_sha'),
        createdAt: stringProp(latest, 'created_at'),
        updatedAt: stringProp(latest, 'updated_at'),
        url: stringProp(latest, 'html_url'),
      },
    }
  } catch (error) {
    return {
      workflowFile: input.workflowFile,
      branch: input.branch ?? null,
      latestRun: null,
      error: conciseGhError(error),
    }
  }
}

async function ghApi(
  runCommand: CommandRunner,
  endpoint: string,
  signal?: AbortSignal,
): Promise<unknown> {
  try {
    const { stdout } = await runCommand('gh', ['api', endpoint], { signal })
    return JSON.parse(stdout)
  } catch (error) {
    throw new Error(commandErrorMessage(error))
  }
}

function commandErrorMessage(error: unknown): string {
  if (error && typeof error === 'object') {
    const maybe = error as { stderr?: unknown; stdout?: unknown; message?: unknown }
    const text = [maybe.stderr, maybe.stdout, maybe.message]
      .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
      .map((value) => value.trim())
      .join('\n')
    if (text) return text
  }
  return error instanceof Error ? error.message : String(error)
}

function conciseGhError(error: unknown): string {
  return commandErrorMessage(error)
    .split('\n')
    .map((line) => line.trim())
    .find((line) => line.length > 0)
    ?? 'unknown GitHub status error'
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function stringProp(record: Record<string, unknown>, key: string): string | null {
  const value = record[key]
  return typeof value === 'string' && value.trim() ? value : null
}

function numberOrStringProp(record: Record<string, unknown>, key: string): number | string | null {
  const value = record[key]
  return typeof value === 'number' || typeof value === 'string' ? value : null
}

function toPortablePath(path: string): string {
  return path.split(sep).join('/')
}

async function existingFiles(files: PlannedFile[]): Promise<string[]> {
  const conflicts: string[] = []
  for (const file of files) {
    try {
      await stat(file.absolutePath)
      conflicts.push(file.relativePath)
    } catch {
      // Missing files are expected.
    }
  }
  return conflicts
}

function astroMdxTemplate(siteName: string): Record<string, string> {
  const packageName = slugify(siteName) || 'pages-studio-site'
  return {
    'package.json': `${JSON.stringify({
      name: packageName,
      version: '0.1.0',
      private: true,
      type: 'module',
      packageManager: 'pnpm@10.32.1',
      scripts: {
        dev: 'astro dev',
        build: 'astro build',
        preview: 'astro preview',
      },
      dependencies: {
        '@astrojs/mdx': '^4.0.0',
        astro: '^5.0.0',
      },
      devDependencies: {
        typescript: '~5.7.0',
      },
    }, null, 2)}\n`,
    'astro.config.mjs': `import mdx from '@astrojs/mdx'
import { defineConfig } from 'astro/config'

const repositoryName = process.env.GITHUB_REPOSITORY?.split('/')[1] ?? ''
const isUserOrOrgPages = repositoryName.endsWith('.github.io')
const base = process.env.PUBLIC_BASE_PATH ?? (repositoryName && !isUserOrOrgPages ? \`/\${repositoryName}\` : '/')
const site = process.env.PUBLIC_SITE_URL

export default defineConfig({
  ...(site ? { site } : {}),
  base,
  integrations: [mdx()],
})
`,
    '.gitignore': `node_modules/
dist/
.astro/
.env
.env.*
`,
    'public/.nojekyll': '',
    'public/assets/.gitkeep': '',
    'src/layouts/Layout.astro': `---
import '../styles/global.css'

interface Props {
  title: string
  description?: string
}

const { title, description = 'A public static workspace generated with sepilotd Pages Studio.' } = Astro.props
---

<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="description" content={description} />
    <title>{title}</title>
  </head>
  <body>
    <header class="site-header">
      <a class="brand" href={Astro.site ? new URL(Astro.base, Astro.site).pathname : Astro.base}>{${JSON.stringify(siteName)}}</a>
      <nav aria-label="Main navigation">
        <a href={\`\${Astro.base}posts/first-note/\`}>Posts</a>
        <a href={\`\${Astro.base}wiki/\`}>Wiki</a>
        <a href={\`\${Astro.base}decks/demo/\`}>Deck</a>
      </nav>
    </header>
    <main>
      <slot />
    </main>
  </body>
</html>
`,
    'src/pages/index.astro': `---
import Layout from '../layouts/Layout.astro'
---

<Layout title="${escapeForAstroAttribute(siteName)}">
  <section class="hero">
    <p class="eyebrow">Pages Studio</p>
    <h1>${escapeHtml(siteName)}</h1>
    <p>
      A public static workspace for ideas, notes, wiki pages, lightweight dashboards, and
      presentation material generated through sepilotd.
    </p>
  </section>

  <section class="grid">
    <a class="tile" href={\`\${Astro.base}posts/first-note/\`}>
      <span>Post</span>
      <strong>First note</strong>
      <p>Publish narrative updates and research notes as MDX.</p>
    </a>
    <a class="tile" href={\`\${Astro.base}wiki/\`}>
      <span>Wiki</span>
      <strong>Knowledge base</strong>
      <p>Keep durable public context in static wiki pages.</p>
    </a>
    <a class="tile" href={\`\${Astro.base}decks/demo/\`}>
      <span>Deck</span>
      <strong>Static slides</strong>
      <p>Share presentation material without a server runtime.</p>
    </a>
  </section>
</Layout>
`,
    'src/pages/posts/first-note.mdx': `---
layout: ../../layouts/Layout.astro
title: First note
description: A starter Pages Studio post.
---

# First note

This public note is generated as static MDX. Replace it with an idea, project log,
release note, or research summary.

## What belongs here

- Public ideas and records that should be easy to link.
- Static data snapshots that do not require a private backend.
- Explanations, decisions, and narrative updates.

## What does not belong here

Secrets, tokens, private user state, and local sepilotd session files must stay out of
this repository and out of GitHub Pages artifacts.
`,
    'src/pages/wiki/index.mdx': `---
layout: ../../layouts/Layout.astro
title: Wiki
description: A starter public wiki page.
---

# Wiki

Use this area for durable public knowledge. Keep pages concise, link related entries,
and prefer static JSON or Markdown when the data needs to feed visual pages.

## Starting points

- Project glossary
- Decision log
- Reading notes
- Public operating notes
`,
    'src/pages/decks/demo.astro': `---
import Layout from '../../layouts/Layout.astro'
---

<Layout title="Demo deck">
  <section class="deck" aria-label="Demo presentation">
    <article class="slide">
      <p class="eyebrow">Slide 1</p>
      <h1>Static decks work here</h1>
      <p>Use one HTML page, CSS scroll snapping, and no private runtime.</p>
    </article>
    <article class="slide">
      <p class="eyebrow">Slide 2</p>
      <h1>Publish from commits</h1>
      <p>Every update is reviewable as a Git diff before GitHub Actions deploys it.</p>
    </article>
  </section>
</Layout>
`,
    'src/styles/global.css': `:root {
  color-scheme: light;
  --bg: #f8faf7;
  --ink: #17211a;
  --muted: #56645b;
  --line: #d9e0d8;
  --accent: #1f7a5a;
  --accent-strong: #0d4f3a;
  --panel: #ffffff;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family:
    Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    sans-serif;
  line-height: 1.55;
}

a {
  color: inherit;
}

.site-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 18px clamp(20px, 6vw, 72px);
  border-bottom: 1px solid var(--line);
  background: rgba(248, 250, 247, 0.92);
  position: sticky;
  top: 0;
  backdrop-filter: blur(16px);
}

.brand {
  font-weight: 760;
  text-decoration: none;
}

nav {
  display: flex;
  gap: 16px;
  color: var(--muted);
  font-size: 0.95rem;
}

nav a {
  text-decoration: none;
}

main {
  width: min(960px, calc(100% - 40px));
  margin: 0 auto;
  padding: 48px 0 80px;
}

.hero {
  padding: 64px 0 44px;
}

.eyebrow,
.tile span {
  color: var(--accent-strong);
  font-size: 0.78rem;
  font-weight: 760;
  letter-spacing: 0;
  text-transform: uppercase;
}

h1 {
  margin: 8px 0 16px;
  max-width: 780px;
  font-size: clamp(2.4rem, 7vw, 5.2rem);
  line-height: 0.98;
  letter-spacing: 0;
}

h2 {
  margin-top: 36px;
}

p,
li {
  color: var(--muted);
  font-size: 1.04rem;
}

.hero p {
  max-width: 680px;
  font-size: 1.18rem;
}

.grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}

.tile {
  min-height: 190px;
  padding: 22px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  text-decoration: none;
}

.tile strong {
  display: block;
  margin-top: 10px;
  font-size: 1.2rem;
}

.deck {
  display: grid;
  gap: 24px;
}

.slide {
  min-height: 68vh;
  display: grid;
  align-content: center;
  padding: clamp(28px, 8vw, 72px);
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}

@media (max-width: 760px) {
  .site-header,
  nav {
    align-items: flex-start;
    flex-direction: column;
  }

  .grid {
    grid-template-columns: 1fr;
  }
}
`,
  }
}

function pagesWorkflow(input: { sitePath: string; defaultBranch: string }): string {
  const siteDir = input.sitePath === '.' ? '.' : input.sitePath
  const installCommand = 'if [ -f pnpm-lock.yaml ]; then pnpm install --frozen-lockfile; else pnpm install --no-frozen-lockfile; fi'
  return `name: Deploy Pages Studio

on:
  push:
    branches: [${input.defaultBranch}]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages-studio
  cancel-in-progress: false

env:
  SITE_DIR: ${siteDir}

jobs:
  build:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: \${{ env.SITE_DIR }}
    steps:
      - name: Checkout
        uses: actions/checkout@v6
      - name: Setup pnpm
        uses: pnpm/action-setup@v6
        with:
          version: 10
          run_install: false
      - name: Setup Node
        uses: actions/setup-node@v6
        with:
          node-version: 24
          cache: pnpm
          cache-dependency-path: \${{ env.SITE_DIR }}/pnpm-lock.yaml
      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v5
      - name: Install dependencies
        run: ${installCommand}
      - name: Build
        run: pnpm build
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v4
        with:
          path: \${{ env.SITE_DIR }}/dist

  deploy:
    environment:
      name: github-pages
      url: \${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
`
}

function slugify(value: string): string {
  const slug = value
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/gu, '-')
    .replace(/^-+|-+$/gu, '')
  return slug || 'pages-studio-site'
}

function escapeForAstroAttribute(value: string): string {
  return value.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;')
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}

function normalizeLimit(raw: unknown, fallback: number, min: number, max: number): number {
  if (typeof raw !== 'number' || !Number.isFinite(raw)) return fallback
  return Math.max(min, Math.min(max, Math.trunc(raw)))
}

async function scanTree(
  root: string,
  options: {
    context?: ToolExecutionContext
    maxFiles: number
    maxBytesPerFile: number
  },
): Promise<{ filesScanned: number; findings: Finding[]; truncated: boolean }> {
  const findings: Finding[] = []
  let filesScanned = 0
  let truncated = false
  const rootStat = await stat(root)
  if (!rootStat.isDirectory()) {
    throw new Error('pages.scan path must be a directory')
  }

  async function walk(dir: string, isRoot = false): Promise<void> {
    if (findings.length >= MAX_FINDINGS || truncated) return
    throwIfAborted(options.context?.signal, 'Pages scan aborted')
    const entries = await readdir(dir, { withFileTypes: true })
    for (const entry of entries) {
      if (findings.length >= MAX_FINDINGS || truncated) return
      const absolutePath = resolve(dir, entry.name)
      const relPath = toPortablePath(relative(root, absolutePath)) || '.'

      if (entry.isSymbolicLink()) {
        findings.push({ path: relPath, reason: 'symbolic links are not allowed in GitHub Pages artifacts' })
        continue
      }

      if (entry.isDirectory()) {
        if (!isRoot && SKIPPED_SCAN_DIRS.has(entry.name)) continue
        await walk(absolutePath)
        continue
      }

      if (!entry.isFile()) continue
      filesScanned += 1
      if (filesScanned > options.maxFiles) {
        truncated = true
        return
      }

      for (const blocked of BLOCKED_FILE_PATTERNS) {
        if (blocked.pattern.test(entry.name)) {
          findings.push({ path: relPath, reason: blocked.reason })
        }
      }

      const fileStat = await stat(absolutePath)
      if (fileStat.size > options.maxBytesPerFile) continue
      const buffer = await readFile(absolutePath)
      if (buffer.includes(0)) continue
      const text = buffer.toString('utf-8')
      for (const secret of SECRET_PATTERNS) {
        const match = secret.pattern.exec(text)
        if (!match) continue
        findings.push({
          path: relPath,
          line: lineNumberForIndex(text, match.index),
          reason: secret.reason,
        })
      }
    }
  }

  await walk(root, true)
  return { filesScanned, findings, truncated }
}

function lineNumberForIndex(text: string, index: number): number {
  return text.slice(0, index).split('\n').length
}

function formatScanFailure(root: string, findings: Finding[]): string {
  const displayed = findings.slice(0, MAX_FINDINGS)
  return [
    `Pages publish scan found ${findings.length} blocker(s) under ${root}.`,
    ...displayed.map((finding) => `- ${finding.path}${finding.line ? `:${finding.line}` : ''}: ${finding.reason}`),
    findings.length > displayed.length ? `- ... ${findings.length - displayed.length} more` : '',
    'Remove these files or values before committing, pushing, or uploading a Pages artifact.',
  ].filter(Boolean).join('\n')
}
