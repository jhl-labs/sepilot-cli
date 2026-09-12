import { access, mkdir, writeFile } from 'node:fs/promises'
import { basename, dirname, join, relative, resolve } from 'node:path'
import chalk from 'chalk'

const API_CLIENT_VERSION = '^0.2.10'
const REACT_VERSION = '^19.0.0'
const REACT_TYPES_VERSION = '^19.0.0'
const WS_VERSION = '^8.20.0'
const DEFAULT_PACKAGE_MANAGER = 'pnpm@10.32.1'
const EXTENSION_TEMPLATES = ['minimal', 'full'] as const

type ExtensionTemplate = (typeof EXTENSION_TEMPLATES)[number]

export interface ExtensionCreateOptions {
  cwd?: string
  description?: string
  dir?: string
  template?: ExtensionTemplate
  workspace?: boolean
}

export interface ExtensionScaffoldResult {
  projectName: string
  packageName: string
  targetDir: string
  workspaceRoot?: string
  linkedSdkRoot?: string
  template: ExtensionTemplate
}

interface ExtensionTemplateFile {
  relativePath: string
  content: string
}

interface ExtensionTemplatePlan {
  projectName: string
  packageName: string
  targetDir: string
  workspaceRoot?: string
  linkedSdkRoot?: string
  template: ExtensionTemplate
  files: ExtensionTemplateFile[]
}

export async function extensionCreateCommand(
  name: string,
  options: ExtensionCreateOptions = {},
): Promise<void> {
  const result = await createExtensionScaffold(name, options)

  console.log(chalk.green(`Extension starter created: ${result.targetDir}`))
  console.log(chalk.gray('\nNext steps:'))
  if (result.workspaceRoot) {
    console.log(chalk.cyan(`  cd ${result.workspaceRoot}`))
    console.log(chalk.cyan('  pnpm install'))
    console.log(chalk.cyan(`  pnpm --filter ${result.packageName} build`))
    console.log(chalk.cyan(`  pnpm --filter ${result.packageName} exec node dist/index.js help`))
  } else {
    console.log(chalk.cyan(`  cd ${result.targetDir}`))
    console.log(chalk.cyan('  pnpm install'))
    console.log(chalk.cyan('  pnpm build'))
    console.log(chalk.cyan('  node dist/index.js help'))
  }
}

async function planExtensionScaffold(
  name: string,
  options: ExtensionCreateOptions = {},
): Promise<ExtensionTemplatePlan> {
  const cwd = options.cwd ?? process.cwd()
  const template = normalizeTemplate(options.template)
  const projectName = normalizeProjectName(name)
  if (!projectName) {
    throw new Error('Extension name must include at least one letter or number')
  }

  const workspaceRoot = await resolveWorkspaceRoot(cwd)
  // --workspace and --dir are conceptually contradictory: --workspace
  // forces monorepo placement (packages/<name>), --dir picks an
  // explicit standalone path. Silently ignoring --dir under
  // --workspace caused real confusion — surface it instead.
  if (options.workspace && options.dir) {
    throw new Error('--workspace and --dir cannot be combined; pick one')
  }
  const useWorkspace = options.workspace || (!options.dir && !!workspaceRoot)
  if (options.workspace && !workspaceRoot) {
    throw new Error('Could not find a pnpm workspace root from the current directory')
  }

  const targetDir = useWorkspace
    ? resolve(workspaceRoot as string, 'packages', projectName)
    : resolve(cwd, options.dir ?? projectName)
  const linkedSdkRoot = !useWorkspace && workspaceRoot ? workspaceRoot : undefined
  const packageName = useWorkspace
    ? `@sepilotd/${projectName}`
    : basename(targetDir)
  const description = options.description ?? `${projectName} extension for sepilotd`

  return {
    projectName,
    packageName,
    targetDir,
    workspaceRoot: useWorkspace ? workspaceRoot ?? undefined : undefined,
    linkedSdkRoot,
    template,
    files: buildExtensionTemplateFiles({
      projectName,
      packageName,
      description,
      template,
      useWorkspace,
      targetDir,
      linkedSdkRoot,
    }),
  }
}

function buildExtensionTemplateFiles(params: {
  projectName: string
  packageName: string
  description: string
  template: ExtensionTemplate
  useWorkspace: boolean
  targetDir: string
  linkedSdkRoot?: string
}): ExtensionTemplateFile[] {
  const {
    projectName,
    packageName,
    description,
    template,
    useWorkspace,
    targetDir,
    linkedSdkRoot,
  } = params

  const packageJson = createPackageJson({
    packageName,
    description,
    template,
    useWorkspace,
    linkedSdkRoot,
  })

  const files: ExtensionTemplateFile[] = [
    {
      relativePath: 'package.json',
      content: `${JSON.stringify(packageJson, null, 2)}\n`,
    },
    {
      relativePath: 'tsconfig.json',
      content: useWorkspace
        ? createWorkspaceTsconfig()
        : createTsconfig({ targetDir, linkedSdkRoot }),
    },
    {
      relativePath: 'tsup.config.ts',
      content: createTsupConfig({
        bundleLocalSdk: useWorkspace || Boolean(linkedSdkRoot),
        template,
      }),
    },
    {
      relativePath: 'README.md',
      content: createReadme({
        packageName,
        description,
        template,
        useWorkspace,
        linkedSdkRoot,
      }),
    },
    {
      relativePath: 'sepilot.extension.json',
      content: `${JSON.stringify(createExtensionManifest({
        projectName,
        packageName,
        description,
        template,
      }), null, 2)}\n`,
    },
    {
      relativePath: 'src/index.ts',
      content: createIndexSource(packageName, template),
    },
  ]

  if (!useWorkspace) {
    files.push({
      relativePath: '.gitignore',
      content: createGitignore(),
    })
  }

  return files
}

export async function createExtensionScaffold(
  name: string,
  options: ExtensionCreateOptions = {},
): Promise<ExtensionScaffoldResult> {
  const plan = await planExtensionScaffold(name, options)
  await assertPathDoesNotExist(plan.targetDir)

  for (const file of plan.files) {
    const fullPath = join(plan.targetDir, file.relativePath)
    await mkdir(dirname(fullPath), { recursive: true })
    await writeFile(fullPath, file.content)
  }

  return {
    projectName: plan.projectName,
    packageName: plan.packageName,
    targetDir: plan.targetDir,
    workspaceRoot: plan.workspaceRoot,
    linkedSdkRoot: plan.linkedSdkRoot,
    template: plan.template,
  }
}

async function assertPathDoesNotExist(targetDir: string): Promise<void> {
  try {
    await access(targetDir)
    throw new Error(`Target path already exists: ${targetDir}`)
  } catch (error) {
    if (error instanceof Error && error.message.startsWith('Target path already exists:')) {
      throw error
    }
  }
}

function normalizeProjectName(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function normalizeTemplate(template?: string): ExtensionTemplate {
  if (!template) return 'minimal'

  if ((EXTENSION_TEMPLATES as readonly string[]).includes(template)) {
    return template as ExtensionTemplate
  }

  throw new Error(`Unknown extension template: ${template}`)
}

async function resolveWorkspaceRoot(startDir: string): Promise<string | null> {
  let current = resolve(startDir)

  while (true) {
    try {
      await access(join(current, 'pnpm-workspace.yaml'))
      return current
    } catch {
      // keep walking upward
    }

    const parent = resolve(current, '..')
    if (parent === current) {
      return null
    }
    current = parent
  }
}

function createPackageJson(options: {
  packageName: string
  description: string
  template: ExtensionTemplate
  useWorkspace: boolean
  linkedSdkRoot?: string
}) {
  const dependencies = createDependencies(
    options.template,
    options.useWorkspace,
    options.linkedSdkRoot,
  )
  const pnpmConfig = createPnpmConfig(options.useWorkspace, options.linkedSdkRoot)
  const devDependencies = createDevDependencies(options.useWorkspace, options.linkedSdkRoot)

  return {
    name: options.packageName,
    version: '0.1.0',
    private: true,
    packageManager: resolvePackageManager(),
    type: 'module',
    description: options.description,
    main: './dist/index.js',
    types: './dist/index.d.ts',
    scripts: {
      build: 'tsup',
      dev: 'tsup --watch',
      start: 'node ./dist/index.js',
      typecheck: 'tsc --noEmit',
    },
    dependencies,
    devDependencies,
    ...(pnpmConfig ? { pnpm: pnpmConfig } : {}),
  }
}

function resolvePackageManager(env: NodeJS.ProcessEnv = process.env): string {
  const userAgent = env.npm_config_user_agent ?? ''
  const pnpmVersion = userAgent.match(/\bpnpm\/([^\s]+)/)?.[1]
  return pnpmVersion ? `pnpm@${pnpmVersion}` : DEFAULT_PACKAGE_MANAGER
}

function createDependencies(
  template: ExtensionTemplate,
  useWorkspace: boolean,
  linkedSdkRoot?: string,
): Record<string, string> {
  const needsDirectWsDependency = template === 'full' && (useWorkspace || Boolean(linkedSdkRoot))

  if (useWorkspace) {
    return {
      '@sepilotd/api-client': 'workspace:*',
      ...(needsDirectWsDependency ? { ws: WS_VERSION } : {}),
    }
  }

  if (linkedSdkRoot) {
    return {
      '@sepilotd/api-client': `file:${join(linkedSdkRoot, 'packages', 'api-client')}`,
      '@sepilotd/core': `file:${join(linkedSdkRoot, 'packages', 'core')}`,
      react: REACT_VERSION,
      'react-dom': REACT_VERSION,
      ...(needsDirectWsDependency ? { ws: WS_VERSION } : {}),
    }
  }

  return {
    '@sepilotd/api-client': API_CLIENT_VERSION,
  }
}

function createDevDependencies(
  useWorkspace: boolean,
  linkedSdkRoot?: string,
): Record<string, string> {
  return {
    ...(useWorkspace ? {} : { '@types/node': '^24.0.0' }),
    ...(linkedSdkRoot
      ? {
          '@types/react': REACT_TYPES_VERSION,
          '@types/react-dom': REACT_TYPES_VERSION,
        }
      : {}),
    tsup: '^8.0.0',
    typescript: '~5.7.0',
  }
}

function createPnpmConfig(
  useWorkspace: boolean,
  linkedSdkRoot?: string,
): { overrides: Record<string, string> } | undefined {
  if (useWorkspace || !linkedSdkRoot) {
    return undefined
  }

  return {
    overrides: {
      '@sepilotd/core': `file:${join(linkedSdkRoot, 'packages', 'core')}`,
    },
  }
}

function createTsconfig(options: {
  targetDir: string
  linkedSdkRoot?: string
}): string {
  const compilerOptions: Record<string, unknown> = {
    target: 'ES2024',
    module: 'ESNext',
    moduleResolution: 'Bundler',
    strict: true,
    esModuleInterop: true,
    skipLibCheck: true,
    forceConsistentCasingInFileNames: true,
    jsx: 'react-jsx',
    outDir: './dist',
    types: ['node'],
  }

  if (options.linkedSdkRoot) {
    compilerOptions.baseUrl = '.'
    compilerOptions.paths = createSourceSdkPaths(options.targetDir, options.linkedSdkRoot)
  } else {
    compilerOptions.rootDir = './src'
  }

  return `${JSON.stringify({ compilerOptions, include: ['src/**/*'] }, null, 2)}\n`
}

function createWorkspaceTsconfig(): string {
  return `${JSON.stringify(
    {
      extends: '../../tsconfig.base.json',
      compilerOptions: {
        baseUrl: '../..',
        outDir: './dist',
        jsx: 'react-jsx',
        types: ['node'],
        paths: createWorkspaceSourceSdkPaths(),
      },
      include: ['src/**/*'],
    },
    null,
    2,
  )}\n`
}

function createTsupConfig(options: {
  bundleLocalSdk: boolean
  template: ExtensionTemplate
}): string {
  const noExternal = options.bundleLocalSdk
    ? '\n  noExternal: [/@sepilotd\\/api-client/, /@sepilotd\\/core/],'
    : ''
  const external = options.bundleLocalSdk && options.template === 'full'
    ? "\n  external: ['ws'],"
    : ''

  return `import { defineConfig } from 'tsup'

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm'],
  dts: true,
  sourcemap: true,
  clean: true,${noExternal}${external}
})
`
}

function createWorkspaceSourceSdkPaths(): Record<string, string[]> {
  return {
    '@sepilotd/api-client': ['packages/api-client/src/index.ts'],
    '@sepilotd/api-client/*': ['packages/api-client/src/*'],
    '@sepilotd/core': ['packages/core/src/index.ts'],
    '@sepilotd/core/*': ['packages/core/src/*'],
  }
}

function createSourceSdkPaths(
  targetDir: string,
  linkedSdkRoot: string,
): Record<string, string[]> {
  return {
    '@sepilotd/api-client': [relativeConfigPath(targetDir, linkedSdkRoot, 'packages/api-client/src/index.ts')],
    '@sepilotd/api-client/*': [relativeConfigPath(targetDir, linkedSdkRoot, 'packages/api-client/src/*')],
    '@sepilotd/core': [relativeConfigPath(targetDir, linkedSdkRoot, 'packages/core/src/index.ts')],
    '@sepilotd/core/*': [relativeConfigPath(targetDir, linkedSdkRoot, 'packages/core/src/*')],
  }
}

function relativeConfigPath(
  fromDir: string,
  rootDir: string,
  targetPath: string,
): string {
  return toConfigPath(relative(fromDir, join(rootDir, targetPath)))
}

function toConfigPath(path: string): string {
  return path.replaceAll('\\', '/')
}

function createGitignore(): string {
  return `node_modules
dist
.sepilot/
`
}

function createExtensionManifest(options: {
  projectName: string
  packageName: string
  description: string
  template: ExtensionTemplate
}) {
  return {
    schemaVersion: 1,
    id: options.projectName,
    name: options.packageName,
    description: options.description,
    token: {
      scopes: options.template === 'full'
        ? ['inspect', 'chat', 'files', 'ws', 'approvals']
        : ['inspect', 'chat'],
    },
  }
}

function createReadme(options: {
  packageName: string
  description: string
  template: ExtensionTemplate
  useWorkspace: boolean
  linkedSdkRoot?: string
}): string {
  const usage = options.useWorkspace
    ? `\`\`\`bash
pnpm install
pnpm --filter ${options.packageName} build
pnpm --filter ${options.packageName} exec node dist/index.js help
\`\`\``
    : `\`\`\`bash
pnpm install
pnpm build
node dist/index.js help
\`\`\``
  const sdkNote = options.linkedSdkRoot
    ? `\nThis starter links to the local SDK packages in:\n\n\`\`\`text\n${options.linkedSdkRoot}\n\`\`\`\n`
    : ''
  const commands = options.template === 'full'
    ? `- \`health\`
- \`chat <message...>\` (SSE streaming)
- \`stream <message...>\`
- \`review-file <path> [prompt...]\`
- \`ws-chat <message...>\``
    : `- \`health\`
- \`chat <message...>\` (SSE streaming)
- \`stream <message...>\``

  return `# ${options.packageName}

${options.description}

This starter talks to a local \`sepilotd\` daemon over the shared extension client.
${sdkNote}

It also includes \`sepilot.extension.json\` so you can bootstrap a scoped daemon token with:

\`\`\`bash
sepilot extension-install ${options.useWorkspace ? `packages/${options.packageName.replace('@sepilotd/', '')}` : '.'}
\`\`\`

## Architecture Boundary

- Import browser-safe runtime SDK code from \`@sepilotd/api-client\` root exports.
- Use \`@sepilotd/api-client/node\` only for Node-only WebSocket helpers.
- Do not deep-import private \`@sepilotd/api-client/*\` entrypoints in extension runtime code.
- Do not import other \`@sepilotd/*\` packages directly from extension runtime code.

## Commands

${commands}

## Environment

- \`SEPILOT_DAEMON_URL\`
- \`SEPILOT_DAEMON_TOKEN\`
- \`SEPILOT_DAEMON_TOKEN_FILE\`
- \`.sepilot/extension.env\` is auto-loaded when present

## Usage

${usage}

Edit \`src/index.ts\` to add your own daemon workflows.
`
}

function createIndexSource(
  packageName: string,
  template: ExtensionTemplate,
): string {
  if (template === 'full') {
    return createFullIndexSource(packageName)
  }

  return createMinimalIndexSource(packageName)
}

function createMinimalIndexSource(packageName: string): string {
  return `import { readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import {
  DEFAULT_CHAT_OPTION_DEFAULTS,
  DaemonClient,
  createTerminalChatStreamRenderer,
  sanitizeChatOptions,
  streamDaemonChatEvents,
  type ChatStreamOptions,
} from '@sepilotd/api-client'

interface RuntimeConfig {
  baseUrl: string
  token: string | null
}

async function main(): Promise<void> {
  const [command, ...rest] = process.argv.slice(2)

  try {
    switch (command ?? 'help') {
      case 'help':
      case '--help':
      case '-h':
        printHelp()
        return
      case 'health':
        await runHealth()
        return
      case 'chat':
        await runChat(rest)
        return
      case 'stream':
        await runStream(rest)
        return
      default:
        throw new Error(\`Unknown command: \${command}\`)
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    console.error(\`Error: \${message}\`)
    process.exitCode = 1
  }
}

async function runHealth(): Promise<void> {
  const client = await createDaemonClient()
  const health = await client.health()
  console.log(\`status: \${health.status}\`)
  console.log(\`version: \${health.version}\`)
}

async function runChat(args: string[]): Promise<void> {
  await runStream(args)
}

async function runStream(args: string[]): Promise<void> {
  const message = joinArgs(args, 'stream <message...>')
  const client = await createDaemonClient()
  const response = await client.chatStream(
    message,
    undefined,
    buildStreamChatOptions(),
  )
  const renderer = createTerminalChatStreamRenderer({
    write: (text) => process.stdout.write(text),
  })

  for await (const event of streamDaemonChatEvents(response)) {
    renderer.handleEvent(event)
  }
}

async function createDaemonClient(): Promise<DaemonClient> {
  const config = await resolveRuntimeConfig()
  return new DaemonClient({ ...config, surface: 'cli' })
}

function buildStreamChatOptions(fileIds?: string[]): ChatStreamOptions | undefined {
  return sanitizeChatOptions(
    {
      mode: DEFAULT_CHAT_OPTION_DEFAULTS.mode,
      fileIds,
    },
    DEFAULT_CHAT_OPTION_DEFAULTS,
  )
}

async function resolveRuntimeConfig(): Promise<RuntimeConfig> {
  await loadManagedEnv()
  return {
    baseUrl: process.env.SEPILOT_DAEMON_URL ?? 'http://127.0.0.1:17600',
    token: await loadToken(),
  }
}

async function loadManagedEnv(): Promise<void> {
  const envPath = join(process.cwd(), '.sepilot', 'extension.env')

  try {
    const raw = await readFile(envPath, 'utf-8')
    for (const line of raw.split(/\\r?\\n/)) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith('#')) {
        continue
      }

      const separator = trimmed.indexOf('=')
      if (separator <= 0) {
        continue
      }

      const key = trimmed.slice(0, separator).trim()
      const value = trimmed.slice(separator + 1)
      if (key && !process.env[key]) {
        process.env[key] = value
      }
    }
  } catch {
    // Managed env file is optional.
  }
}

async function loadToken(): Promise<string | null> {
  if (process.env.SEPILOT_DAEMON_TOKEN) {
    return process.env.SEPILOT_DAEMON_TOKEN
  }

  const tokenFile = process.env.SEPILOT_DAEMON_TOKEN_FILE
    ?? join(homedir(), '.sepilotd', 'security', 'daemon.token')

  try {
    return (await readFile(tokenFile, 'utf-8')).trim()
  } catch {
    return null
  }
}

function joinArgs(args: string[], usage: string): string {
  if (args.length === 0) {
    throw new Error(\`Usage: \${usage}\`)
  }
  return args.join(' ')
}

function printHelp(): void {
  console.log(\`${packageName}

Usage:
  node dist/index.js health
  node dist/index.js chat <message...>
  node dist/index.js stream <message...>

Environment:
  SEPILOT_DAEMON_URL
  SEPILOT_DAEMON_TOKEN
  SEPILOT_DAEMON_TOKEN_FILE\`)
}

void main()
`
}

function createFullIndexSource(packageName: string): string {
  return `import { readFile } from 'node:fs/promises'
import { createInterface } from 'node:readline/promises'
import { homedir } from 'node:os'
import { join, basename, extname } from 'node:path'
import { stdin, stdout } from 'node:process'
import {
  DEFAULT_CHAT_OPTION_DEFAULTS,
  DaemonClient,
  createTerminalChatStreamRenderer,
  normalizeWsEvent,
  sanitizeChatOptions,
  streamDaemonChatEvents,
  type ChatStreamOptions,
  type DaemonWsEvent,
} from '@sepilotd/api-client'
import { DaemonWsClient } from '@sepilotd/api-client/node'

interface RuntimeConfig {
  baseUrl: string
  token: string | null
}

async function main(): Promise<void> {
  const [command, ...rest] = process.argv.slice(2)
  const resolvedCommand = command ?? 'help'

  try {
    switch (resolvedCommand) {
      case 'help':
      case '--help':
      case '-h':
        printHelp()
        return
      case 'health':
        await runHealth()
        return
      case 'chat':
        await runChat(rest)
        return
      case 'stream':
        await runStream(rest)
        return
      case 'review-file':
        await runReviewFile(rest)
        return
      case 'ws-chat':
        await runWsChat(rest)
        return
      default:
        throw new Error(\`Unknown command: \${resolvedCommand}\`)
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    console.error(\`Error: \${message}\`)
    process.exitCode = 1
  }
}

async function runHealth(): Promise<void> {
  const client = await createDaemonClient()
  const health = await client.health()
  console.log(\`status: \${health.status}\`)
  console.log(\`version: \${health.version}\`)
}

async function runChat(args: string[]): Promise<void> {
  await runStream(args)
}

async function runStream(args: string[]): Promise<void> {
  const message = joinArgs(args, 'stream <message...>')
  const client = await createDaemonClient()
  const response = await client.chatStream(
    message,
    undefined,
    buildStreamChatOptions(),
  )
  const renderer = createTerminalChatStreamRenderer({
    write: (text) => stdout.write(text),
  })

  for await (const event of streamDaemonChatEvents(response)) {
    renderer.handleEvent(event)
  }
}

async function runReviewFile(args: string[]): Promise<void> {
  const [filePath, ...promptParts] = args
  if (!filePath) {
    throw new Error('Usage: review-file <path> [prompt...]')
  }

  const prompt = promptParts.length > 0
    ? promptParts.join(' ')
    : 'Review this file and summarize the important points.'

  const client = await createDaemonClient()
  const content = await readFile(filePath)
  const uploaded = await client.uploadFiles([
    {
      filename: basename(filePath),
      content,
      mimeType: guessMimeType(filePath),
    },
  ])

  const response = await client.chatStream(
    prompt,
    undefined,
    buildStreamChatOptions(uploaded.files.map((file) => file.id)),
  )
  const renderer = createTerminalChatStreamRenderer({
    write: (text) => stdout.write(text),
  })

  console.log(\`uploaded: \${uploaded.files.map((file) => file.filename).join(', ')}\`)

  for await (const event of streamDaemonChatEvents(response)) {
    renderer.handleEvent(event)
  }
}

async function runWsChat(args: string[]): Promise<void> {
  const message = joinArgs(args, 'ws-chat <message...>')
  const config = await resolveRuntimeConfig()

  if (!config.token) {
    throw new Error('Daemon token is required for WebSocket chat')
  }

  const client = new DaemonWsClient({
    baseUrl: config.baseUrl,
    token: config.token,
    surface: 'cli',
  })

  const rl = createInterface({ input: stdin, output: stdout })

  try {
    await client.connect()
    const renderer = createTerminalChatStreamRenderer({
      write: (text) => stdout.write(text),
    })
    for await (const event of client.chat(message)) {
      await printWsEvent(event, client, rl, renderer)
    }
  } finally {
    rl.close()
    client.close()
  }
}

async function createDaemonClient(): Promise<DaemonClient> {
  const config = await resolveRuntimeConfig()
  return new DaemonClient(config)
}

function buildStreamChatOptions(fileIds?: string[]): ChatStreamOptions | undefined {
  return sanitizeChatOptions(
    {
      mode: DEFAULT_CHAT_OPTION_DEFAULTS.mode,
      fileIds,
    },
    DEFAULT_CHAT_OPTION_DEFAULTS,
  )
}

async function resolveRuntimeConfig(): Promise<RuntimeConfig> {
  await loadManagedEnv()
  return {
    baseUrl: process.env.SEPILOT_DAEMON_URL ?? 'http://127.0.0.1:17600',
    token: await loadToken(),
  }
}

async function loadManagedEnv(): Promise<void> {
  const envPath = join(process.cwd(), '.sepilot', 'extension.env')

  try {
    const raw = await readFile(envPath, 'utf-8')
    for (const line of raw.split(/\\r?\\n/)) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith('#')) {
        continue
      }

      const separator = trimmed.indexOf('=')
      if (separator <= 0) {
        continue
      }

      const key = trimmed.slice(0, separator).trim()
      const value = trimmed.slice(separator + 1)
      if (key && !process.env[key]) {
        process.env[key] = value
      }
    }
  } catch {
    // Managed env file is optional.
  }
}

async function loadToken(): Promise<string | null> {
  if (process.env.SEPILOT_DAEMON_TOKEN) {
    return process.env.SEPILOT_DAEMON_TOKEN
  }

  const tokenFile = process.env.SEPILOT_DAEMON_TOKEN_FILE
    ?? join(homedir(), '.sepilotd', 'security', 'daemon.token')

  try {
    return (await readFile(tokenFile, 'utf-8')).trim()
  } catch {
    return null
  }
}

function joinArgs(args: string[], usage: string): string {
  if (args.length === 0) {
    throw new Error(\`Usage: \${usage}\`)
  }
  return args.join(' ')
}

function printHelp(): void {
  console.log(\`${packageName}

Usage:
  node dist/index.js health
  node dist/index.js chat <message...>
  node dist/index.js stream <message...>
  node dist/index.js review-file <path> [prompt...]
  node dist/index.js ws-chat <message...>

Environment:
  SEPILOT_DAEMON_URL
  SEPILOT_DAEMON_TOKEN
  SEPILOT_DAEMON_TOKEN_FILE\`)
}

async function printWsEvent(
  event: DaemonWsEvent,
  client: DaemonWsClient,
  rl: ReturnType<typeof createInterface>,
  renderer: ReturnType<typeof createTerminalChatStreamRenderer>,
): Promise<void> {
  switch (event.type) {
    case 'agent.approval_request': {
      renderer.handleEvent(normalizeWsEvent(event)!)
      const answer = await rl.question(
        \`\\nApprove tool "\${event.toolCall.name}"? [y/N] \`,
      )
      client.respondApproval(
        event.requestId,
        answer.trim().toLowerCase() === 'y',
      )
      return
    }
    case 'pong':
      console.log('[pong]')
      return
    default: {
      const normalized = normalizeWsEvent(event)
      if (normalized) {
        renderer.handleEvent(normalized)
      }
      return
    }
  }
}

function guessMimeType(path: string): string {
  switch (extname(path).toLowerCase()) {
    case '.md':
    case '.txt':
    case '.ts':
    case '.tsx':
    case '.js':
    case '.jsx':
    case '.json':
    case '.yml':
    case '.yaml':
      return 'text/plain'
    case '.png':
      return 'image/png'
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg'
    case '.gif':
      return 'image/gif'
    case '.pdf':
      return 'application/pdf'
    default:
      return 'application/octet-stream'
  }
}

void main()
`
}
