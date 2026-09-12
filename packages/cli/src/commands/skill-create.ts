import { ApiHttpError, type DaemonSkillCreateInput, type DaemonSkillCreateResult } from '@sepilotd/api-client'
import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { friendlyErrorMessage as errorMessage, printApiError } from '../utils/error-message.js'

export interface SkillCreateCommandOptions {
  url?: string
  description?: string
  tools?: string
  force?: boolean
}

export interface SkillCreateClient {
  createSkill(req: DaemonSkillCreateInput): Promise<DaemonSkillCreateResult>
}

interface SkillValidationResult {
  errors?: string[]
  warnings?: string[]
}

function slugifySkillId(input: string): string {
  return input
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function titleizeSkillName(input: string): string {
  return input
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

function parseTools(input?: string): string[] {
  const raw = input ?? 'terminal.run'
  return raw
    .split(',')
    .map((tool) => tool.trim())
    .filter(Boolean)
}

function extractSkillValidation(err: unknown): SkillValidationResult | null {
  if (!(err instanceof ApiHttpError)) return null
  try {
    const body = JSON.parse(err.rawBody) as {
      error?: {
        code?: string
        validation?: SkillValidationResult
      }
    }
    if (body.error?.code !== 'VALIDATION_FAILED') return null
    return body.error.validation ?? null
  } catch {
    return null
  }
}

function printSkillCreateError(err: unknown): void {
  const validation = extractSkillValidation(err)
  if (validation) {
    for (const e of validation.errors ?? []) console.error(chalk.red(`error: ${e}`))
    for (const w of validation.warnings ?? []) console.error(chalk.yellow(`warning: ${w}`))
    console.error(chalk.gray('Use --force to create anyway.'))
    return
  }

  if (!printApiError(err)) {
    console.error(chalk.red(`create failed: ${errorMessage(err)}`))
  }
}

export function buildSkillCreateInput(
  name: string,
  options: Pick<SkillCreateCommandOptions, 'description' | 'tools' | 'force'> = {},
): DaemonSkillCreateInput {
  const id = slugifySkillId(name)
  if (!id) throw new Error('Skill name must contain at least one letter or number.')

  const description = options.description?.trim() || `${name} skill`
  const tools = parseTools(options.tools)
  if (tools.length === 0) throw new Error('At least one tool is required.')

  const content = `# ${titleizeSkillName(name)}

${description}

## Usage

Describe how to use this skill and what input it expects.

## Steps

1. First step
2. Second step
3. Third step

## Examples

\`\`\`
sepilot run ${name} --input "your task here"
\`\`\`
`

  return {
    metadata: {
      id,
      name,
      version: '1.0.0',
      description,
      tools,
      tags: [],
    },
    content,
    ...(options.force ? { force: true } : {}),
  }
}

export async function skillCreateCommandImpl(
  name: string,
  options: SkillCreateCommandOptions,
  client: SkillCreateClient,
): Promise<void> {
  let request: DaemonSkillCreateInput
  try {
    request = buildSkillCreateInput(name, options)
  } catch (err) {
    console.error(chalk.red(errorMessage(err)))
    process.exit(1)
  }

  try {
    const result = await client.createSkill(request)
    output(
      { ok: true, id: result.id, version: result.version },
      () => [
        chalk.green(`Skill created: ${chalk.bold(request.metadata.name)} v${result.version}`),
        chalk.gray('Stored by daemon and available immediately.'),
        chalk.cyan('  sepilot skills list'),
        chalk.cyan(`  sepilot run ${request.metadata.name} --input "your task"`),
      ].join('\n'),
    )
  } catch (err) {
    printSkillCreateError(err)
    process.exit(1)
  }
}

export async function skillCreateCommand(name: string, options: SkillCreateCommandOptions) {
  const client = new DaemonClient(options.url)
  await skillCreateCommandImpl(name, options, client)
}
