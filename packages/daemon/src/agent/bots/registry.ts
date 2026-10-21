import { externalCliFromEngine } from '../../external-cli/executor.js'
import { mkdir, readFile, readdir } from 'node:fs/promises'
import { extname, join } from 'node:path'
import YAML from 'yaml'
import { z } from 'zod'
import { createLogger } from '../../logger.js'
import { TOOL_EXPOSURE_GROUPS } from '../../tools/role-filter.js'
import type { Persona } from '../personas.js'
import { BUILTIN_ROLE_BOTS } from './catalog.js'
import {
  ROLE_BOT_STAGE_TIERS,
  summarizeRoleBot,
  type RoleBotProfile,
  type RoleBotStagePolicy,
  type RoleBotSummary,
} from './types.js'

const log = createLogger('role-bots')

const stagePolicySchema = z.object({
  contractPlanner: z.boolean(),
  outcomeReview: z.boolean(),
  completionGate: z.boolean(),
  recoveryController: z.boolean(),
}).strict()

export const roleBotYamlSchema = z.object({
  version: z.literal(1),
  id: z.string().regex(/^[a-z0-9][a-z0-9-]{1,63}$/),
  name: z.string().trim().min(1).max(80),
  description: z.string().trim().min(1).max(240),
  category: z.enum(['engineering', 'operations', 'quality', 'design', 'knowledge', 'personal', 'communication', 'creative']).default('personal'),
  systemPrompt: z.string().trim().min(1).max(8_000),
  engine: z.string().trim().min(1).max(64).default('react'),
  toolGroups: z.array(z.enum(TOOL_EXPOSURE_GROUPS)).max(TOOL_EXPOSURE_GROUPS.length).optional(),
  allowedTools: z.array(z.string().trim().min(1).max(128)).max(128).optional(),
  deniedTools: z.array(z.string().trim().min(1).max(128)).max(128).optional(),
  skills: z.array(z.string().trim().min(1).max(64)).max(8).optional(),
  maxIterations: z.number().int().min(1).max(200).optional(),
  tier: z.enum(['answer', 'assist', 'operate', 'build']).optional(),
  stages: stagePolicySchema.optional(),
  memory: z.enum(['off', 'ambient', 'focused']).optional(),
  documents: z.boolean().optional(),
  thinkingLevel: z.enum(['off', 'low', 'medium', 'high']).optional(),
  routable: z.boolean().default(true),
}).strict()

export type RoleBotYaml = z.infer<typeof roleBotYamlSchema>

export function roleBotFromYaml(input: RoleBotYaml): RoleBotProfile {
  const stages: RoleBotStagePolicy = input.stages
    ?? ROLE_BOT_STAGE_TIERS[input.tier ?? (input.engine === 'instant' ? 'answer' : 'assist')]
  return {
    id: input.id,
    name: input.name,
    description: input.description,
    category: input.category,
    systemPrompt: input.systemPrompt,
    engine: input.engine,
    ...(input.toolGroups ? { toolGroups: input.toolGroups } : {}),
    ...(input.allowedTools ? { allowedTools: input.allowedTools } : {}),
    ...(input.deniedTools ? { deniedTools: input.deniedTools } : {}),
    ...(input.skills ? { skills: input.skills } : {}),
    ...(input.maxIterations ? { maxIterations: input.maxIterations } : {}),
    ...(input.thinkingLevel ? { thinkingLevel: input.thinkingLevel } : {}),
    stages,
    memory: input.memory ?? (input.engine === 'instant' ? 'ambient' : 'focused'),
    documents: input.documents ?? input.engine !== 'instant',
    routable: input.routable,
    source: 'user',
  }
}

export function roleBotToPersona(bot: RoleBotProfile): Persona {
  return {
    id: bot.id,
    name: bot.name,
    description: bot.description,
    systemPromptAddition: bot.systemPrompt,
    ...(bot.allowedTools ? { allowedTools: [...bot.allowedTools] } : {}),
    ...(bot.deniedTools ? { deniedTools: [...bot.deniedTools] } : {}),
  }
}

export interface RoleBotRegistryOptions {
  botsDir?: string
  knownEngines?: () => Iterable<string>
  builtins?: readonly RoleBotProfile[]
}

export class RoleBotRegistry {
  private readonly bots = new Map<string, RoleBotProfile>()
  private readonly options: RoleBotRegistryOptions

  constructor(options: RoleBotRegistryOptions = {}) {
    this.options = options
    for (const entry of options.builtins ?? BUILTIN_ROLE_BOTS) this.bots.set(entry.id, entry)
  }

  async init(): Promise<void> {
    const dir = this.options.botsDir
    if (!dir) return
    await mkdir(dir, { recursive: true })
    const entries = await readdir(dir).catch(() => [] as string[])
    for (const entry of entries) {
      const ext = extname(entry).toLowerCase()
      if (ext !== '.yaml' && ext !== '.yml') continue
      const file = join(dir, entry)
      try {
        const parsed = roleBotYamlSchema.parse(YAML.parse(await readFile(file, 'utf8')))
        const bot = roleBotFromYaml(parsed)
        const issue = this.engineIssue(bot)
        if (issue) {
          log.warn('role_bot.rejected', { file: entry, id: bot.id, reason: issue })
          continue
        }
        this.bots.set(bot.id, bot)
        log.info('role_bot.loaded', { file: entry, id: bot.id, engine: bot.engine })
      } catch (error) {
        log.warn('role_bot.invalid', { file: entry, error: (error as Error).message })
      }
    }
  }

  private engineIssue(bot: RoleBotProfile): string | null {
    if (bot.engine === 'instant' || bot.engine === 'react') return null
    if (bot.engine.startsWith('external:')) return externalCliFromEngine(bot.engine) ? null : `unknown external CLI in engine "${bot.engine}"`
    const known = this.options.knownEngines
    if (!known) return null
    for (const id of known()) if (id === bot.engine) return null
    return `unknown engine "${bot.engine}"`
  }

  get(id: string | undefined): RoleBotProfile | undefined {
    return id ? this.bots.get(id) : undefined
  }

  has(id: string): boolean {
    return this.bots.has(id)
  }

  list(): RoleBotProfile[] {
    return [...this.bots.values()]
  }

  summaries(): RoleBotSummary[] {
    return this.list().map(summarizeRoleBot)
  }

  routable(): RoleBotProfile[] {
    return this.list().filter((entry) => entry.routable)
  }
}
