import { watch, type FSWatcher } from 'node:fs'
import { mkdir, readFile, readdir } from 'node:fs/promises'
import { extname, join, resolve } from 'node:path'
import YAML from 'yaml'
import { z } from 'zod'
import { createLogger } from '../../logger.js'
import type { Deps } from './nodes.js'
import {
  builtinGraphBuilders,
  builtinGraphIds,
  type BuiltinGraphId,
} from './presets/index.js'
import type { GraphAgentInfo, GraphAgentRegistry } from './registry.js'

const log = createLogger('agent.graph.yaml')

const yamlGraphSchema = z.object({
  version: z.literal(1),
  id: z.string().trim().min(1).regex(/^[a-z0-9][a-z0-9_-]*$/i),
  name: z.string().trim().min(1),
  description: z.string().trim().min(1),
  base: z.enum(builtinGraphIds as [BuiltinGraphId, ...BuiltinGraphId[]]),
  systemPromptAppend: z.string().trim().min(1).optional(),
  limits: z.object({
    maxIterations: z.number().int().positive().max(1000).optional(),
  }).optional(),
}).strict()

type YamlGraphConfig = z.infer<typeof yamlGraphSchema>

interface LoadedYamlGraphAgent {
  filePath: string
  info: GraphAgentInfo
}

export interface QuarantinedYamlGraphAgent {
  filePath: string
  reason: string
  observedAt: string
}

function appendSystemPrompt(
  basePrompt?: string,
  appendedPrompt?: string,
): string | undefined {
  const base = basePrompt?.trim()
  const appendix = appendedPrompt?.trim()
  if (!base && !appendix) return undefined
  if (!base) return appendix
  if (!appendix) return base
  return `${base}\n\n${appendix}`
}

function parseYamlGraphConfig(raw: string): YamlGraphConfig {
  const parsed = YAML.parse(raw)
  return yamlGraphSchema.parse(parsed)
}

function createGraphInfo(config: YamlGraphConfig): GraphAgentInfo {
  return {
    id: config.id,
    name: config.name,
    description: config.description,
    source: 'yaml',
    limits: config.limits,
    builder: (deps: Deps) =>
      builtinGraphBuilders[config.base]({
        ...deps,
        systemPrompt: appendSystemPrompt(
          deps.systemPrompt,
          config.systemPromptAppend,
        ),
      }),
  }
}

function isYamlAgentFilename(path: string): boolean {
  const extension = extname(path).toLowerCase()
  return extension === '.yaml' || extension === '.yml'
}

export class YamlGraphAgentLoader {
  private watcher: FSWatcher | null = null
  private debounceTimer: ReturnType<typeof setTimeout> | null = null
  private loadedByFile = new Map<string, LoadedYamlGraphAgent>()
  private quarantinedByFile = new Map<string, QuarantinedYamlGraphAgent>()
  private reloadQueue: Promise<void> = Promise.resolve()

  constructor(
    private readonly agentsDir: string,
    private readonly registry: GraphAgentRegistry,
  ) {}

  async init(): Promise<void> {
    await mkdir(this.agentsDir, { recursive: true })
    await this.reload()
  }

  start(): void {
    try {
      this.watcher = watch(this.agentsDir, () => {
        if (this.debounceTimer) {
          clearTimeout(this.debounceTimer)
        }
        this.debounceTimer = setTimeout(() => {
          void this.reload()
        }, 150)
      })
    } catch {
      // The directory may not exist yet during early startup.
    }
  }

  stop(): void {
    if (this.watcher) {
      this.watcher.close()
      this.watcher = null
    }
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer)
      this.debounceTimer = null
    }
  }

  async reload(): Promise<void> {
    this.reloadQueue = this.reloadQueue
      .catch(() => {})
      .then(async () => {
        await this.performReload()
      })
    return this.reloadQueue
  }

  listQuarantined(): QuarantinedYamlGraphAgent[] {
    return Array.from(this.quarantinedByFile.values())
  }

  private async performReload(): Promise<void> {
    await mkdir(this.agentsDir, { recursive: true })

    const previousLoadedIds = new Set(
      Array.from(this.loadedByFile.values()).map((entry) => entry.info.id),
    )
    const reservedAgents = new Map<string, GraphAgentInfo>()
    for (const info of this.registry.list()) {
      if (info.source !== 'yaml' || !previousLoadedIds.has(info.id)) {
        reservedAgents.set(info.id, info)
      }
    }

    const nextLoadedByFile = new Map<string, LoadedYamlGraphAgent>()
    const nextQuarantinedByFile = new Map<string, QuarantinedYamlGraphAgent>()
    const claimedYamlIds = new Map<string, string>()
    const entries = await readdir(this.agentsDir, { withFileTypes: true })
    const yamlFiles = entries
      .filter((entry) => entry.isFile() && isYamlAgentFilename(entry.name))
      .map((entry) => resolve(join(this.agentsDir, entry.name)))
      .sort()

    for (const filePath of yamlFiles) {
      try {
        const raw = await readFile(filePath, 'utf-8')
        const config = parseYamlGraphConfig(raw)
        const conflictingAgent = reservedAgents.get(config.id)
        if (conflictingAgent) {
          throw new Error(
            `Agent id '${config.id}' conflicts with existing ${conflictingAgent.source ?? 'custom'} agent`,
          )
        }

        const duplicateFile = claimedYamlIds.get(config.id)
        if (duplicateFile) {
          throw new Error(
            `Agent id '${config.id}' is already declared in ${duplicateFile}`,
          )
        }

        const info = createGraphInfo(config)
        claimedYamlIds.set(info.id, filePath)
        nextLoadedByFile.set(filePath, {
          filePath,
          info,
        })
      } catch (error) {
        const reason = error instanceof Error ? error.message : String(error)
        nextQuarantinedByFile.set(filePath, {
          filePath,
          reason,
          observedAt: new Date().toISOString(),
        })
        log.warn('Ignoring invalid YAML graph agent', {
          path: filePath,
          error: reason,
        })
      }
    }

    const nextIds = new Set(
      Array.from(nextLoadedByFile.values()).map((entry) => entry.info.id),
    )
    for (const loaded of this.loadedByFile.values()) {
      if (!nextIds.has(loaded.info.id)) {
        this.registry.unregister(loaded.info.id)
      }
    }
    for (const loaded of nextLoadedByFile.values()) {
      this.registry.register(loaded.info)
    }

    this.loadedByFile = nextLoadedByFile
    this.quarantinedByFile = nextQuarantinedByFile
  }
}
