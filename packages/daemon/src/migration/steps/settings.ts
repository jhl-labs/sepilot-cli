import {
  existsSync,
  mkdirSync,
  writeFileSync,
} from 'node:fs'
import { join } from 'node:path'
import yaml from 'yaml'
import { safeReadFileSync } from '../safe-fs.js'
import type {
  MigrationContext,
  MigrationStepDefinition,
} from '../types.js'

interface SourceSettings {
  theme?: string
  locale?: string
  shortcuts?: Record<string, string>
  llm?: { providerId?: string; modelId?: string }
}

// Normalize a desktop locale (e.g. 'en', 'en-US', 'ko-KR') to an
// agent.outputLanguage value. Strips the region subtag; an empty/whitespace
// locale yields undefined so it is not mapped.
function localeToOutputLanguage(locale: string): string | undefined {
  const base = locale.trim().split(/[-_]/)[0]?.trim().toLowerCase()
  return base ? base : undefined
}

export const settingsStep: MigrationStepDefinition = {
  name: 'settings',
  async validate(_ctx: MigrationContext): Promise<void> {},
  async execute(ctx) {
    const file = join(ctx.sourcePath, 'settings.json')
    if (!existsSync(file)) return { copied: 0, skipped: 0, errors: [] }
    const src = JSON.parse(safeReadFileSync(file, 'utf-8') as string) as SourceSettings
    const targetFile = join(ctx.targetHome, 'config.yaml')
    let current: Record<string, unknown> = {}
    if (existsSync(targetFile)) {
      try {
        current = yaml.parse(safeReadFileSync(targetFile, 'utf-8') as string) ?? {}
      } catch {
        current = {}
      }
    }
    const next: Record<string, unknown> = { ...current }

    function setIfAllowed(key: string, value: unknown): void {
      if (value === undefined) return
      if (current[key] !== undefined && ctx.conflict === 'skip') return
      next[key] = value
    }

    setIfAllowed('theme', src.theme)
    setIfAllowed('locale', src.locale)
    // Map the imported desktop locale onto agent.outputLanguage so the daemon
    // answers in the user's chosen language. Previously the top-level `locale`
    // was dropped by the config schema on load, silently losing the preference.
    if (src.locale !== undefined) {
      const language = localeToOutputLanguage(src.locale)
      if (language) {
        const currentAgent =
          current.agent && typeof current.agent === 'object'
            ? (current.agent as Record<string, unknown>)
            : undefined
        const alreadySet = currentAgent?.outputLanguage !== undefined
        if (!(alreadySet && ctx.conflict === 'skip')) {
          const nextAgent =
            next.agent && typeof next.agent === 'object'
              ? { ...(next.agent as Record<string, unknown>) }
              : {}
          nextAgent.outputLanguage = language
          next.agent = nextAgent
        }
      }
    }
    if (src.llm) setIfAllowed('llm', src.llm)
    if (src.shortcuts) setIfAllowed('shortcuts', src.shortcuts)

    if (ctx.dryRun) return { copied: 0, skipped: 0, errors: [] }
    mkdirSync(ctx.targetHome, { recursive: true })
    writeFileSync(targetFile, yaml.stringify(next), 'utf-8')
    return { copied: 1, skipped: 0, errors: [] }
  },
}
