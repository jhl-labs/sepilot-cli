import {
  existsSync,
  mkdirSync,
  readdirSync,
  statSync,
} from 'node:fs'
import { join } from 'node:path'
import { safeCopyFileSync } from '../safe-fs.js'
import type {
  MigrationContext,
  MigrationStepDefinition,
} from '../types.js'

export const conversationsStep: MigrationStepDefinition = {
  name: 'conversations',
  async validate(_ctx: MigrationContext): Promise<void> {},
  async execute(ctx) {
    const src = join(ctx.sourcePath, 'conversations')
    if (!existsSync(src)) return { copied: 0, skipped: 0, errors: [] }
    const target = join(ctx.targetHome, 'sessions')
    let copied = 0
    let skipped = 0
    const errors: { path: string; error: string }[] = []
    for (const name of readdirSync(src)) {
      const file = join(src, name)
      if (!statSync(file).isFile() || !/\.(jsonl|json)$/i.test(name)) continue
      const targetName = name.endsWith('.json')
        ? name.replace(/\.json$/i, '.jsonl')
        : name
      const targetFile = join(target, targetName)
      if (existsSync(targetFile) && ctx.conflict === 'skip') {
        skipped++
        continue
      }
      if (ctx.dryRun) {
        copied++
        continue
      }
      try {
        mkdirSync(target, { recursive: true })
        safeCopyFileSync(file, targetFile)
        copied++
      } catch (err) {
        errors.push({ path: name, error: (err as Error).message })
      }
    }
    return { copied, skipped, errors }
  },
}
