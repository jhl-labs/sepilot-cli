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

export const skillsStep: MigrationStepDefinition = {
  name: 'skills',
  async validate(_ctx: MigrationContext): Promise<void> {},
  async execute(ctx) {
    const src = join(ctx.sourcePath, 'skills')
    if (!existsSync(src)) return { copied: 0, skipped: 0, errors: [] }
    const targetRoot = join(ctx.targetHome, 'skills')
    let copied = 0
    let skipped = 0
    const errors: { path: string; error: string }[] = []
    for (const name of readdirSync(src)) {
      const dir = join(src, name)
      if (!statSync(dir).isDirectory()) continue
      const skillMd = join(dir, 'SKILL.md')
      if (!existsSync(skillMd)) continue
      const targetDir = join(targetRoot, name)
      const targetMd = join(targetDir, 'SKILL.md')
      if (existsSync(targetMd) && ctx.conflict === 'skip') {
        skipped++
        continue
      }
      if (ctx.dryRun) {
        copied++
        continue
      }
      try {
        mkdirSync(targetDir, { recursive: true })
        safeCopyFileSync(skillMd, targetMd)
        copied++
      } catch (err) {
        errors.push({ path: name, error: (err as Error).message })
      }
    }
    return { copied, skipped, errors }
  },
}
