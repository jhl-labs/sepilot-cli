import {
  existsSync,
  readdirSync,
  statSync,
} from 'node:fs'
import { join } from 'node:path'
import { safeReadFileSync } from '../safe-fs.js'
import type {
  MigrationContext,
  MigrationStepDefinition,
} from '../types.js'

export const ragStep: MigrationStepDefinition = {
  name: 'rag',
  async validate(_ctx: MigrationContext): Promise<void> {},
  async execute(ctx) {
    const src = join(ctx.sourcePath, 'knowledge')
    if (!existsSync(src)) return { copied: 0, skipped: 0, errors: [] }
    if (!ctx.daemonOrigin) {
      return {
        copied: 0,
        skipped: 0,
        errors: [
          { path: '', error: 'daemonOrigin required for rag step' },
        ],
      }
    }
    const headers: HeadersInit = ctx.daemonToken
      ? {
          authorization: `Bearer ${ctx.daemonToken}`,
          'content-type': 'application/json',
        }
      : { 'content-type': 'application/json' }
    let copied = 0
    const errors: { path: string; error: string }[] = []
    for (const folder of readdirSync(src)) {
      const dir = join(src, folder)
      if (!statSync(dir).isDirectory()) continue
      try {
        if (!ctx.dryRun) {
          const fr = await fetch(`${ctx.daemonOrigin}/rag/folders`, {
            method: 'POST',
            headers,
            body: JSON.stringify({ id: folder, name: folder }),
          })
          if (!fr.ok && fr.status !== 409) {
            errors.push({ path: folder, error: String(fr.status) })
            continue
          }
        }
        for (const file of readdirSync(dir)) {
          const filePath = join(dir, file)
          if (
            !statSync(filePath).isFile() ||
            !/\.(md|txt|markdown)$/i.test(file)
          ) {
            continue
          }
          const body = safeReadFileSync(filePath, 'utf-8') as string
          if (!ctx.dryRun) {
            const dr = await fetch(`${ctx.daemonOrigin}/rag/documents`, {
              method: 'POST',
              headers,
              body: JSON.stringify({ folderId: folder, title: file, body }),
            })
            if (!dr.ok && dr.status !== 409) {
              errors.push({
                path: `${folder}/${file}`,
                error: String(dr.status),
              })
              continue
            }
          }
          copied++
        }
      } catch (err) {
        errors.push({ path: folder, error: (err as Error).message })
      }
    }
    return { copied, skipped: 0, errors }
  },
}
