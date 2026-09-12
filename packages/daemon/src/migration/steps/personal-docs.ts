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

export const personalDocsStep: MigrationStepDefinition = {
  name: 'personal-docs',
  async validate(_ctx: MigrationContext): Promise<void> {},
  async execute(ctx) {
    const src = join(ctx.sourcePath, 'docs')
    if (!existsSync(src)) return { copied: 0, skipped: 0, errors: [] }
    if (!ctx.daemonOrigin) {
      return {
        copied: 0,
        skipped: 0,
        errors: [
          {
            path: '',
            error: 'daemonOrigin required for personal-docs step',
          },
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
    const targets: { rel: string; body: string }[] = []
    function walk(dir: string, relPrefix: string): void {
      for (const name of readdirSync(dir)) {
        const p = join(dir, name)
        const st = statSync(p)
        const rel = relPrefix ? `${relPrefix}/${name}` : name
        if (st.isDirectory()) {
          walk(p, rel)
          continue
        }
        if (!/\.(md|markdown|txt)$/i.test(name)) continue
        try {
          const body = safeReadFileSync(p, 'utf-8') as string
          targets.push({ rel, body })
        } catch (err) {
          errors.push({ path: rel, error: (err as Error).message })
        }
      }
    }
    walk(src, '')
    for (const { rel, body } of targets) {
      const payload = { path: rel, content: body }
      if (!ctx.dryRun) {
        try {
          const r = await fetch(`${ctx.daemonOrigin}/personal-docs`, {
            method: 'POST',
            headers,
            body: JSON.stringify(payload),
          })
          if (!r.ok && r.status !== 409) {
            errors.push({ path: rel, error: String(r.status) })
            continue
          }
        } catch (err) {
          errors.push({ path: rel, error: (err as Error).message })
          continue
        }
      }
      copied++
    }
    return { copied, skipped: 0, errors }
  },
}
