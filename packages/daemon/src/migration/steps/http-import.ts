import {
  existsSync,
  readdirSync,
  statSync,
} from 'node:fs'
import { join } from 'node:path'
import type { MigrationContext } from '../types.js'

interface HttpImportEntry {
  filePath: string
  name: string
}

interface HttpImportOptions {
  acceptedFile: RegExp
  buildPayload(entry: HttpImportEntry): unknown
  endpoint: string
  sourceDir: string
  stepName: string
}

export async function executeHttpImportStep(
  ctx: MigrationContext,
  options: HttpImportOptions,
): Promise<{
  copied: number
  skipped: number
  errors: { path: string; error: string }[]
}> {
  const src = join(ctx.sourcePath, options.sourceDir)
  if (!existsSync(src)) return { copied: 0, skipped: 0, errors: [] }
  if (!ctx.daemonOrigin) {
    return {
      copied: 0,
      skipped: 0,
      errors: [
        { path: '', error: `daemonOrigin required for ${options.stepName} step` },
      ],
    }
  }

  const headers = buildJsonHeaders(ctx)
  let copied = 0
  const errors: { path: string; error: string }[] = []

  for (const name of readdirSync(src)) {
    const filePath = join(src, name)
    if (!statSync(filePath).isFile() || !options.acceptedFile.test(name)) {
      continue
    }

    try {
      const payload = options.buildPayload({ filePath, name })
      if (!ctx.dryRun) {
        const response = await fetch(`${ctx.daemonOrigin}${options.endpoint}`, {
          method: 'POST',
          headers,
          body: JSON.stringify(payload),
        })
        if (!response.ok && response.status !== 409) {
          errors.push({ path: name, error: String(response.status) })
          continue
        }
      }
      copied++
    } catch (err) {
      errors.push({ path: name, error: (err as Error).message })
    }
  }

  return { copied, skipped: 0, errors }
}

function buildJsonHeaders(ctx: MigrationContext): HeadersInit {
  return ctx.daemonToken
    ? {
        authorization: `Bearer ${ctx.daemonToken}`,
        'content-type': 'application/json',
      }
    : { 'content-type': 'application/json' }
}
