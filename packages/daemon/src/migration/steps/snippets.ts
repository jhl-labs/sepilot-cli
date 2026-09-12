import { safeReadFileSync } from '../safe-fs.js'
import type {
  MigrationContext,
  MigrationStepDefinition,
} from '../types.js'
import { executeHttpImportStep } from './http-import.js'

interface SourceSnippet {
  id?: string
  title?: string
  language?: string
  body?: string
  content?: string
  tags?: string[]
}

export const snippetsStep: MigrationStepDefinition = {
  name: 'snippets',
  async validate(_ctx: MigrationContext): Promise<void> {},
  async execute(ctx) {
    return executeHttpImportStep(ctx, {
      acceptedFile: /\.json$/i,
      endpoint: '/snippets',
      sourceDir: 'snippets',
      stepName: 'snippets',
      buildPayload({ filePath, name }) {
        const raw = JSON.parse(safeReadFileSync(filePath, 'utf-8') as string) as SourceSnippet
        return {
          id: raw.id,
          title: raw.title ?? name.replace(/\.json$/i, ''),
          language: raw.language ?? 'text',
          body: raw.body ?? raw.content ?? '',
          tags: raw.tags ?? [],
        }
      },
    })
  },
}
