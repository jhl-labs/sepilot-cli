import { safeReadFileSync } from '../safe-fs.js'
import type {
  MigrationContext,
  MigrationStepDefinition,
} from '../types.js'
import { executeHttpImportStep } from './http-import.js'

interface SourceWikiNode {
  id?: string
  parentId?: string | null
  title?: string
  icon?: string | null
  group?: string | null
  body?: string
}

export const wikiStep: MigrationStepDefinition = {
  name: 'wiki',
  async validate(_ctx: MigrationContext): Promise<void> {},
  async execute(ctx) {
    return executeHttpImportStep(ctx, {
      acceptedFile: /\.json$/i,
      endpoint: '/wiki/nodes',
      sourceDir: 'wiki',
      stepName: 'wiki',
      buildPayload({ filePath, name }) {
        const raw = JSON.parse(safeReadFileSync(filePath, 'utf-8') as string) as SourceWikiNode
        return {
          id: raw.id,
          parentId: raw.parentId ?? null,
          title: raw.title ?? name.replace(/\.json$/i, ''),
          icon: raw.icon ?? null,
          group: raw.group ?? null,
          body: raw.body ?? '',
        }
      },
    })
  },
}
