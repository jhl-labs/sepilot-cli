import { safeReadFileSync } from '../safe-fs.js'
import type {
  MigrationContext,
  MigrationStepDefinition,
} from '../types.js'
import { executeHttpImportStep } from './http-import.js'

interface SourcePrompt {
  id?: string
  title?: string
  body?: string
  content?: string
}

export const promptsStep: MigrationStepDefinition = {
  name: 'prompts',
  async validate(_ctx: MigrationContext): Promise<void> {},
  async execute(ctx) {
    return executeHttpImportStep(ctx, {
      acceptedFile: /\.(json|md|markdown)$/i,
      endpoint: '/prompt-templates',
      sourceDir: 'prompts',
      stepName: 'prompts',
      buildPayload({ filePath, name }) {
        let payload: { id?: string; title: string; body: string }
        if (name.endsWith('.json')) {
          const raw = JSON.parse((safeReadFileSync(filePath, 'utf-8') as string)) as SourcePrompt
          payload = {
            id: raw.id,
            title: raw.title ?? name.replace(/\.json$/i, ''),
            body: raw.body ?? raw.content ?? '',
          }
        } else {
          payload = {
            title: name.replace(/\.(md|markdown)$/i, ''),
            body: (safeReadFileSync(filePath, 'utf-8') as string),
          }
        }
        return payload
      },
    })
  },
}
