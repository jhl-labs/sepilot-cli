import { safeReadFileSync } from '../safe-fs.js'
import type {
  MigrationContext,
  MigrationStepDefinition,
} from '../types.js'
import { executeHttpImportStep } from './http-import.js'

interface SourcePersona {
  id?: string
  name?: string
  systemPrompt?: string
  prompt?: string
}

export const personaStep: MigrationStepDefinition = {
  name: 'persona',
  async validate(_ctx: MigrationContext): Promise<void> {},
  async execute(ctx) {
    return executeHttpImportStep(ctx, {
      acceptedFile: /\.json$/i,
      endpoint: '/persona',
      sourceDir: 'personas',
      stepName: 'persona',
      buildPayload({ filePath, name }) {
        const raw = JSON.parse(safeReadFileSync(filePath, 'utf-8') as string) as SourcePersona
        return {
          id: raw.id,
          name: raw.name ?? name.replace(/\.json$/i, ''),
          systemPrompt: raw.systemPrompt ?? raw.prompt ?? '',
        }
      },
    })
  },
}
