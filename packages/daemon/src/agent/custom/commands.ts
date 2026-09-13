import type { CustomDef } from './loader.js'

export interface CustomCommand {
  id: string
  description?: string
  agent?: string
  model?: string
  template: string
  source: string
}

export function compileCustomCommands(defs: CustomDef[]): CustomCommand[] {
  return defs.filter((d) => !d.error).map((d) => ({
    id: d.id,
    description: typeof d.data.description === 'string' ? d.data.description : undefined,
    agent: typeof d.data.agent === 'string' ? d.data.agent : undefined,
    model: typeof d.data.model === 'string' ? d.data.model : undefined,
    template: d.body.trim(),
    source: d.source,
  }))
}
