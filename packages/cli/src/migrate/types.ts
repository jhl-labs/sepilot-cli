/**
 * Shape of a detected legacy sepilot-desktop installation. Only used
 * by `source-detect.ts` for the `sepilot migrate --source <path>`
 * resolution path. The daemon's MigrationRunner now owns the rest of
 * the migration vocabulary (steps, contexts, reports), so the CLI no
 * longer needs MigrationContext / StepReport / MigrationStep.
 */
export interface SourceManifest {
  root: string
  version: 1
  features: {
    conversations: boolean
    rag: boolean
    wiki: boolean
    persona: boolean
    snippets: boolean
    prompts: boolean
    personalDocs: boolean
    skills: boolean
    extensions: boolean
    settings: boolean
  }
}
