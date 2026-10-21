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
