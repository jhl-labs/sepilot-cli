import type { SkillMetadata, SkillSignatureRecord, SkillSourceRecord } from '@sepilotd/core'

export type SkillRef =
  | { type: 'marketplace'; marketplace: string | null; name: string }
  | { type: 'git'; repo: string; branch: string | null; path: string | null }
  | { type: 'url'; url: string }

export interface FetchedSkill {
  metadata: SkillMetadata
  content: string
  source: SkillSourceRecord
  publisher?: string
  sourceRef?: string
  // ed25519 signature advertised by the source (frontmatter/manifest). It is
  // only trusted after the install pipeline verifies it against a trusted key.
  signature?: SkillSignatureRecord
}

export interface SkillSource {
  fetch(ref: SkillRef): Promise<FetchedSkill[]>
}
