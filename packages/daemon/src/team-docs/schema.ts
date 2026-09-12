import { z } from 'zod'
import { safeIdSchema } from '../utils/safe-id.js'

export const TeamDocsConfigInput = z.object({
  id: safeIdSchema.optional(),
  name: z.string().min(1),
  description: z.string().default(''),
  serverType: z.enum(['github.com', 'ghes']).default('github.com'),
  ghesUrl: z.string().default(''),
  token: z.string().min(1),
  owner: z.string().min(1),
  repo: z.string().min(1),
  branch: z.string().min(1).default('main'),
  docsPath: z.string().min(1).default('sepilot/documents'),
  enabled: z.boolean().default(true),
  autoSync: z.boolean().default(false),
  syncInterval: z.number().int().min(5).max(24 * 60).default(60),
})

export type TeamDocsConfigInput = z.infer<typeof TeamDocsConfigInput>

export interface TeamDocsConfig {
  id: string
  name: string
  description: string
  serverType: 'github.com' | 'ghes'
  ghesUrl: string
  token: string
  owner: string
  repo: string
  branch: string
  docsPath: string
  enabled: boolean
  autoSync: boolean
  syncInterval: number
  lastTestedAt: number | null
  lastSyncAt: number | null
  lastSyncStatus: 'success' | 'error' | null
  lastSyncError: string | null
  syncedDocuments: number
  updatedAt: number
}

export interface TeamDocsDocument {
  path: string
  sha: string | null
  size: number
  syncedAt: number
}

export interface TeamDocsDocumentContent extends TeamDocsDocument {
  content: string
}
