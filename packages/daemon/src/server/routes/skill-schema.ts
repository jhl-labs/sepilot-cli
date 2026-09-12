import { AutonomyLevel } from '@sepilotd/core'
import { z } from 'zod'

export const skillMetadataSchema = z.object({
  id: z.string(),
  name: z.string().min(1),
  version: z.string().min(1),
  description: z.string().min(1),
  author: z.string().optional(),
  tags: z.array(z.string()).optional(),
  tools: z.array(z.string()),
  autonomy_required: z.nativeEnum(AutonomyLevel).optional(),
  created: z.string().optional(),
  enabled: z.boolean().optional(),
  source: z.object({
    type: z.enum(['marketplace', 'git', 'url']),
    ref: z.string().min(1),
  }).optional(),
  provenance: z.object({
    source: z.object({
      type: z.enum(['marketplace', 'git', 'url']),
      ref: z.string().min(1),
    }),
    digest: z.string().min(1),
    installedAt: z.string().optional(),
    verified: z.boolean(),
    verification: z.enum(['builtin', 'digest', 'signature', 'manual', 'unverified']),
    publisher: z.string().optional(),
    sourceRef: z.string().optional(),
    signature: z.object({
      algorithm: z.literal('ed25519'),
      keyId: z.string().optional(),
      value: z.string().min(1),
      verified: z.boolean().optional(),
    }).optional(),
    scan: z.object({
      result: z.enum(['pass', 'warn', 'fail']),
      checkedAt: z.string(),
      errors: z.array(z.string()).optional(),
      warnings: z.array(z.string()).optional(),
    }).optional(),
  }).optional(),
  risk_tier: z.enum(['low', 'medium', 'high']).optional(),
  permissions: z.object({
    tools: z.array(z.string()).optional(),
    network: z.array(z.string()).optional(),
    files: z.array(z.string()).optional(),
  }).optional(),
})

export const skillValidationRequestSchema = z.object({
  metadata: skillMetadataSchema,
  content: z.string().min(1),
})

export const skillsSearchQuerySchema = z.object({
  query: z.string().min(1),
  includeBuiltins: z.enum(['true', 'false']).optional(),
  cwd: z.string().trim().min(1).optional(),
  workspaceRoot: z.string().trim().min(1).optional(),
})
