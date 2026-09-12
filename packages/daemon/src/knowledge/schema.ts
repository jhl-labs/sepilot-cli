import { z } from 'zod'

export const KnowledgeKind = z.enum([
  'fact',
  'concept',
  'decision',
  'procedure',
  'preference',
  'category',
])
export const KnowledgeRelation = z
  .object({
    targetId: z.string().min(1).max(200),
    type: z.enum(['related', 'supports', 'contradicts', 'supersedes']),
  })
  .strict()
export const KnowledgeSource = z
  .object({
    kind: z.enum(['conversation', 'wiki', 'memory', 'manual']),
    id: z.string().min(1).max(200),
    title: z.string().max(500),
    excerpt: z.string().min(1).max(24000),
    capturedAt: z.number().int().nonnegative(),
  })
  .strict()
export const KnowledgeWrite = z
  .object({
    id: z.string().min(1).max(200).optional(),
    expectedRevision: z.number().int().positive().optional(),
    title: z.string().trim().min(1).max(300),
    body: z.string().trim().min(1).max(24000),
    kind: KnowledgeKind,
    status: z.enum(['candidate', 'accepted', 'archived']),
    parentId: z.string().min(1).max(200).nullable(),
    tags: z.array(z.string().trim().min(1).max(100)).max(30),
    relations: z.array(KnowledgeRelation).max(30),
    reason: z.string().trim().min(1).max(1000),
  })
  .strict()
export const KnowledgeRecord = KnowledgeWrite.omit({ expectedRevision: true }).extend({
  id: z.string(),
  sources: z.array(KnowledgeSource).max(20),
  revision: z.number().int().positive(),
  createdAt: z.number().int().nonnegative(),
  updatedAt: z.number().int().nonnegative(),
})
export type KnowledgeRecord = z.infer<typeof KnowledgeRecord>
export type KnowledgeWrite = z.infer<typeof KnowledgeWrite>
export type KnowledgeSource = z.infer<typeof KnowledgeSource>
export const KnowledgeCapture = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('conversation'), id: z.string().min(1).max(200) }).strict(),
  z.object({ kind: z.literal('wiki'), id: z.string().min(1).max(200) }).strict(),
  z
    .object({
      kind: z.enum(['memory', 'manual']),
      id: z.string().min(1).max(200),
      title: z.string().trim().min(1).max(300),
      excerpt: z.string().trim().min(1).max(24000),
    })
    .strict(),
])
export const KnowledgeSuggestion = z
  .object({
    id: z.string(),
    expectedRevision: z.number().int().positive(),
    parentId: z.string().nullable(),
    tags: z.array(z.string().trim().min(1).max(100)).max(30),
    relations: z.array(KnowledgeRelation).max(30),
    reason: z.string().trim().min(1).max(1000),
  })
  .strict()

export function knowledgeError(message: string, statusCode = 400): never {
  throw Object.assign(new Error(message), { statusCode })
}

export const KnowledgeReview = z.object({
  autoOrganize: z.boolean(),
  status: z.enum(['idle', 'running', 'ready', 'failed']),
  suggestions: z.array(KnowledgeSuggestion).max(30),
  error: z.string().nullable(),
  updatedAt: z.number(),
})
export type KnowledgeReview = z.infer<typeof KnowledgeReview>
