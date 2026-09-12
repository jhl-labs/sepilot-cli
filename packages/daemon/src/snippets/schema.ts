import { z } from 'zod'

export const SnippetInput = z.object({
  id: z.string().optional(),
  title: z.string().min(1),
  language: z.string().min(1),
  body: z.string().min(0),
  tags: z.array(z.string()).default([]),
})
export type SnippetInput = z.infer<typeof SnippetInput>

export interface SnippetGistLink {
  id: string
  file: string
  htmlUrl: string | null
  updatedAt: string | null
  syncedAt: number | null
}

export interface Snippet {
  id: string
  title: string
  language: string
  body: string
  tags: string[]
  gist: SnippetGistLink | null
}
