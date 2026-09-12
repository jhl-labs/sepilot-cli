import { z } from 'zod'

export const PromptTemplateInput = z.object({
  id: z.string().optional(),
  title: z.string().min(1),
  body: z.string().default(''),
})
export type PromptTemplateInput = z.infer<typeof PromptTemplateInput>

export interface PromptTemplate {
  id: string
  title: string
  body: string
}
