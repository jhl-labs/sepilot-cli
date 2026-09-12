import { z } from 'zod'

export const PersonaInput = z.object({
  id: z.string().optional(),
  name: z.string().min(1),
  systemPrompt: z.string().default(''),
  isActive: z.boolean().optional(),
  memoryScope: z.enum(['shared', 'isolated']).optional(),
})
export type PersonaInput = z.infer<typeof PersonaInput>

export interface Persona {
  id: string
  name: string
  systemPrompt: string
  isActive: boolean
  memoryScope?: 'shared' | 'isolated'
}
