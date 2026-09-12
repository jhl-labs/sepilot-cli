import { z } from 'zod'
import { safeIdSchema } from '../utils/safe-id.js'

export const PersonalDocInput = z.object({
  id: safeIdSchema.optional(),
  path: z.string().min(1),
  content: z.string().default(''),
})
export type PersonalDocInput = z.infer<typeof PersonalDocInput>

export interface PersonalDoc {
  id: string
  path: string
  updatedAt: number
}

export interface PersonalDocContent extends PersonalDoc {
  content: string
}
