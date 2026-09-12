import { conversationsStep } from './conversations.js'
import { personaStep } from './persona.js'
import { personalDocsStep } from './personal-docs.js'
import { promptsStep } from './prompts.js'
import { ragStep } from './rag.js'
import { settingsStep } from './settings.js'
import { skillsStep } from './skills.js'
import { snippetsStep } from './snippets.js'
import { wikiStep } from './wiki.js'
import type { MigrationStepDefinition } from '../types.js'

export const ALL_STEPS: MigrationStepDefinition[] = [
  conversationsStep,
  settingsStep,
  personaStep,
  promptsStep,
  personalDocsStep,
  ragStep,
  skillsStep,
  snippetsStep,
  wikiStep,
]

export const STEP_BY_NAME = new Map(
  ALL_STEPS.map((s) => [s.name, s] as const),
)
