import type { ToolExposureGroup } from '../../tools/role-filter.js'

export interface RoleBotStagePolicy {
  contractPlanner: boolean
  outcomeReview: boolean
  completionGate: boolean
  recoveryController: boolean
}

export type RoleBotStageTier = 'answer' | 'assist' | 'operate' | 'build'

export const ROLE_BOT_STAGE_TIERS: Readonly<Record<RoleBotStageTier, RoleBotStagePolicy>> = {
  answer: { contractPlanner: false, outcomeReview: false, completionGate: false, recoveryController: false },
  assist: { contractPlanner: false, outcomeReview: true, completionGate: false, recoveryController: false },
  operate: { contractPlanner: true, outcomeReview: false, completionGate: true, recoveryController: true },
  build: { contractPlanner: true, outcomeReview: true, completionGate: true, recoveryController: true },
}

export type RoleBotMemoryRetrieval = 'off' | 'ambient' | 'focused'

export type RoleBotCategory =
  | 'engineering'
  | 'operations'
  | 'quality'
  | 'design'
  | 'knowledge'
  | 'personal'
  | 'communication'
  | 'creative'

export interface RoleBotProfile {
  id: string
  name: string
  description: string
  category: RoleBotCategory
  systemPrompt: string
  engine: string
  toolGroups?: readonly ToolExposureGroup[]
  allowedTools?: readonly string[]
  deniedTools?: readonly string[]
  skills?: readonly string[]
  maxIterations?: number
  stages: RoleBotStagePolicy
  memory: RoleBotMemoryRetrieval
  documents: boolean
  thinkingLevel?: 'off' | 'low' | 'medium' | 'high'
  routable: boolean
  source: 'builtin' | 'user'
}

export interface RoleBotSummary {
  id: string
  name: string
  description: string
  category: RoleBotCategory
  engine: string
  stages: RoleBotStagePolicy
  memory: RoleBotMemoryRetrieval
  documents: boolean
  routable: boolean
  source: 'builtin' | 'user'
  skills: readonly string[]
  toolGroups: readonly string[]
}

export function summarizeRoleBot(bot: RoleBotProfile): RoleBotSummary {
  return {
    id: bot.id,
    name: bot.name,
    description: bot.description,
    category: bot.category,
    engine: bot.engine,
    stages: bot.stages,
    memory: bot.memory,
    documents: bot.documents,
    routable: bot.routable,
    source: bot.source,
    skills: bot.skills ?? [],
    toolGroups: bot.toolGroups ?? [],
  }
}
