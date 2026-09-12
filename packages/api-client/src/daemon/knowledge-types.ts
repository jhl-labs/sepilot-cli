/** Personal knowledge is owned by the daemon profile, independent of Memory TTL. */
export type PersonalKnowledgeKind =
  | 'fact'
  | 'concept'
  | 'decision'
  | 'procedure'
  | 'preference'
  | 'category'
export type PersonalKnowledgeStatus = 'candidate' | 'accepted' | 'archived'
export interface PersonalKnowledgeSource {
  kind: 'conversation' | 'wiki' | 'memory' | 'manual'
  id: string
  title: string
  excerpt: string
  capturedAt: number
}
export interface PersonalKnowledgeRelation {
  targetId: string
  type: 'related' | 'supports' | 'contradicts' | 'supersedes'
}
export interface PersonalKnowledge {
  id: string
  title: string
  body: string
  kind: PersonalKnowledgeKind
  status: PersonalKnowledgeStatus
  parentId: string | null
  tags: string[]
  relations: PersonalKnowledgeRelation[]
  sources: PersonalKnowledgeSource[]
  revision: number
  createdAt: number
  updatedAt: number
  reason: string
}
export interface PersonalKnowledgeWrite {
  id?: string
  expectedRevision?: number
  title: string
  body: string
  kind: PersonalKnowledgeKind
  status: PersonalKnowledgeStatus
  parentId: string | null
  tags: string[]
  relations: PersonalKnowledgeRelation[]
  reason: string
}
export type PersonalKnowledgeCapture =
  | { kind: 'conversation'; id: string }
  | { kind: 'wiki'; id: string }
  | { kind: 'memory' | 'manual'; id: string; title: string; excerpt: string }
export interface PersonalKnowledgeSuggestion {
  id: string
  expectedRevision: number
  parentId: string | null
  tags: string[]
  relations: PersonalKnowledgeRelation[]
  reason: string
}

export interface PersonalKnowledgeReview {
  autoOrganize: boolean
  status: 'idle' | 'running' | 'ready' | 'failed'
  suggestions: PersonalKnowledgeSuggestion[]
  error: string | null
  updatedAt: number
}

export interface KnowledgeActivity {
  id: string
  sequence: number
  kind: string
  trigger: 'manual' | 'automatic'
  parentId: string | null
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'interrupted'
  createdAt: number
  startedAt: number | null
  finishedAt: number | null
  summary: string
  targets: Array<{ id: string; title?: string; revision?: number }>
  resultIds: string[]
  resultPreview: string | null
  error: string | null
  events: Array<{ at: number; message: string }>
}
export interface KnowledgeLlmCall {
  id: string
  activityId: string
  provider: string
  model: string
  status: 'running' | 'succeeded' | 'failed' | 'interrupted'
  startedAt: number
  finishedAt: number | null
  inputTokens: number | null
  outputTokens: number | null
  thinkingTokens: number | null
  cacheReadTokens: number | null
  cacheCreationTokens: number | null
  requestPreview: string
  requestSha256: string
  requestChars: number
  responsePreview: string | null
  error: string | null
}
export interface KnowledgeActivityDetail {
  activity: KnowledgeActivity
  calls: KnowledgeLlmCall[]
}
export interface KnowledgeActivityPage {
  items: KnowledgeActivity[]
  nextCursor: number | null
  totals: {
    operations: number
    running: number
    queued: number
    calls: number
    inputTokens: number
    outputTokens: number
    unknownUsageCalls: number
    todayInputTokens: number
    todayOutputTokens: number
    byModel: Array<{
      provider: string
      model: string
      calls: number
      inputTokens: number
      outputTokens: number
      unknownUsageCalls: number
    }>
  }
}

export type KnowledgeChatUpdateMode = 'off' | 'suggest' | 'automatic'
export interface KnowledgeContentProposal {
  id: string
  status: 'pending' | 'accepted' | 'rejected'
  before: PersonalKnowledge | null
  proposed: { id?: string; expectedRevision?: number; title: string; body: string; reason: string }
  source: PersonalKnowledgeSource
  createdAt: number
  resolvedAt: number | null
  resultId: string | null
  reviewReason?: string
}
export interface KnowledgeMaintenanceState {
  recordsVersion: number
  mode: KnowledgeChatUpdateMode
  proposals: KnowledgeContentProposal[]
}

export interface KnowledgeBudgetState {
  day: string
  limit: number
  chargedTokens: number
  measuredTokens: number
  reservedTokens: number
  remainingTokens: number
}
export interface KnowledgeVerification {
  id: string
  title: string
  revision: number
  verifiedRevision: number | null
  verifiedAt: number | null
  nextReviewAt: number | null
  note: string
  state: 'inactive' | 'changed' | 'due' | 'verified' | 'unverified'
}
export interface KnowledgeLifecycleState {
  verifications: KnowledgeVerification[]
  budget: KnowledgeBudgetState
}
export interface KnowledgeVerificationInput {
  id: string
  expectedRevision: number
  action: 'schedule' | 'verify'
  nextReviewAt: number | null
  note: string
}
