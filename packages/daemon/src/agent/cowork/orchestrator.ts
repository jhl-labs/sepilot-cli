import { AgentGraph } from '../graph/engine.js'
import type { Deps } from '../graph/nodes.js'
import type { AgentRunContract, ChatRequest, ChatResponse, Message, ToolCall } from '@sepilotd/core'
import type {
  AgentCoworkTaskStatus,
  AgentEvidenceLedger,
  AgentState,
  GraphExecutionContext,
} from '../graph/types.js'
import { agentSubgraphNode } from '../graph/subgraph.js'
import { childIterationBudget } from '../graph/iteration-budget.js'
import { buildCoderGraph } from '../graph/presets/coder.js'
import { buildResearcherGraph } from '../graph/presets/researcher.js'
import { buildReviewerGraph } from '../graph/presets/reviewer.js'
import { childStateFrom, mergeChildInto } from '../graph/subgraph-state.js'
import { capCoworkResults } from './cowork-results-cap.js'
import { getAbortError, isAbortError } from '../../abort.js'
import { logLlmCallTrace } from '../../observability/agent-trace.js'
import { runUserFacingTextCall } from '../graph/streaming.js'
import { enforceRunOutcomeReviewEvidenceFloor } from '../outcome-review.js'
import { runAuxiliaryLlmChat } from '../auxiliary-llm.js'
import { evaluateCompletionGate } from '../graph/completion-gate.js'
import {
  addRenderedUiValidationToContract,
  contractRequiresRenderedUiValidation,
  inputExpressesMutationIntent,
  inputRequiresRenderedUiValidation,
} from '../task-contract.js'

interface CoworkSubtask {
  role: string
  instruction: string
  choices?: string[]
}

const COWORK_DECOMPOSE_MAX_TOKENS = 2048
const DEFAULT_COWORK_MAX_SUBTASKS = 8
const DEFAULT_COWORK_MAX_DISPATCHES = 16

function resolveCoworkMaxSubtasks(): number {
  const raw = Number(process.env.SEPILOTD_COWORK_MAX_SUBTASKS)
  return Number.isFinite(raw) && raw >= 1 ? Math.floor(raw) : DEFAULT_COWORK_MAX_SUBTASKS
}

function resolveCoworkMaxDispatches(): number {
  const raw = Number(process.env.SEPILOTD_COWORK_MAX_DISPATCHES)
  return Number.isFinite(raw) && raw >= 1 ? Math.floor(raw) : DEFAULT_COWORK_MAX_DISPATCHES
}

function resolveCoworkMaxTotalTokens(): number | null {
  const raw = Number(process.env.SEPILOTD_COWORK_MAX_TOTAL_TOKENS)
  return Number.isFinite(raw) && raw >= 1 ? Math.floor(raw) : null
}

// Bound the plan length. A cowork plan's total cost is Σ(role budget × plan
// length); the length is model-controlled (decomposer output + mid-run
// retry/backtrack insertions), so cap it to keep runaway cost in check.
function capCoworkPlan(subtasks: CoworkSubtask[]): CoworkSubtask[] {
  const max = resolveCoworkMaxSubtasks()
  return subtasks.length > max ? subtasks.slice(0, max) : subtasks
}

const DISCUSS_ROLES = new Set(['discuss', 'question', 'feedback', 'user'])
const EXECUTABLE_ROLES = new Set(['coder', 'reviewer', 'researcher'])
const COWORK_RESEARCHER_MAX_TOKENS = 2048
const COWORK_REVIEWER_MAX_TOKENS = 4096
const COWORK_RESEARCHER_MAX_TOOL_CALLS_PER_TURN = 6
const COWORK_REVIEWER_MAX_TOOL_CALLS_PER_TURN = 4
const COWORK_CODER_MAX_CONTINUATION_CYCLES = 2
const COWORK_READONLY_MAX_CONTINUATION_CYCLES = 0
const COWORK_EMPTY_CODER_RETRY_LIMIT = 1

type CoworkExecutableRole = 'coder' | 'reviewer' | 'researcher'

function coworkModelSupportsTextGeneration(
  model: Deps['provider']['models'][number],
): boolean {
  // Embedding-only catalog entries cannot serve any coordinator text node.
  // A provider may expose a hybrid model with embedding + tool/chat support;
  // preserve it when generation capability is explicit.
  return model.capabilities.embedding !== true || model.capabilities.toolUse === true
}

function resolveCoworkTextGenerationModel(
  provider: Deps['provider'],
  preferredModels: readonly (string | undefined)[],
): string {
  for (const preferred of preferredModels) {
    const candidate = preferred?.trim()
    if (!candidate) continue
    const known = provider.models.find((model) => model.id === candidate)
    // An explicitly selected main model can be absent from a stale catalog;
    // honor it unless current metadata proves that it is embedding-only.
    if (!known || coworkModelSupportsTextGeneration(known)) return candidate
  }
  return provider.models.find(coworkModelSupportsTextGeneration)?.id ?? 'default'
}

function cloneSubtasks(subtasks: CoworkSubtask[]): CoworkSubtask[] {
  return subtasks.map((subtask) => ({
    role: subtask.role,
    instruction: subtask.instruction,
    ...(subtask.choices ? { choices: [...subtask.choices] } : {}),
  }))
}

function buildFallbackSubtasks(state: AgentState): CoworkSubtask[] {
  const contract = state.seedContract
  const hasContractedArtifactWork = hasDurableArtifactContract(contract)

  if (!hasContractedArtifactWork) {
    return [{ role: 'coder', instruction: state.input }]
  }

  return [
    {
      role: 'coder',
      instruction:
        'Gather the required evidence and create or update the required artifact(s) incrementally, preserving partial status when scope remains.',
    },
    {
      role: 'reviewer',
      instruction:
        'Review the artifact against required deliverables, artifact sections, and evidence requirements before finalization.',
    },
  ]
}

function hasDurableArtifactContract(contract: AgentRunContract | undefined): boolean {
  return (
    (contract?.requiredArtifacts?.length ?? 0) > 0 ||
    (contract?.artifactSections?.length ?? 0) > 0 ||
    (contract?.evidenceRequirements?.length ?? 0) > 0
  )
}

function normalizeCoworkPlanForContract(
  subtasks: CoworkSubtask[],
  state: AgentState,
  fallbackSubtasks: CoworkSubtask[],
): CoworkSubtask[] {
  if (!hasDurableArtifactContract(state.seedContract)) {
    return capCoworkPlan(subtasks)
  }

  const normalized = subtasks
    .map((subtask) => ({ ...subtask, role: normalizeCoworkRole(subtask.role) }))
    .filter((subtask) => subtask.role !== 'discuss')

  const first = normalized[0]
  if (!first || first.role !== 'coder') {
    return cloneSubtasks(fallbackSubtasks)
  }

  const hasReviewer = normalized.some((subtask) => subtask.role === 'reviewer')
  if (hasReviewer) {
    return capCoworkPlan(normalized)
  }

  const fallbackReviewer = fallbackSubtasks.find((subtask) => normalizeCoworkRole(subtask.role) === 'reviewer')
  // Reserve the last slot for the appended reviewer so review is never dropped
  // by the cap.
  return fallbackReviewer
    ? [...capCoworkPlan(normalized).slice(0, resolveCoworkMaxSubtasks() - 1), { ...fallbackReviewer }]
    : capCoworkPlan(normalized)
}

function parseSubtasks(content: string, fallbackSubtasks: CoworkSubtask[]): CoworkSubtask[] {
  const fallback = cloneSubtasks(fallbackSubtasks)
  try {
    const match = content.match(/\[[\s\S]*\]/)
    if (!match) {
      return fallback
    }

    const parsed = JSON.parse(match[0]) as Array<Record<string, unknown>>
    const subtasks = parsed
      .map((item) => ({
        role:
          String(item.role ?? '')
            .trim()
            .toLowerCase() || 'coder',
        instruction: String(item.instruction ?? '').trim(),
        choices: Array.isArray(item.choices)
          ? item.choices
              .filter((choice): choice is string => typeof choice === 'string')
              .map((choice) => choice.trim())
              .filter(Boolean)
          : undefined,
      }))
      .filter((item) => item.instruction.length > 0)

    return subtasks.length > 0 ? capCoworkPlan(subtasks) : fallback
  } catch {
    return fallback
  }
}

function isDiscussSubtask(role: string): boolean {
  return DISCUSS_ROLES.has(role)
}

function normalizeCoworkRole(role: string): CoworkExecutableRole | 'discuss' {
  const normalized = role.trim().toLowerCase()
  if (isDiscussSubtask(normalized)) return 'discuss'
  if (normalized.includes('review')) return 'reviewer'
  if (normalized.includes('research')) return 'researcher'
  if (EXECUTABLE_ROLES.has(normalized)) {
    return normalized as CoworkExecutableRole
  }
  return 'coder'
}

function currentSubtask(state: AgentState): CoworkSubtask | null {
  const plan = state.coworkPlan ?? []
  return plan[state.planIndex] ?? null
}

function formatCoworkMemory(role: string, detail: string): string {
  return `[${role}] ${detail}`
}

function resultHasIncompleteStem(text: string): boolean {
  return text.split(/\r?\n/).some((line) => line.trim().startsWith('INCOMPLETE:'))
}

function classifyCoworkTaskResult(
  role: CoworkExecutableRole,
  result: string,
  options: { retryScheduled?: boolean } = {},
): AgentCoworkTaskStatus {
  if (options.retryScheduled) {
    return 'retry_scheduled'
  }
  if (role === 'reviewer' && finalUnverifiedLine(result)) {
    return 'unverified'
  }
  return resultHasIncompleteStem(result) ? 'incomplete' : 'complete'
}

function recordCoworkTaskResult(
  state: AgentState,
  task: CoworkSubtask,
  role: CoworkExecutableRole,
  result: string,
  status: AgentCoworkTaskStatus,
): void {
  const prior = state.coworkTaskResults ?? []
  state.coworkTaskResults = capCoworkResults([
    ...prior,
    {
      sequence: prior.length,
      planIndex: state.planIndex,
      role,
      instruction: task.instruction,
      status,
      result,
      recordedAt: new Date().toISOString(),
    },
  ])
}

function hasLaterCompleteReviewerResult(
  results: NonNullable<AgentState['coworkTaskResults']>,
  index: number,
): boolean {
  return results
    .slice(index + 1)
    .some((later) => later.role === 'reviewer' && later.status === 'complete')
}

function unresolvedCoworkTaskBlockers(state: AgentState): string[] {
  const results = state.coworkTaskResults ?? []
  const blockers: string[] = []
  for (const [index, result] of results.entries()) {
    if (result.status === 'complete' || result.status === 'retry_scheduled') {
      continue
    }
    if (result.status === 'unverified' && hasLaterCompleteReviewerResult(results, index)) {
      continue
    }
    blockers.push(`${result.role} "${result.instruction}": ${result.result}`)
  }
  return blockers
}

function summarizeMemories(memories: string[]): string {
  return memories.length > 0
    ? `Synthesizing ${memories.length} cowork updates into one response.`
    : 'Synthesizing the cowork run without specialist notes.'
}

function formatRecentCoworkToolEvidence(state: AgentState): string {
  const recent = (state.recentToolResults ?? []).slice(-10).map((result, index) => {
    const tool = result.toolName ?? 'tool'
    const output = result.output.trim().replace(/\s+/g, ' ')
    const compact = output.length > 1200 ? `${output.slice(0, 1197).trimEnd()}...` : output
    return `${index + 1}. ${tool} ${result.status}: ${compact || '(empty)'}`
  })
  return recent.length > 0
    ? `Recent observed tool evidence:\n${recent.join('\n')}`
    : 'Recent observed tool evidence: none available.'
}

function formatPriorCoworkResults(
  results: NonNullable<AgentState['coworkTaskResults']>,
): string {
  const completed = results.filter((result) => result.status !== 'retry_scheduled')
  return completed.length > 0
    ? [
        'Completed sibling outcomes (context only; do not repeat their tool work):',
        ...completed.map((result) => `[${result.role}] ${result.result}`),
      ].join('\n\n')
    : 'Completed sibling outcomes: none.'
}

function formatCoworkRunContract(state: Pick<AgentState, 'seedContract'>): string {
  const contract = state.seedContract
  if (!contract) return ''
  const lines = [
    'Run contract:',
    `Goal: ${contract.summary}`,
    'Acceptance criteria:',
    ...contract.acceptanceCriteria.map((criterion) => `- ${criterion.id}: ${criterion.text}`),
  ]
  if (contract.constraints.length > 0) {
    lines.push('Constraints:', ...contract.constraints.map((constraint) => `- ${constraint}`))
  }
  if (contract.outOfScope.length > 0) {
    lines.push('Out of scope:', ...contract.outOfScope.map((item) => `- ${item}`))
  }
  if (contract.requiredArtifacts?.length) {
    lines.push(
      'Required artifacts:',
      ...contract.requiredArtifacts.map((artifact) => {
        const description = artifact.description ? ` — ${artifact.description}` : ''
        return `- ${artifact.path} (${artifact.kind})${description}`
      }),
    )
  }
  if (contract.evidenceRequirements?.length) {
    lines.push(
      'Evidence requirements:',
      ...contract.evidenceRequirements.map((requirement) => {
        const details = [
          `${requirement.kind}: ${requirement.description}`,
          typeof requirement.minSourceObservations === 'number'
            ? `minSourceObservations=${requirement.minSourceObservations}`
            : '',
          typeof requirement.minSourceFiles === 'number'
            ? `minSourceFiles=${requirement.minSourceFiles}`
            : '',
          typeof requirement.minSourceScopes === 'number'
            ? `minSourceScopes=${requirement.minSourceScopes}`
            : '',
          requirement.sourceToolNames?.length
            ? `sourceToolNames=${requirement.sourceToolNames.join(',')}`
            : '',
          requirement.requiresArtifactEvidenceMap ? 'requiresArtifactEvidenceMap=true' : '',
          requirement.requiresArtifactSelfReview ? 'requiresArtifactSelfReview=true' : '',
          requirement.requiresSearch ? 'requiresSearch=true' : '',
        ].filter(Boolean)
        return `- ${details.join('; ')}`
      }),
    )
  }
  if (contract.artifactSections?.length) {
    lines.push(
      'Required artifact sections:',
      ...contract.artifactSections.map((section) => {
        const artifactPath = section.artifactPath ? `; artifact=${section.artifactPath}` : ''
        const required = section.required === false ? '; optional' : '; required'
        const description = section.description ? ` — ${section.description}` : ''
        return `- ${section.id}: ${section.title}${artifactPath}${required}${description}`
      }),
    )
  }
  return lines.join('\n')
}

function buildSpecialistInput(
  parent: AgentState,
  role: CoworkExecutableRole,
  instruction: string,
): string {
  const coderOwnsMutation = role === 'coder' && inputExpressesMutationIntent(instruction)
  const roleGuidance = {
    coder: coderOwnsMutation
      ? 'You own the assigned implementation subtask. Read relevant code, make targeted edits, and run meaningful validation through the normal coding workflow. The assigned subtask is your current scope; the original request is context for boundaries and evidence.'
      : 'You own the assigned validation or inspection subtask. Gather the requested current-turn evidence with the matching runtime, terminal, process, or browser tools and report it without inventing a source edit. The assigned subtask is authoritative for this child turn; the original request is context, not an instruction to perform later implementation phases. When running existing package scripts, invoke their declared commands without adding unrequested CLI flags. Tool results are evidence; do not create transcript artifacts unless the assigned subtask names a path.',
    reviewer:
      'You own review. Inspect the work for correctness, regressions, security issues, and missing validation. Be specific. Use only observed tool evidence and paths named by the run contract; never invent filenames, command transcripts, or filesystem results. Describe an absent deliverable by its contract category when no concrete path was observed.',
    researcher:
      'You own one bounded evidence packet, not the whole run. Search/read high-signal material for the assigned scope, then summarize concrete findings, uncertainty, and coverage gaps so the next specialist can continue.',
  } satisfies Record<CoworkExecutableRole, string>

  // Observation-only coder children are deliberately capability-isolated.
  // Repeating the full parent request here makes weaker models execute sibling
  // phases (for example tests/build inside the browser-QA child) even though
  // the assigned user message is correctly scoped.  They only need completed
  // sibling outcomes so they can avoid duplicate work.  Mutation-owning coders
  // and evidence/review specialists still receive the parent boundary.
  const parentBoundary = role === 'coder' && !coderOwnsMutation
    ? formatPriorCoworkResults(parent.coworkTaskResults ?? [])
    : [
        `Original user request:\n${parent.input}`,
        formatCoworkRunContract(parent),
        formatPriorCoworkResults(parent.coworkTaskResults ?? []),
      ]
        .filter(Boolean)
        .join('\n\n')

  return [
    `You are the ${role} specialist inside a cowork team.`,
    roleGuidance[role],
    '',
    parentBoundary,
  ].join('\n')
}

function roleIterationBudget(
  parent: AgentState,
  role: CoworkExecutableRole,
  seedContract: AgentState['seedContract'] = parent.seedContract,
): number {
  const caps = {
    coder: { min: 12, legacyMax: 36, contractMin: 24, parentShare: 0.6, hardCap: 80 },
    reviewer: { min: 6, legacyMax: 14, contractMin: 8, parentShare: 0.25, hardCap: 28 },
    researcher: { min: 3, legacyMax: 4, contractMin: 4, parentShare: 0.2, hardCap: 8 },
  } satisfies Record<
    CoworkExecutableRole,
    {
      min: number
      legacyMax: number
      contractMin: number
      parentShare: number
      hardCap: number
    }
  >
  const cap = caps[role]
  return childIterationBudget({ maxIterations: parent.maxIterations, seedContract }, cap)
}

function cloneRunContract(contract: AgentRunContract): AgentRunContract {
  return {
    ...contract,
    acceptanceCriteria: contract.acceptanceCriteria.map((criterion) => ({
      ...criterion,
    })),
    constraints: [...contract.constraints],
    outOfScope: [...contract.outOfScope],
    requiredArtifacts: contract.requiredArtifacts
      ? contract.requiredArtifacts.map((artifact) => ({ ...artifact }))
      : undefined,
    evidenceRequirements: contract.evidenceRequirements
      ? contract.evidenceRequirements.map((requirement) => ({ ...requirement }))
      : undefined,
    artifactSections: contract.artifactSections
      ? contract.artifactSections.map((section) => ({ ...section }))
      : undefined,
  }
}

function buildRoleScopedRunContract(
  contract: AgentRunContract,
  role: Exclude<CoworkExecutableRole, 'coder'>,
): AgentRunContract {
  const acceptanceCriteria =
    role === 'researcher'
      ? [
          {
            id: 'AC-research-1',
            text: 'Gather a bounded, high-signal evidence packet for the assigned subtask.',
          },
          {
            id: 'AC-research-2',
            text: 'Summarize findings, uncertainty, and remaining evidence gaps without trying to satisfy the whole parent deliverable.',
          },
        ]
      : [
          {
            id: 'AC-review-1',
            text: 'Inspect cowork outputs, observed tool evidence, and available artifacts for correctness against the assigned review scope.',
          },
          {
            id: 'AC-review-2',
            text: 'Report VERIFIED when no blocking issue remains, or UNVERIFIED with concrete blockers and next actions.',
          },
        ]

  return {
    summary: `${role} specialist subtask for: ${contract.summary}`,
    acceptanceCriteria,
    constraints: [...contract.constraints],
    outOfScope: [...contract.outOfScope],
    source: contract.source,
  }
}

function cloneCoworkSeedContractForRole(
  parent: AgentState,
  role: CoworkExecutableRole,
  instruction = parent.input,
): AgentRunContract | undefined {
  const contract = parent.seedContract
  if (!contract) return undefined
  if (role === 'coder') {
    // A cowork plan deliberately decomposes one parent outcome into bounded
    // child tasks.  A test/build, browser-QA, or evidence-only coder child must
    // not inherit the parent's edit requirement: doing so activates the coder
    // edit guard after successful validation and forces a fabricated source
    // change.  Mutation-bearing coder tasks retain full artifact ownership.
    if (inputExpressesMutationIntent(instruction)) {
      return cloneRunContract(contract)
    }

    const capabilities = [
      'filesystem-read' as const,
      'terminal' as const,
      'process' as const,
      ...(inputRequiresRenderedUiValidation(instruction)
        ? ['browser' as const]
        : []),
    ]
    return addRenderedUiValidationToContract({
      summary: `Coder validation/inspection subtask: ${instruction}`,
      acceptanceCriteria: [
        {
          id: 'AC-coder-observe-1',
          text: `Complete the assigned validation or inspection using current-turn tool evidence: ${instruction}`,
        },
        {
          id: 'AC-coder-observe-2',
          text: 'Report exact observed outcomes and any remaining unverified scope without claiming or making an unrequested workspace edit.',
        },
      ],
      constraints: [...contract.constraints],
      outOfScope: [
        ...contract.outOfScope,
        'Workspace mutation not explicitly requested by this assigned subtask.',
      ],
      source: contract.source,
      executionIntent: {
        kind: 'operational-action',
        workspaceMutation: 'forbidden',
        capabilities,
      },
    }, instruction, 6)
  }
  return buildRoleScopedRunContract(contract, role)
}

function capCoworkSpecialistMaxTokens(
  context: GraphExecutionContext | undefined,
  role: CoworkExecutableRole,
): GraphExecutionContext | undefined {
  if (!context) return context
  if (role === 'coder') {
    return {
      ...context,
      graphId: 'coder',
      maxContinuationCycles: Math.min(
        context.maxContinuationCycles ?? COWORK_CODER_MAX_CONTINUATION_CYCLES,
        COWORK_CODER_MAX_CONTINUATION_CYCLES,
      ),
    }
  }
  const cap = role === 'researcher' ? COWORK_RESEARCHER_MAX_TOKENS : COWORK_REVIEWER_MAX_TOKENS
  const toolCallCap =
    role === 'researcher'
      ? COWORK_RESEARCHER_MAX_TOOL_CALLS_PER_TURN
      : COWORK_REVIEWER_MAX_TOOL_CALLS_PER_TURN
  return {
    ...context,
    graphId: role,
    maxTokens: Math.min(context.maxTokens ?? cap, cap),
    maxToolCallsPerTurn: Math.min(context.maxToolCallsPerTurn ?? toolCallCap, toolCallCap),
    maxContinuationCycles: Math.min(
      context.maxContinuationCycles ?? COWORK_READONLY_MAX_CONTINUATION_CYCLES,
      COWORK_READONLY_MAX_CONTINUATION_CYCLES,
    ),
  }
}

function buildRoleChildState(
  parent: AgentState,
  role: CoworkExecutableRole,
  instruction: string,
): AgentState {
  const seedContract = cloneCoworkSeedContractForRole(parent, role, instruction)
  const specialistContext = buildSpecialistInput(parent, role, instruction)
  return childStateFrom(parent, {
    // Keep the assigned task as the child's sole executable user request.
    // Parent context remains available as a system boundary, but classifiers,
    // planners, completion guards, and the model itself must not mistake later
    // parent phases for this child's current authorization.
    input: instruction,
    currentUserContent: instruction,
    messages: [
      { role: 'system', content: specialistContext },
      { role: 'user', content: instruction },
    ],
    memories: [],
    // Each specialist reasons only from its own current-turn tool evidence.
    // Parent evidence is aggregated again in mapRoleChildOut; inheriting it
    // here makes completion guards and prompts confuse sibling work for work
    // performed by this child.
    toolCallHistory: [],
    recentToolResults: [],
    evidenceLedger: undefined,
    validationOutcome: undefined,
    todoList: undefined,
    plan: undefined,
    validationPlan: undefined,
    maxIterations: roleIterationBudget(parent, role, seedContract),
    maxToolCallsPerTurn:
      role === 'researcher'
        ? COWORK_RESEARCHER_MAX_TOOL_CALLS_PER_TURN
        : role === 'reviewer'
          ? COWORK_REVIEWER_MAX_TOOL_CALLS_PER_TURN
          : undefined,
    taskType: role === 'coder' ? 'code' : 'complex',
    seedContract,
  })
}

const EVIDENCE_LEDGER_KEYS = [
  'sourceReads',
  'sourceSearches',
  'artifactWrites',
  'artifactReadBacks',
  'validationRuns',
  'errors',
] as const satisfies readonly (keyof AgentEvidenceLedger)[]

function mergeEvidenceLedgers(
  parent: AgentEvidenceLedger | undefined,
  child: AgentEvidenceLedger | undefined,
): AgentEvidenceLedger | undefined {
  if (!parent && !child) return undefined
  return Object.fromEntries(
    EVIDENCE_LEDGER_KEYS.map((key) => [
      key,
      [...(parent?.[key] ?? []), ...(child?.[key] ?? [])].slice(-200),
    ]),
  ) as unknown as AgentEvidenceLedger
}

function mapRoleChildOut(parent: AgentState, child: AgentState): AgentState {
  const task = currentSubtask(parent)
  const role = task ? normalizeCoworkRole(task.role) : 'discuss'
  const visibleOutput = child.output.trim()
  // A child graph can run without a GraphExecutionContext in tests and
  // embedded callers. In that case its reporter may apply top-level
  // presentation and remove the reviewer verdict label. Recover only the
  // authoritative final debate blocker so the parent can still schedule the
  // required coder/reviewer backtrack.
  const parentDebateCount = parent.debateRounds?.length ?? 0
  const childOwnedDebate = child.debateRounds?.slice(parentDebateCount).at(-1)
  const debateVerdict = role === 'reviewer' && childOwnedDebate
    ? finalReviewerVerdictLine(childOwnedDebate.rationale ?? '')
    : null
  const childMessageVerdict = role === 'reviewer'
    ? [...child.messages]
        .reverse()
        .filter((message) => message.role === 'assistant')
        .map((message) => finalReviewerVerdictLine(textFromMessage(message)))
        .find((verdict): verdict is string => verdict !== null) ?? null
    : null
  const output = debateVerdict ?? childMessageVerdict ?? visibleOutput
  const parentRecentToolResults = [...(parent.recentToolResults ?? [])]
  const parentToolCallHistory = [...(parent.toolCallHistory ?? [])]
  const parentEvidenceLedger = parent.evidenceLedger
  mergeChildInto(parent, child)
  parent.recentToolResults = [
    ...parentRecentToolResults,
    ...(child.recentToolResults ?? []),
  ].slice(-20)
  parent.toolCallHistory = [
    ...parentToolCallHistory,
    ...(child.toolCallHistory ?? []),
  ].slice(-200)
  parent.evidenceLedger = mergeEvidenceLedgers(parentEvidenceLedger, child.evidenceLedger)
  // syncParentStateFromChild intentionally mirrors a child while it is live,
  // which means the transient child memory list replaces the parent's list.
  // Rebuild durable cowork summaries from structured task results before the
  // current result is captured.
  parent.memories = (parent.coworkTaskResults ?? []).map((result) =>
    formatCoworkMemory(result.role, result.result),
  )
  parent.output = output
  return parent
}

function routeNextCoworkNode(state: AgentState): string {
  if (state.coworkBudgetExhausted) return 'synthesize'
  const task = currentSubtask(state)
  if (!task) return 'synthesize'
  const role = normalizeCoworkRole(task.role)
  const latestCoderResult = [...(state.coworkTaskResults ?? [])]
    .reverse()
    .find((result) => result.role === 'coder')
  if (
    role === 'reviewer'
    && latestCoderResult?.status === 'incomplete'
  ) {
    return 'synthesize'
  }
  if (role === 'reviewer') return 'reviewer_subgraph'
  if (role === 'researcher') return 'researcher_subgraph'
  return 'coder_subgraph'
}

function finalReviewerVerdictLine(text: string): string | null {
  const verdictLines = text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const marker = /\b(?:UNVERIFIED|VERIFIED):/i.exec(line)
      return marker?.index === undefined ? null : line.slice(marker.index)
    })
    .filter((line): line is string => line !== null)
  // The reviewer response contract is an explicit verdict envelope. Preserve
  // the last explicit decision even when the model puts supporting evidence
  // after it; trailing explanation must not turn UNVERIFIED into completion.
  return verdictLines.at(-1) ?? null
}

function finalUnverifiedLine(text: string): string | null {
  const verdict = finalReviewerVerdictLine(text)
  return verdict && /^UNVERIFIED:/i.test(verdict) ? verdict : null
}

function textFromMessage(message: Message): string {
  if (typeof message.content === 'string') return message.content
  return message.content
    .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

function normalizedEvidencePath(path: string): string {
  return path.replace(/\\/g, '/').replace(/^\.\//, '').replace(/\/+$/g, '')
}

function evidencePathsMatch(left: string, right: string): boolean {
  const normalizedLeft = normalizedEvidencePath(left)
  const normalizedRight = normalizedEvidencePath(right)
  return (
    normalizedLeft === normalizedRight ||
    normalizedLeft.endsWith(`/${normalizedRight}`) ||
    normalizedRight.endsWith(`/${normalizedLeft}`)
  )
}

function extractApplyPatchPaths(patch: string): string[] {
  const paths = new Set<string>()
  const pattern = /^\*\*\* (?:Add|Update|Delete) File: (.+)$/gm
  let match: RegExpExecArray | null
  while ((match = pattern.exec(patch)) !== null) {
    const path = match[1]?.trim()
    if (path) {
      paths.add(path)
    }
  }
  return Array.from(paths)
}

function toolCallPaths(toolCall: ToolCall): string[] {
  if (toolCall.name === 'apply_patch' && typeof toolCall.arguments.patch === 'string') {
    return extractApplyPatchPaths(toolCall.arguments.patch)
  }
  const path = toolCall.arguments.path
  return typeof path === 'string' && path.trim() ? [path.trim()] : []
}

function successfulWriteResult(toolName: string, content: string): boolean {
  if (toolName === 'fs.write') return /\bWrote\s+\d+\s+bytes\s+to\s+/i.test(content)
  if (toolName === 'fs.append') return /\bAppended\s+\d+\s+bytes\s+to\s+/i.test(content)
  if (toolName === 'fs.edit') return /\bReplaced\s+\d+\s+occurrences?\s+in\s+/i.test(content)
  if (toolName === 'apply_patch') return /\bApplied patch to\s+\d+\s+file\(s\):/i.test(content)
  return false
}

function normalizeArtifactHeading(value: string): string {
  return value
    .normalize('NFKC')
    .toLowerCase()
    .replace(/^#+\s*/, '')
    .replace(/^[\s\d.)-]+/, '')
    .replace(/[`*_~:：\-–—()[\]{}]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function markdownHeadingTitles(content: string): string[] {
  return content
    .split(/\r?\n/)
    .map((line) => line.match(/^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/)?.[1] ?? '')
    .filter(Boolean)
    .map(normalizeArtifactHeading)
    .filter(Boolean)
}

function artifactHasSection(content: string, title: string): boolean {
  const expected = normalizeArtifactHeading(title)
  if (!expected) return false
  return markdownHeadingTitles(content).some(
    (heading) => heading === expected || heading.includes(expected) || expected.includes(heading),
  )
}

function artifactEvidence(
  messages: Message[],
  artifactPath: string,
): { written: boolean; readAfterWrite: boolean; latestReadContent?: string } {
  const callsById = new Map<string, ToolCall>()
  let lastWriteIndex = -1
  let readAfterWrite = false
  let latestReadContent: string | undefined

  messages.forEach((message, index) => {
    for (const call of message.toolCalls ?? []) {
      callsById.set(call.id, call)
    }
    if (message.role !== 'tool' || !message.toolCallId) {
      return
    }
    const call = callsById.get(message.toolCallId)
    if (!call) return
    const paths = toolCallPaths(call)
    if (paths.length === 0 || !paths.some((path) => evidencePathsMatch(path, artifactPath))) {
      return
    }
    const content = textFromMessage(message)
    if (successfulWriteResult(call.name, content)) {
      lastWriteIndex = index
      readAfterWrite = false
      return
    }
    if (call.name === 'fs.read' && index > lastWriteIndex && !/^\[error:/i.test(content.trim())) {
      readAfterWrite = true
      latestReadContent = content
    }
  })

  return { written: lastWriteIndex >= 0, readAfterWrite, latestReadContent }
}

function missingRequiredArtifactWritePaths(state: AgentState): string[] {
  return (state.seedContract?.requiredArtifacts ?? [])
    .map((artifact) => artifact.path)
    .filter((path) => path.trim().length > 0)
    .filter((path) => !artifactEvidence(state.messages, path).written)
}

function buildEmptyCoderRetryTask(input: {
  originalInstruction: string
  missingPaths: string[]
  previousResult: string
  state: AgentState
}): CoworkSubtask {
  const ownsMutation = input.missingPaths.length > 0
    || inputExpressesMutationIntent(input.originalInstruction)
  const missingArtifactLine = input.missingPaths.length > 0
    ? `Missing required artifact write evidence: ${input.missingPaths.join(', ')}.`
    : ''
  const recoveryAction = input.missingPaths.length > 0
    ? 'Use a file-writing/editing tool to create or update the required artifact(s), then read them back before finishing.'
    : ownsMutation
      ? 'Resume the implementation from observed workspace state, make the requested change with the appropriate tools, validate it, and provide a concise final handoff.'
      : 'Continue only the same assigned observation task. Reuse successful current-run evidence, perform only the remaining checks, do not repeat completed commands or observations, and provide a concise final handoff.'
  return {
    role: 'coder',
    instruction: [
      'The previous coder turn produced no final response, so its work cannot be handed to review yet.',
      missingArtifactLine,
      `Original coder instruction: ${input.originalInstruction}`,
      input.previousResult ? `Previous result: ${input.previousResult}` : '',
      ownsMutation ? formatCoworkRunContract(input.state) : '',
      recoveryAction,
      'If a concrete blocker prevents the requested work, answer INCOMPLETE with the blocker instead of silently completing.',
    ]
      .filter(Boolean)
      .join('\n'),
  }
}

function buildRequiredArtifactSynthesis(
  state: AgentState,
  contract: AgentRunContract,
): string | null {
  const requiredArtifacts = contract.requiredArtifacts ?? []
  if (requiredArtifacts.length === 0) return null

  const statuses = requiredArtifacts.map((artifact) => ({
    artifact,
    ...artifactEvidence(state.messages, artifact.path),
  }))
  const missing = statuses.filter((status) => !status.written)
  if (missing.length > 0) {
    return [
      `INCOMPLETE: Required artifact(s) were not written: ${missing.map((status) => status.artifact.path).join(', ')}.`,
      'Next step: write the required artifact(s), read them back, and then finalize.',
    ].join('\n')
  }

  const unread = statuses.filter((status) => !status.readAfterWrite)
  if (unread.length > 0) {
    return [
      `INCOMPLETE: Required artifact(s) were written but not read back after the latest write: ${unread.map((status) => status.artifact.path).join(', ')}.`,
      'Next step: read the artifact content after the latest write and verify it against the run contract.',
    ].join('\n')
  }

  const statusByPath = new Map(
    statuses.map((status) => [normalizedEvidencePath(status.artifact.path), status]),
  )
  const defaultArtifactPath = requiredArtifacts[0]?.path
  const missingSections = (contract.artifactSections ?? [])
    .filter((section) => section.required !== false)
    .map((section) => {
      const artifactPath = section.artifactPath ?? defaultArtifactPath
      if (!artifactPath) {
        return { section, artifactPath: '(unknown artifact)' }
      }
      const normalizedPath = normalizedEvidencePath(artifactPath)
      const status =
        statusByPath.get(normalizedPath) ?? artifactEvidence(state.messages, artifactPath)
      const content = status.latestReadContent
      if (
        !status.written ||
        !status.readAfterWrite ||
        !content ||
        !artifactHasSection(content, section.title)
      ) {
        return { section, artifactPath }
      }
      return null
    })
    .filter(
      (
        entry,
      ): entry is {
        section: NonNullable<AgentRunContract['artifactSections']>[number]
        artifactPath: string
      } => Boolean(entry),
    )
  if (missingSections.length > 0) {
    return [
      `INCOMPLETE: Required artifact section(s) are missing or unverified: ${missingSections.map(({ section, artifactPath }) => `${section.title} (${artifactPath})`).join(', ')}.`,
      'Next step: update the artifact to include the missing required sections, read it back, and verify the section headings before finalizing.',
    ].join('\n')
  }

  const unresolvedTaskBlockers = unresolvedCoworkTaskBlockers(state)
  if (unresolvedTaskBlockers.length > 0) {
    return [
      `INCOMPLETE: ${unresolvedTaskBlockers.length} cowork specialist task(s) remain unresolved.`,
      ...unresolvedTaskBlockers.slice(0, 3).map((blocker) => `- ${blocker}`),
      'Next step: resolve the incomplete specialist scope, update the artifact if needed, and review again before claiming completion.',
    ].join('\n')
  }

  const blocker =
    state.coworkTaskResults && state.coworkTaskResults.length > 0
      ? null
      : state.memories
          .map((memory) => finalUnverifiedLine(memory))
          .find((line): line is string => Boolean(line))
  if (blocker) {
    return [
      `INCOMPLETE: ${blocker.replace(/^UNVERIFIED:\s*/i, '')}`,
      'Next step: resolve the reviewer blocker and update the required artifact before finalizing.',
    ].join('\n')
  }

  return [
    'ANSWER: Required artifact work completed.',
    '',
    'Artifacts:',
    ...statuses.map(
      (status) => `- ${status.artifact.path}: written and read back after the latest write.`,
    ),
    '',
    'The detailed result is in the requested artifact. No additional architecture facts are introduced in this final summary beyond the tool-backed artifact evidence.',
  ].join('\n')
}

function enforceCoworkRenderedUiCompletionGate(
  state: AgentState,
  output: string,
): string {
  if (!state.seedContract || !contractRequiresRenderedUiValidation(state.seedContract)) {
    return output
  }
  if (/^\s*(?:INCOMPLETE|UNVERIFIED):/im.test(output)) {
    return output
  }
  const gate = evaluateCompletionGate(state, output)
  if (gate.criterionVerdictSnapshot) {
    state.completionDiagnostics = {
      ...state.completionDiagnostics,
      criterionVerdictSnapshot: structuredClone(gate.criterionVerdictSnapshot),
    }
  }
  if (gate.decision === 'pass' && !gate.budgetExhausted) {
    return output
  }

  const reason = gate.reason
    ?? (gate.unmet.length > 0
      ? `acceptance criteria not closed as MET: ${gate.unmet.join(', ')}`
      : 'rendered UI completion is not backed by required validation evidence')
  const nextStep = gate.cause === 'ui_audit'
    ? 'Capture fresh desktop and mobile browser screenshot/click/evaluate audits after the latest UI change whose output says `Screenshot image attachment: attached`, include attached active-state desktop/mobile interaction smoke when the UI is interactive, record a completed visual QA todowrite after the latest browser audit with concrete issues found or explicitly none while checking layout/spacing, text wrapping/overflow, contrast/readability, overlap/collision, controls/touch targets, and assets/media completeness, then finalize again.'
    : 'Restate every run contract criterion as CRITERION <id>: MET|UNMET and include the rendered UI validation evidence before finalizing.'

  return [
    `INCOMPLETE: ${reason}`,
    `Next step: ${nextStep}`,
  ].join('\n')
}

function refreshCoworkPlanStrings(state: AgentState): void {
  state.plan = (state.coworkPlan ?? []).map((subtask) => `${subtask.role}: ${subtask.instruction}`)
}

export function buildCoworkGraph(deps: Deps): AgentGraph {
  const graph = new AgentGraph()
  const coderGraph = buildCoderGraph(deps)
  // A cowork reviewer is already the independent critical pass. Nesting a
  // proposer/critic/resolver debate inside it multiplies the same evidence and
  // can recursively retry a negative finding without giving the coder another
  // chance to act. Keep cowork review single-pass; explicit reviewer mode can
  // still opt into debate independently.
  const reviewerGraph = buildReviewerGraph(deps)
  const researcherGraph = buildResearcherGraph(deps)

  const logCoworkCall = async (
    context: GraphExecutionContext | undefined,
    node: string,
    model: string,
    request: ChatRequest,
    response?: ChatResponse,
    error?: unknown,
  ) => {
    const errorMessage =
      error instanceof Error ? error.message : error !== undefined ? String(error) : undefined

    await logLlmCallTrace({
      source: 'graph',
      mode: context?.graphId ?? 'cowork',
      graphId: context?.graphId,
      node,
      sessionId: context?.agentContext.sessionId,
      provider: context?.agentContext.provider ?? deps.provider.id,
      model: context?.agentContext.model ?? model,
      request,
      response,
      error: errorMessage,
    })
  }

  graph.addNode(
    'decompose',
    async function* (state: AgentState, context?: GraphExecutionContext) {
      // Decomposition is an internal aux call (like planning/deliberation):
      // route it through the configured aux model when the provider has it,
      // otherwise fall back to the run's main model. Keeps expensive main
      // models off the pre-dispatch planning round-trip.
      const auxModel = context?.auxModel?.trim()
      const model = resolveCoworkTextGenerationModel(deps.provider, [
        auxModel && deps.provider.models.some((entry) => entry.id === auxModel)
          ? auxModel
          : undefined,
        context?.agentContext.model,
      ])
      const request: ChatRequest = {
        model,
        messages: [
          {
            role: 'system',
            content: [
              'Break the task into 1-5 ordered specialist subtasks.',
              'Use roles coder, reviewer, researcher, and discuss.',
              'Use discuss only when a human decision, missing requirement, or risky permission is needed before proceeding.',
              'For implementation work, prefer a coder subtask followed by reviewer validation; add researcher only when outside evidence or broad investigation is useful.',
              'For durable artifact or broad analysis contracts, make a coder own gathering representative evidence and writing/updating the requested artifact incrementally, followed by review.',
              'Use a separate researcher before the coder only when the evidence packet is independently bounded and the run can still make durable artifact progress promptly.',
              'Keep each researcher subtask bounded to an evidence packet or scope slice; do not ask a researcher to inspect every file, satisfy the whole deliverable, or keep reading until global coverage is complete.',
              'Return compact JSON only as an array of {role, instruction, choices?}.',
              'No markdown fences, no prose, and keep each instruction under 160 characters.',
            ].join(' '),
          },
          {
            role: 'user',
            content: [state.input, formatCoworkRunContract(state)].filter(Boolean).join('\n\n'),
          },
        ],
        temperature: 0,
        maxTokens: COWORK_DECOMPOSE_MAX_TOKENS,
      }
      const fallbackSubtasks = buildFallbackSubtasks(state)
      let subtasks: CoworkSubtask[]
      try {
        const response = await runAuxiliaryLlmChat({
          provider: deps.provider,
          request,
          label: 'Cowork decomposer',
          signal: context?.signal,
          breaker: deps.providerCircuitBreaker,
          budget: context?.auxiliaryLlmBudget,
        })
        await logCoworkCall(context, 'cowork.decompose', model, request, response)
        state.totalUsage.inputTokens += response.usage.inputTokens
        state.totalUsage.outputTokens += response.usage.outputTokens

        const content = typeof response.message.content === 'string' ? response.message.content : ''
        subtasks = parseSubtasks(content, fallbackSubtasks)
      } catch (error) {
        await logCoworkCall(context, 'cowork.decompose', model, request, undefined, error)
        if (isAbortError(error) || (context?.signal?.aborted ?? false)) {
          throw getAbortError(context?.signal, 'Cowork decomposition aborted')
        }
        subtasks = cloneSubtasks(fallbackSubtasks)
      }
      subtasks = normalizeCoworkPlanForContract(subtasks, state, fallbackSubtasks)
      state.coworkPlan = subtasks
      state.planIndex = 0
      state.plan = subtasks.map((subtask) => `${subtask.role}: ${subtask.instruction}`)
      state.memories = []
      state.messages.push({
        role: 'assistant',
        content: `Decomposed into ${subtasks.length} specialist tasks.`,
      })
      yield {
        type: 'cowork_plan',
        plan: subtasks.map(({ role, instruction }) => ({ role, instruction })),
      }

      return state
    },
    { lifecycleState: 'thinking' },
  )

  graph.addNode(
    'dispatch',
    async function* (state: AgentState, context?: GraphExecutionContext) {
      while (true) {
        const task = currentSubtask(state)
        if (!task) return state

        // Top-level cost guard: bound the total number of specialist dispatches
        // (and, when configured, cumulative tokens) so a model-grown plan cannot
        // run the swarm unboundedly. On exhaustion we stop dispatching and let
        // synthesize surface an INCOMPLETE result.
        const maxDispatches = resolveCoworkMaxDispatches()
        const maxTotalTokens = resolveCoworkMaxTotalTokens()
        const usedTokens = state.totalUsage.inputTokens + state.totalUsage.outputTokens
        if (
          (state.coworkDispatchCount ?? 0) >= maxDispatches
          || (maxTotalTokens !== null && usedTokens >= maxTotalTokens)
        ) {
          state.coworkBudgetExhausted = true
          state.coworkBudgetReason = (state.coworkDispatchCount ?? 0) >= maxDispatches
            ? `cowork dispatch budget of ${maxDispatches} specialist runs was reached`
            : `cowork token budget of ${maxTotalTokens} tokens was reached`
          return state
        }

        const { role, instruction, choices } = task
        const normalizedRole = normalizeCoworkRole(role)
        if (normalizedRole === 'discuss') {
          // Non-interactive run (no requestQuestion transport): a discuss step
          // cannot get a human answer, so skip it instead of spending latency on
          // a canned "no interactive response" note.
          if (!context?.requestQuestion) {
            state.planIndex += 1
            continue
          }
          yield {
            type: 'cowork_discuss_request',
            prompt: instruction,
            choices,
          }
          const response = await context.requestQuestion({
            sessionId: context.agentContext.sessionId,
            prompt: instruction,
            choices,
          })
          state.coworkDiscussPrompt = instruction
          state.coworkDiscussChoices = choices
          state.coworkDiscussCount = (state.coworkDiscussCount ?? 0) + 1
          state.memories.push(formatCoworkMemory('user-feedback', response))
          yield {
            type: 'cowork_discuss_response',
            prompt: instruction,
            response,
          }
          state.planIndex += 1
          continue
        }

        yield {
          type: 'cowork_task_start',
          role: normalizedRole,
          instruction,
        }
        return state
      }
    },
    { lifecycleState: 'thinking' },
  )

  graph.addNode(
    'coder_subgraph',
    agentSubgraphNode({
      nodeId: 'cowork_coder_subgraph',
      graph: coderGraph,
      nodePrefix: 'coder',
      forwardMessages: false,
      mapIn: (parent) => {
        const task = currentSubtask(parent)
        return buildRoleChildState(parent, 'coder', task?.instruction ?? parent.input)
      },
      mapContext: (_parent, context) => capCoworkSpecialistMaxTokens(context, 'coder'),
      mapOut: mapRoleChildOut,
    }),
    { lifecycleState: 'thinking' },
  )

  graph.addNode(
    'reviewer_subgraph',
    agentSubgraphNode({
      nodeId: 'cowork_reviewer_subgraph',
      graph: reviewerGraph,
      nodePrefix: 'reviewer',
      forwardMessages: false,
      mapIn: (parent) => {
        const task = currentSubtask(parent)
        return buildRoleChildState(parent, 'reviewer', task?.instruction ?? parent.input)
      },
      mapContext: (_parent, context) => capCoworkSpecialistMaxTokens(context, 'reviewer'),
      mapOut: mapRoleChildOut,
    }),
    { lifecycleState: 'thinking' },
  )

  graph.addNode(
    'researcher_subgraph',
    agentSubgraphNode({
      nodeId: 'cowork_researcher_subgraph',
      graph: researcherGraph,
      nodePrefix: 'researcher',
      forwardMessages: false,
      mapIn: (parent) => {
        const task = currentSubtask(parent)
        return buildRoleChildState(parent, 'researcher', task?.instruction ?? parent.input)
      },
      mapContext: (_parent, context) => capCoworkSpecialistMaxTokens(context, 'researcher'),
      mapOut: mapRoleChildOut,
    }),
    { lifecycleState: 'thinking' },
  )

  graph.addNode(
    'capture_result',
    async function* (state: AgentState) {
      const task = currentSubtask(state)
      if (!task) return state

      const role = normalizeCoworkRole(task.role)
      if (role === 'discuss') {
        state.planIndex += 1
        return state
      }
      // A specialist subgraph just ran — count it against the dispatch budget.
      state.coworkDispatchCount = (state.coworkDispatchCount ?? 0) + 1
      const rawResult = state.output.trim()
      let result = rawResult || `${role} completed without a final response.`
      let retryScheduled = false
      const missingRequiredArtifactWrites =
        role === 'coder' ? missingRequiredArtifactWritePaths(state) : []
      if (
        role === 'coder' &&
        !rawResult &&
        (state.coworkArtifactRetryCount ?? 0) < COWORK_EMPTY_CODER_RETRY_LIMIT
      ) {
        retryScheduled = true
        state.coworkArtifactRetryCount = (state.coworkArtifactRetryCount ?? 0) + 1
        const retryTask = buildEmptyCoderRetryTask({
          originalInstruction: task.instruction,
          missingPaths: missingRequiredArtifactWrites,
          previousResult: result,
          state,
        })
        const plan = state.coworkPlan ?? []
        state.coworkPlan = capCoworkPlan([
          ...plan.slice(0, state.planIndex + 1),
          retryTask,
          ...plan.slice(state.planIndex + 1),
        ])
        refreshCoworkPlanStrings(state)
        result = [
          `INCOMPLETE: ${result}`,
          missingRequiredArtifactWrites.length > 0
            ? `Required artifact write evidence is still missing: ${missingRequiredArtifactWrites.join(', ')}.`
            : 'The coder did not provide a reviewable handoff.',
          'A retry coder subtask was scheduled before review.',
        ].join(' ')
      } else if (role === 'coder' && !rawResult) {
        result = [
          'INCOMPLETE: coder produced no final response after the bounded recovery attempt.',
          'No reviewer was run because there is no reviewable implementation handoff.',
        ].join(' ')
      }
      const status = classifyCoworkTaskResult(role, result, { retryScheduled })
      recordCoworkTaskResult(state, task, role, result, status)
      state.memories.push(formatCoworkMemory(role, result))
      yield {
        type: 'cowork_task_complete',
        role,
        instruction: task.instruction,
        result,
      }
      const reviewerBlocker = role === 'reviewer' ? finalUnverifiedLine(result) : null
      if (reviewerBlocker && (state.backtrackCount ?? 0) < 1) {
        state.backtrackCount = (state.backtrackCount ?? 0) + 1
        state.backtrackReason = reviewerBlocker
        const followups: CoworkSubtask[] = [
          {
            role: 'coder',
            instruction: [
              'Address the reviewer blocker with the smallest safe change.',
              `Reviewer blocker: ${reviewerBlocker}`,
              `Original reviewer instruction: ${task.instruction}`,
              formatCoworkRunContract(state),
            ].join('\n'),
          },
          {
            role: 'reviewer',
            instruction: [
              'Re-review the coder follow-up and verify whether the blocker is resolved.',
              `Original blocker: ${reviewerBlocker}`,
              'End with VERIFIED or UNVERIFIED.',
            ].join('\n'),
          },
        ]
        const plan = state.coworkPlan ?? []
        state.coworkPlan = capCoworkPlan([
          ...plan.slice(0, state.planIndex + 1),
          ...followups,
          ...plan.slice(state.planIndex + 1),
        ])
        refreshCoworkPlanStrings(state)
      }
      state.output = ''
      state.toolCalls = []
      state.toolResults = []
      state.planIndex += 1
      return state
    },
    { lifecycleState: 'thinking' },
  )

  graph.addNode(
    'synthesize',
    async function* (state: AgentState, context?: GraphExecutionContext) {
      const summary = summarizeMemories(state.memories)
      yield {
        type: 'cowork_synthesizing',
        summary,
      }

      const results = state.memories
        .map((memory) => {
          const match = memory.match(/^\[(.+?)\]\s([\s\S]*)$/)
          if (!match) {
            return memory
          }

          return `### ${match[1]}\n${match[2]}`
        })
        .join('\n\n')

      const remainingSubtasks = Math.max(0, (state.coworkPlan?.length ?? 0) - state.planIndex)
      if (state.coworkBudgetExhausted) {
        state.output = [
          `INCOMPLETE: cowork stopped early — ${state.coworkBudgetReason ?? 'the run budget was reached'}.`,
          remainingSubtasks > 0
            ? `- ${remainingSubtasks} planned subtask(s) were not run.`
            : '',
          results ? `\n${results}` : '',
          'Next step: narrow the plan or raise SEPILOTD_COWORK_MAX_DISPATCHES / SEPILOTD_COWORK_MAX_TOTAL_TOKENS, then continue the unfinished work.',
        ].filter(Boolean).join('\n')
        return state
      }

      const unresolvedTaskBlockers = unresolvedCoworkTaskBlockers(state)
      if (unresolvedTaskBlockers.length > 0) {
        state.output = [
          `INCOMPLETE: ${unresolvedTaskBlockers.length} cowork specialist task(s) remain unresolved.`,
          ...unresolvedTaskBlockers.slice(0, 3).map((blocker) => `- ${blocker}`),
          'Next step: resolve the incomplete specialist scope, update the artifact if needed, and review again before claiming completion.',
        ].join('\n')
        return state
      }

      const artifactSynthesis = state.seedContract
        ? buildRequiredArtifactSynthesis(state, state.seedContract)
        : null
      if (artifactSynthesis) {
        state.output = artifactSynthesis
        const synthesisReview = enforceRunOutcomeReviewEvidenceFloor({
          review: { status: 'complete', reason: 'Cowork artifact synthesis completed.' },
          messages: state.messages,
          runContract: state.seedContract,
          evidenceLedger: state.evidenceLedger,
          assistantAnswer: state.output,
          pathClaimEvidenceScope: 'all',
        })
        if (synthesisReview?.status === 'needs_recovery') {
          state.output = [
            `INCOMPLETE: ${synthesisReview.reason}`,
            synthesisReview.instruction ? `Next step: ${synthesisReview.instruction}` : '',
          ]
            .filter(Boolean)
            .join('\n')
        }
        state.output = enforceCoworkRenderedUiCompletionGate(state, state.output)
        return state
      }

      const model = resolveCoworkTextGenerationModel(
        deps.provider,
        [context?.agentContext.model],
      )
      const request: ChatRequest = {
        model,
        messages: [
          {
            role: 'system',
            content: [
              'Synthesize the specialist outputs into one cohesive response.',
              'Do not introduce new architecture facts, file paths, module names, validation claims, or completion claims that are not present in the specialist outputs or recent observed tool evidence.',
              'If specialist outputs contain unsupported concrete paths or contradict observed tool evidence, say INCOMPLETE with the missing verification instead of repeating the unsupported claims.',
              'For requested artifact work, keep the final answer focused on the artifact path, verified evidence, and remaining risk; do not rewrite the artifact body from memory.',
            ].join(' '),
          },
          {
            role: 'user',
            content: [
              `Original task: ${state.input}`,
              formatCoworkRunContract(state),
              formatRecentCoworkToolEvidence(state),
              `Specialist results:\n\n${results || state.memories.join('\n\n')}`,
            ]
              .filter(Boolean)
              .join('\n\n'),
          },
        ],
        // Final synthesized answer is user-facing — honor overrides.
        temperature: context?.temperature,
        maxTokens: context?.maxTokens ?? 8192,
      }
      let response: ChatResponse
      try {
        response = yield* runUserFacingTextCall({
          provider: deps.provider,
          request,
          signal: context?.signal,
          breaker: deps.providerCircuitBreaker,
          live: context?.textDeltaMode === 'live',
        })
        await logCoworkCall(context, 'cowork.synthesize', model, request, response)
      } catch (error) {
        await logCoworkCall(context, 'cowork.synthesize', model, request, undefined, error)
        if (isAbortError(error) || context?.signal?.aborted) {
          throw getAbortError(context?.signal, 'Cowork synthesis aborted')
        }
        // Specialist results are already bounded, persisted run evidence. A
        // late text-generation transport failure must not erase them and end
        // the CLI without a reply. Preserve the evidence packet verbatim and
        // let the ordinary completion gate reject unsupported success claims.
        state.output = results
          ? `Cowork specialist results (final synthesis unavailable):\n\n${results}`
          : 'INCOMPLETE: cowork specialists produced no result and final synthesis was unavailable.'
        state.output = enforceCoworkRenderedUiCompletionGate(state, state.output)
        return state
      }

      state.totalUsage.inputTokens += response.usage.inputTokens
      state.totalUsage.outputTokens += response.usage.outputTokens
      state.output = typeof response.message.content === 'string' ? response.message.content : ''
      const synthesisReview = enforceRunOutcomeReviewEvidenceFloor({
        review: { status: 'complete', reason: 'Cowork synthesis completed.' },
        messages: state.messages,
        runContract: state.seedContract,
        evidenceLedger: state.evidenceLedger,
        assistantAnswer: state.output,
        pathClaimEvidenceScope: 'all',
      })
      if (synthesisReview?.status === 'needs_recovery') {
        state.output = [
          `INCOMPLETE: ${synthesisReview.reason}`,
          synthesisReview.instruction ? `Next step: ${synthesisReview.instruction}` : '',
        ]
          .filter(Boolean)
          .join('\n')
      }
      state.output = enforceCoworkRenderedUiCompletionGate(state, state.output)

      return state
    },
    { lifecycleState: 'thinking' },
  )

  graph.addNode(
    'reporter',
    async (state: AgentState) => {
      if (!state.output) {
        state.output = '[Cowork completed without a synthesized response]'
      }
      state.shouldStop = true
      return state
    },
    { lifecycleState: 'done' },
  )

  graph.setStart('decompose')
  graph.addEdge('decompose', 'dispatch')
  graph.addConditionalEdge('dispatch', routeNextCoworkNode, [
    'coder_subgraph',
    'reviewer_subgraph',
    'researcher_subgraph',
    'synthesize',
  ])
  graph.addEdge('coder_subgraph', 'capture_result')
  graph.addEdge('reviewer_subgraph', 'capture_result')
  graph.addEdge('researcher_subgraph', 'capture_result')
  graph.addEdge('capture_result', 'dispatch')
  graph.addEdge('synthesize', 'reporter')
  graph.addEdge('reporter', '__end__')

  return graph
}

export const __testables = {
  buildRoleChildState,
  buildEmptyCoderRetryTask,
  capCoworkPlan,
  capCoworkSpecialistMaxTokens,
  cloneCoworkSeedContractForRole,
  enforceCoworkRenderedUiCompletionGate,
  missingRequiredArtifactWritePaths,
  mapRoleChildOut,
  normalizeCoworkPlanForContract,
  parseSubtasks,
  resolveCoworkTextGenerationModel,
  routeNextCoworkNode,
  unresolvedCoworkTaskBlockers,
}
