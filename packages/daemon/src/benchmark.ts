import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { performance } from 'node:perf_hooks'
import { fileURLToPath, pathToFileURL } from 'node:url'
import {
  AutonomyLevel,
  type AgentContext,
  type AgentEvent,
  type ChatRequest,
  type ChatResponse,
  type ILLMProvider,
  type ModelInfo,
  type StreamChunk,
  type ToolCall,
} from '@sepilotd/core'
import { AgentModeRouter } from './agent/mode-router.js'
import { extractTaggedPayload } from './agent/prompt-react.js'
import { createGraphApprovalCheckpoint } from './agent/graph/checkpoints.js'
import type { AgentState, GraphExecutionContext } from './agent/graph/types.js'
import { registerBuiltinGraphs } from './agent/graph/presets/index.js'
import { GraphAgentRegistry } from './agent/graph/registry.js'
import { createApp } from './server/app.js'
import type { RuntimeServices } from './server/runtime/types.js'
import { JsonlSessionStore } from './memory/session-store.js'
import { SqliteSemanticIndex, type MemoryEmbedder } from './memory/semantic-index.js'
import { ProviderRegistry } from './providers/registry.js'
import { ApprovalRegistry } from './server/runtime/approvals.js'
import { RunLimiter } from './server/runtime/run-limiter.js'
import { PolicyEngine, createDefaultPolicy } from './security/policy-engine.js'
import { FileSkillRegistry } from './skills/registry.js'
import { HookRegistry } from './hook/registry.js'
import { ToolRegistry } from './tools/registry.js'

interface StreamStep {
  delayMs?: number
  chunk: StreamChunk
}

const SEMANTIC_SEARCH_BENCHMARK_SAMPLES = 3

interface RunMetrics {
  firstTokenMs: number
  wallMs: number
  textLength: number
  textDeltaCount: number
  toolResultCount: number
  approvalRequestCount: number
  errorCount: number
  doneCount: number
  success: boolean
}

export interface ReactSimpleQaBenchmark {
  firstTokenMs: number
  wallMs: number
  textLength: number
  success: boolean
}

export interface GraphCoderStreamBenchmark {
  firstTokenMs: number
  wallMs: number
  textLength: number
  success: boolean
}

export interface ReactPromptFallbackBenchmark {
  firstTokenMs: number
  wallMs: number
  textLength: number
  toolResultCount: number
  providerCallCount: number
  promptRequestCount: number
  success: boolean
}

export interface GraphToolApprovalResumeBenchmark {
  postResumeFirstTokenMs: number
  wallMs: number
  textLength: number
  toolResultCount: number
  resumeSuccessRate: number
}

export interface SemanticSearchBenchmark {
  coldMs: number
  warmMs: number
  coldResultCount: number
  warmResultCount: number
  status: string
}

export interface ConcurrentSaturationBenchmark {
  wallMs: number
  acceptedCount: number
  rejectedCount: number
  rejectedStatusCodes: number[]
}

export interface BenchmarkResults {
  version: number
  generatedAt: string
  nodeVersion: string
  idleMemoryRssBytes: number
  scenarios: {
    react_simple_qa: ReactSimpleQaBenchmark
    react_prompt_fallback: ReactPromptFallbackBenchmark
    graph_coder_stream: GraphCoderStreamBenchmark
    graph_tool_approval_resume: GraphToolApprovalResumeBenchmark
    semantic_search: SemanticSearchBenchmark
    concurrent_4_run_saturation: ConcurrentSaturationBenchmark
  }
  check?: {
    ok: boolean
    baselinePath: string
    failures: BenchmarkCheckFailure[]
  }
}

export interface BenchmarkCheckFailure {
  metric: string
  baseline: number
  actual: number
  allowed?: number
  message: string
}

type MetricRule =
  | {
      path: string
      kind: 'latency'
      percentTolerance: number
      absoluteToleranceMs: number
    }
  | {
      path: string
      kind: 'memory'
      percentTolerance: number
      absoluteToleranceBytes: number
    }
  | {
      path: string
      kind: 'minimum'
    }
  | {
      path: string
      kind: 'exact'
    }

const METRIC_RULES: MetricRule[] = [
  {
    path: 'scenarios.react_simple_qa.firstTokenMs',
    kind: 'latency',
    percentTolerance: 0.75,
    absoluteToleranceMs: 20,
  },
  {
    path: 'scenarios.react_simple_qa.wallMs',
    kind: 'latency',
    percentTolerance: 0.75,
    absoluteToleranceMs: 25,
  },
  {
    path: 'scenarios.react_prompt_fallback.firstTokenMs',
    kind: 'latency',
    percentTolerance: 0.75,
    absoluteToleranceMs: 25,
  },
  {
    path: 'scenarios.react_prompt_fallback.wallMs',
    kind: 'latency',
    percentTolerance: 0.75,
    absoluteToleranceMs: 30,
  },
  {
    path: 'scenarios.react_prompt_fallback.toolResultCount',
    kind: 'exact',
  },
  {
    path: 'scenarios.react_prompt_fallback.providerCallCount',
    kind: 'exact',
  },
  {
    path: 'scenarios.react_prompt_fallback.promptRequestCount',
    kind: 'exact',
  },
  {
    path: 'scenarios.graph_coder_stream.firstTokenMs',
    kind: 'latency',
    percentTolerance: 0.75,
    absoluteToleranceMs: 20,
  },
  {
    path: 'scenarios.graph_coder_stream.wallMs',
    kind: 'latency',
    percentTolerance: 0.75,
    absoluteToleranceMs: 25,
  },
  {
    path: 'scenarios.graph_tool_approval_resume.postResumeFirstTokenMs',
    kind: 'latency',
    percentTolerance: 0.75,
    absoluteToleranceMs: 20,
  },
  {
    path: 'scenarios.graph_tool_approval_resume.wallMs',
    kind: 'latency',
    percentTolerance: 0.75,
    absoluteToleranceMs: 30,
  },
  {
    path: 'scenarios.graph_tool_approval_resume.resumeSuccessRate',
    kind: 'minimum',
  },
  {
    path: 'scenarios.semantic_search.coldMs',
    kind: 'latency',
    percentTolerance: 1.0,
    absoluteToleranceMs: 35,
  },
  {
    path: 'scenarios.semantic_search.warmMs',
    kind: 'latency',
    percentTolerance: 1.0,
    absoluteToleranceMs: 20,
  },
  {
    path: 'scenarios.semantic_search.coldResultCount',
    kind: 'minimum',
  },
  {
    path: 'scenarios.semantic_search.warmResultCount',
    kind: 'minimum',
  },
  {
    path: 'scenarios.concurrent_4_run_saturation.wallMs',
    kind: 'latency',
    percentTolerance: 1.0,
    absoluteToleranceMs: 40,
  },
  {
    path: 'scenarios.concurrent_4_run_saturation.acceptedCount',
    kind: 'exact',
  },
  {
    path: 'scenarios.concurrent_4_run_saturation.rejectedCount',
    kind: 'exact',
  },
  {
    path: 'idleMemoryRssBytes',
    kind: 'memory',
    percentTolerance: 0.5,
    absoluteToleranceBytes: 64 * 1024 * 1024,
  },
]

const benchmarkDirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(benchmarkDirname, '../../..')
export const defaultBaselinePath = join(repoRoot, 'docs', 'benchmarks', 'v0.3-baseline.json')
export const defaultOutputPath = join(repoRoot, 'out', 'benchmarks', 'latest.json')

class ScriptedProvider implements ILLMProvider {
  readonly id: string
  readonly name: string
  readonly models: ModelInfo[]

  private callIndex = 0

  constructor(
    id: string,
    private readonly scriptFactory: (request: ChatRequest, callIndex: number) => StreamStep[],
    capabilities?: Partial<ModelInfo['capabilities']>,
  ) {
    this.id = id
    this.name = `${id}-provider`
    this.models = [
      {
        id: `${id}-model`,
        name: `${id}-model`,
        contextWindow: 8_000,
        maxOutputTokens: 4_000,
        capabilities: {
          vision: false,
          toolUse: true,
          streaming: true,
          embedding: false,
          thinking: false,
          ...capabilities,
        },
      },
    ]
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    const steps = this.nextScript(request)
    return consumeStepsAsChatResponse(steps)
  }

  async *stream(request: ChatRequest): AsyncIterable<StreamChunk> {
    const steps = this.nextScript(request)
    for (const step of steps) {
      if (step.delayMs) {
        await sleep(step.delayMs)
      }
      yield step.chunk
    }
  }

  private nextScript(request: ChatRequest): StreamStep[] {
    const steps = this.scriptFactory(request, this.callIndex)
    this.callIndex += 1
    return steps.map((step) => ({ ...step }))
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function createTextScript(
  text: string,
  options: {
    firstTokenDelayMs: number
    trailingDelayMs: number
    usage?: { inputTokens: number; outputTokens: number }
  },
): StreamStep[] {
  return [
    {
      delayMs: options.firstTokenDelayMs,
      chunk: { type: 'text', text },
    },
    {
      delayMs: options.trailingDelayMs,
      chunk: {
        type: 'usage',
        usage: options.usage ?? { inputTokens: 12, outputTokens: 18 },
      },
    },
    {
      chunk: { type: 'done', finishReason: 'stop' },
    },
  ]
}

function consumeStepsAsChatResponse(steps: StreamStep[]): ChatResponse {
  let text = ''
  const toolCalls: ToolCall[] = []
  const toolCallArgs = new Map<string, string>()
  let inputTokens = 0
  let outputTokens = 0
  let finishReason: ChatResponse['finishReason'] = 'stop'

  for (const step of steps) {
    const chunk = step.chunk
    switch (chunk.type) {
      case 'text':
        text += chunk.text
        break
      case 'tool_call_start':
        if (chunk.toolCall.id && chunk.toolCall.name) {
          toolCalls.push({
            id: chunk.toolCall.id,
            name: chunk.toolCall.name,
            arguments: {},
          })
          toolCallArgs.set(chunk.toolCall.id, '')
        }
        break
      case 'tool_call_delta':
        if (chunk.toolCallId) {
          const previous = toolCallArgs.get(chunk.toolCallId) ?? ''
          toolCallArgs.set(chunk.toolCallId, previous + chunk.delta)
        }
        break
      case 'usage':
        inputTokens += chunk.usage.inputTokens
        outputTokens += chunk.usage.outputTokens
        break
      case 'done':
        finishReason = chunk.finishReason
        break
      case 'thinking':
      case 'error':
        break
    }
  }

  for (const toolCall of toolCalls) {
    const raw = toolCallArgs.get(toolCall.id)
    if (!raw) {
      continue
    }
    try {
      toolCall.arguments = JSON.parse(raw) as Record<string, unknown>
    } catch {
      toolCall.arguments = {}
    }
  }

  return {
    message: {
      role: 'assistant',
      content: text,
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    },
    usage: { inputTokens, outputTokens },
    finishReason,
  }
}

async function collectRunMetrics(
  events: AsyncIterable<AgentEvent>,
): Promise<RunMetrics> {
  const startedAt = performance.now()
  let firstTokenMs: number | null = null
  let textLength = 0
  let textDeltaCount = 0
  let toolResultCount = 0
  let approvalRequestCount = 0
  let errorCount = 0
  let doneCount = 0

  for await (const event of events) {
    switch (event.type) {
      case 'text_delta':
        if (firstTokenMs === null) {
          firstTokenMs = performance.now() - startedAt
        }
        textLength += event.text.length
        textDeltaCount += 1
        break
      case 'tool_result':
        toolResultCount += 1
        break
      case 'approval_request':
        approvalRequestCount += 1
        break
      case 'error':
        errorCount += 1
        break
      case 'done':
        doneCount += 1
        break
      default:
        break
    }
  }

  const wallMs = performance.now() - startedAt

  return {
    firstTokenMs: roundMs(firstTokenMs ?? wallMs),
    wallMs: roundMs(wallMs),
    textLength,
    textDeltaCount,
    toolResultCount,
    approvalRequestCount,
    errorCount,
    doneCount,
    success: errorCount === 0 && doneCount > 0,
  }
}

function roundMs(value: number): number {
  return Number(value.toFixed(1))
}

function medianSample<T>(items: T[], valueOf: (item: T) => number): T {
  const sorted = [...items].sort((left, right) => valueOf(left) - valueOf(right))
  return sorted[Math.floor(sorted.length / 2)]!
}

interface RouterFixtureOptions {
  toolRegistry?: ToolRegistry
  policy?: PolicyEngine
  autonomy?: AutonomyLevel
  textDeltaMode?: 'buffered' | 'live'
}

function createRouterFixture(
  provider: ILLMProvider,
  options: RouterFixtureOptions = {},
): AgentModeRouter {
  const graphRegistry = new GraphAgentRegistry()
  registerBuiltinGraphs(graphRegistry)

  return new AgentModeRouter({
    provider,
    tools: options.toolRegistry ?? new ToolRegistry(),
    policy: options.policy ?? new PolicyEngine(createDefaultPolicy()),
    autonomy: options.autonomy ?? AutonomyLevel.Supervised,
    textDeltaMode: options.textDeltaMode,
    graphRegistry,
  })
}

function benchmarkContext(
  provider: ILLMProvider,
  sessionId: string,
): AgentContext {
  return {
    sessionId,
    provider: provider.id,
    model: provider.models[0]?.id ?? 'benchmark-model',
  }
}

async function runReactSimpleQaBenchmark(): Promise<ReactSimpleQaBenchmark> {
  const provider = new ScriptedProvider('react-bench', () =>
    createTextScript('React benchmark response complete.', {
      firstTokenDelayMs: 18,
      trailingDelayMs: 10,
    }),
  )
  const router = createRouterFixture(provider)
  const metrics = await collectRunMetrics(
    router.run(
      'Summarize the daemon status in one sentence.',
      benchmarkContext(provider, 'bench-react'),
      'react',
    ),
  )

  return {
    firstTokenMs: metrics.firstTokenMs,
    wallMs: metrics.wallMs,
    textLength: metrics.textLength,
    success: metrics.success,
  }
}

function extractToolMessageText(request: ChatRequest): string {
  // Prompt-ReAct rewrites native `role: 'tool'` messages into user-visible
  // `<tool_result>` blocks before invoking providers. Accept both transport
  // shapes so this benchmark measures the loop instead of replaying one tool
  // call until the iteration guard stops it.
  const toolMessage = [...request.messages]
    .reverse()
    .find((message) =>
      message.role === 'tool'
      || (
        message.role === 'user'
        && typeof message.content === 'string'
        && message.content.includes('<tool_result')
      ),
    )

  if (!toolMessage) {
    return ''
  }

  const text = typeof toolMessage.content === 'string'
    ? toolMessage.content
    : toolMessage.content
        .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
        .map((part) => part.text)
        .join('')

  return (extractTaggedPayload(text, ['tool_result']) ?? text).trim()
}

async function runReactPromptFallbackBenchmark(): Promise<ReactPromptFallbackBenchmark> {
  const capturedRequests: ChatRequest[] = []
  const tools = new ToolRegistry()
  tools.register({
    name: 'bench.echo',
    description: 'Echo benchmark tool',
    inputSchema: {
      type: 'object',
      properties: {
        text: { type: 'string' },
      },
      required: ['text'],
    },
    async execute(input) {
      await sleep(4)
      return {
        output: `bench:${String(input.text ?? '')}`,
        status: 'success' as const,
        durationMs: 4,
      }
    },
  })

  const provider = new ScriptedProvider(
    'prompt-fallback-bench',
    (request) => {
      capturedRequests.push(request)
      const isOutcomeReview = request.messages.some((message) =>
        message.role === 'system'
        && typeof message.content === 'string'
        && message.content.includes('You are an outcome judge for an autonomous agent run.'),
      )
      if (isOutcomeReview) {
        return createTextScript(
          JSON.stringify({
            status: 'complete',
            reason: 'The benchmark prompt fallback answer is complete.',
          }),
          {
            firstTokenDelayMs: 4,
            trailingDelayMs: 2,
            usage: { inputTokens: 3, outputTokens: 2 },
          },
        )
      }

      const toolOutput = extractToolMessageText(request)

      if (!toolOutput) {
        return [
          {
            delayMs: 15,
            chunk: {
              type: 'text',
              text: '<tool_call>{"name":"bench.echo","arguments":{"text":"hello"}}</tool_call>',
            },
          },
          {
            delayMs: 6,
            chunk: {
              type: 'usage',
              usage: { inputTokens: 12, outputTokens: 4 },
            },
          },
          {
            chunk: { type: 'done', finishReason: 'stop' },
          },
        ]
      }

      return [
        {
          delayMs: 14,
          chunk: {
            type: 'text',
            text: `<final>Prompt fallback complete: ${toolOutput}</final>`,
          },
        },
        {
          delayMs: 8,
          chunk: {
            type: 'usage',
            usage: { inputTokens: 10, outputTokens: 7 },
          },
        },
        {
          chunk: { type: 'done', finishReason: 'stop' },
        },
      ]
    },
    {
      toolUse: false,
    },
  )
  const router = createRouterFixture(provider, {
    toolRegistry: tools,
    policy: new PolicyEngine({
      version: 1,
      defaults: {
        mode: 'autonomous',
        unmatched_policy: 'allow',
        max_timeout_ms: 30_000,
        max_output_bytes: 10_485_760,
      },
      tools: {},
    }),
    autonomy: AutonomyLevel.Autonomous,
  })

  const metrics = await collectRunMetrics(
    router.run(
      'Use the tool to echo hello, then answer.',
      benchmarkContext(provider, 'bench-react-prompt-fallback'),
      'react',
    ),
  )

  return {
    firstTokenMs: metrics.firstTokenMs,
    wallMs: metrics.wallMs,
    textLength: metrics.textLength,
    toolResultCount: metrics.toolResultCount,
    providerCallCount: capturedRequests.length,
    promptRequestCount: capturedRequests.filter((request) =>
      request.tools === undefined
      && request.messages.some((message) =>
        message.role === 'system'
        && typeof message.content === 'string'
        && message.content.includes('You do not have native tool calling in this run.'),
      )).length,
    success: metrics.success,
  }
}

async function runGraphCoderStreamBenchmark(): Promise<GraphCoderStreamBenchmark> {
  const provider = new ScriptedProvider('graph-bench', () =>
    createTextScript('Coder benchmark response with streaming text.', {
      firstTokenDelayMs: 16,
      trailingDelayMs: 12,
    }),
  )
  // This scenario measures graph streaming, so use a read-only request that
  // is safe to stream live. Artifact-changing requests intentionally buffer
  // candidate text until completion and outcome gates accept it.
  const router = createRouterFixture(provider, { textDeltaMode: 'live' })
  const metrics = await collectRunMetrics(
    router.run(
      'Explain the retry policy in one concise sentence without changing files.',
      benchmarkContext(provider, 'bench-graph'),
      'coder',
    ),
  )

  return {
    firstTokenMs: metrics.firstTokenMs,
    wallMs: metrics.wallMs,
    textLength: metrics.textLength,
    success: metrics.success,
  }
}

function buildGraphToolState(
  input: string,
  toolCall: ToolCall,
): AgentState {
  return {
    input,
    messages: [
      { role: 'user', content: input },
      {
        role: 'assistant',
        content: '',
        toolCalls: [toolCall],
      },
    ],
    currentStep: 'tools',
    plan: ['run tool', 'summarize'],
    planIndex: 0,
    toolCalls: [toolCall],
    toolResults: [],
    memories: [],
    output: '',
    totalUsage: { inputTokens: 8, outputTokens: 4 },
    iteration: 0,
    maxIterations: 10,
    shouldStop: false,
    taskType: 'code',
  }
}

async function runGraphToolApprovalResumeBenchmark(): Promise<GraphToolApprovalResumeBenchmark> {
  const toolRegistry = new ToolRegistry()
  toolRegistry.register({
    name: 'bench.approval',
    description: 'Deterministic approval benchmark tool',
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string' },
      },
      required: ['path'],
    },
    async execute(input) {
      await sleep(8)
      return {
        output: `read:${String(input.path ?? '')}`,
        status: 'success' as const,
        durationMs: 8,
      }
    },
  })

  const provider = new ScriptedProvider('approval-bench', () =>
    createTextScript('Approval resume benchmark finalized.', {
      firstTokenDelayMs: 17,
      trailingDelayMs: 9,
    }),
  )
  const router = createRouterFixture(provider, { toolRegistry })
  const toolCall: ToolCall = {
    id: 'bench-tool-1',
    name: 'bench.approval',
    arguments: { path: 'README.md' },
  }
  const graphState = buildGraphToolState(
    'Inspect the repository readme and report back.',
    toolCall,
  )
  const context: GraphExecutionContext = {
    graphId: 'coder',
    agentContext: benchmarkContext(provider, 'bench-approval'),
    provider,
    tools: toolRegistry,
    policy: new PolicyEngine(createDefaultPolicy()),
    autonomy: AutonomyLevel.Supervised,
  }
  const checkpoint = createGraphApprovalCheckpoint(
    'bench-approval-request',
    graphState,
    context,
    [toolCall],
    0,
  )
  const metrics = await collectRunMetrics(
    router.resumeFromApprovalCheckpoint(checkpoint, true),
  )

  return {
    postResumeFirstTokenMs: metrics.firstTokenMs,
    wallMs: metrics.wallMs,
    textLength: metrics.textLength,
    toolResultCount: metrics.toolResultCount,
    resumeSuccessRate: metrics.success ? 1 : 0,
  }
}

async function runSingleSemanticSearchBenchmark(): Promise<SemanticSearchBenchmark> {
  const dir = await mkdtemp(join(tmpdir(), 'sepilot-benchmark-semantic-'))
  let index: SqliteSemanticIndex | null = null

  const embedder: MemoryEmbedder = {
    providerId: 'benchmark',
    async embed(texts: string[]) {
      await sleep(6)
      return texts.map((text) => {
        if (text.includes('TypeScript') || text.includes('static typing')) {
          return [1, 0, 0]
        }
        if (text.includes('Kubernetes') || text.includes('deploy')) {
          return [0, 1, 0]
        }
        return [0, 0, 1]
      })
    },
  }

  try {
    const coldStartedAt = performance.now()
    index = await SqliteSemanticIndex.create(join(dir, 'semantic.db'), {
      embedder,
      embeddingModel: 'benchmark-embedding',
    })
    await index.add({
      id: 'bench-sem-1',
      content: 'TypeScript interfaces and static typing patterns',
      source: 'user',
      tags: ['ts'],
    })
    await index.add({
      id: 'bench-sem-2',
      content: 'Kubernetes deploy checklist for production clusters',
      source: 'user',
      tags: ['k8s'],
    })
    const coldResults = await index.search('static typing checklist', {
      type: 'semantic',
    })
    const coldMs = performance.now() - coldStartedAt

    const warmStartedAt = performance.now()
    const warmResults = await index.search('static typing checklist', {
      type: 'semantic',
    })
    const warmMs = performance.now() - warmStartedAt

    return {
      coldMs: roundMs(coldMs),
      warmMs: roundMs(warmMs),
      coldResultCount: coldResults.length,
      warmResultCount: warmResults.length,
      status: index.getStatus().status,
    }
  } finally {
    index?.close()
    await rm(dir, { recursive: true, force: true })
  }
}

async function runSemanticSearchBenchmark(): Promise<SemanticSearchBenchmark> {
  // Smooth sporadic native/sqlite cold-start spikes without hiding systematic regressions.
  const samples: SemanticSearchBenchmark[] = []
  for (let index = 0; index < SEMANTIC_SEARCH_BENCHMARK_SAMPLES; index++) {
    samples.push(await runSingleSemanticSearchBenchmark())
  }
  return medianSample(samples, (sample) => sample.coldMs)
}

async function runConcurrentSaturationBenchmark(): Promise<ConcurrentSaturationBenchmark> {
  const dir = await mkdtemp(join(tmpdir(), 'sepilot-benchmark-saturation-'))
  const sessions = new JsonlSessionStore(join(dir, 'sessions'))
  const skills = new FileSkillRegistry(join(dir, 'skills'))
  await Promise.all([sessions.init(), skills.init()])

  const provider = new ScriptedProvider('saturation-bench', (request) =>
    createTextScript(
      `Handled ${extractUserMessage(request)}`,
      {
        firstTokenDelayMs: 20,
        trailingDelayMs: 25,
        usage: { inputTokens: 5, outputTokens: 5 },
      },
    ),
  )
  const providerRegistry = new ProviderRegistry()
  providerRegistry.register(provider, { default: true })

  const runLimiter = new RunLimiter({
    maxActive: 4,
    maxQueued: 0,
    queueTimeoutMs: 25,
  })

  const app = await createApp({
    port: 0,
    host: '127.0.0.1',
    runtime: {
      config: {
        version: 1,
        device: { id: 'bench-saturation', name: 'bench-saturation', role: 'desktop' },
        daemon: { port: 0, host: '127.0.0.1', resumeArtifactRetentionDays: 30 },
        gateway: { url: 'http://127.0.0.1:17610' },
        providers: [],
        agent: { autonomy: 'supervised', thinkingLevel: 'medium', mode: 'react' },
        channels: [],
        memory: { encryption: false },
        security: { toolPolicy: 'policies.yaml', auditLog: false, sandbox: 'local' },
        observability: { telemetry: false, otlpEndpoint: '' },
      },
      providerRegistry,
      toolRegistry: new ToolRegistry(),
      policyEngine: new PolicyEngine(createDefaultPolicy()),
      sessions,
      autonomy: AutonomyLevel.Supervised,
      skillRegistry: skills,
      hookRegistry: new HookRegistry(),
      approvalRegistry: new ApprovalRegistry(sessions),
      runLimiter,
      channels: [],
    } as unknown as RuntimeServices,
  })
  await app.ready()

  try {
    const startedAt = performance.now()
    const responses = await Promise.all(
      Array.from({ length: 5 }, (_, index) =>
        app.inject({
          method: 'POST',
          url: '/api/v1/chat',
          payload: { message: `load-${index + 1}` },
        }),
      ),
    )
    const wallMs = performance.now() - startedAt

    return {
      wallMs: roundMs(wallMs),
      acceptedCount: responses.filter((response) => response.statusCode === 200).length,
      rejectedCount: responses.filter((response) => response.statusCode === 503).length,
      rejectedStatusCodes: responses
        .filter((response) => response.statusCode !== 200)
        .map((response) => response.statusCode)
        .sort((left, right) => left - right),
    }
  } finally {
    await app.close()
    await rm(dir, { recursive: true, force: true })
  }
}

function extractUserMessage(request: ChatRequest): string {
  const userMessage = [...request.messages]
    .reverse()
    .find((message) => message.role === 'user')
  return typeof userMessage?.content === 'string' ? userMessage.content : 'unknown'
}

export async function runBenchmarks(): Promise<BenchmarkResults> {
  const reactSimpleQa = await runReactSimpleQaBenchmark()
  const reactPromptFallback = await runReactPromptFallbackBenchmark()
  const graphCoderStream = await runGraphCoderStreamBenchmark()
  const graphToolApprovalResume = await runGraphToolApprovalResumeBenchmark()
  const semanticSearch = await runSemanticSearchBenchmark()
  const concurrentSaturation = await runConcurrentSaturationBenchmark()

  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    nodeVersion: process.version,
    idleMemoryRssBytes: process.memoryUsage().rss,
    scenarios: {
      react_simple_qa: reactSimpleQa,
      react_prompt_fallback: reactPromptFallback,
      graph_coder_stream: graphCoderStream,
      graph_tool_approval_resume: graphToolApprovalResume,
      semantic_search: semanticSearch,
      concurrent_4_run_saturation: concurrentSaturation,
    },
  }
}

export function compareWithBaseline(
  current: BenchmarkResults,
  baseline: BenchmarkResults,
): BenchmarkCheckFailure[] {
  const failures: BenchmarkCheckFailure[] = []

  for (const rule of METRIC_RULES) {
    const actual = getMetric(current, rule.path)
    const expected = getMetric(baseline, rule.path)

    if (typeof actual !== 'number' || typeof expected !== 'number') {
      failures.push({
        metric: rule.path,
        baseline: Number(expected ?? NaN),
        actual: Number(actual ?? NaN),
        message: 'Metric is missing or not numeric',
      })
      continue
    }

    if (rule.kind === 'latency') {
      const allowed = expected * (1 + rule.percentTolerance) + rule.absoluteToleranceMs
      if (actual > allowed) {
        failures.push({
          metric: rule.path,
          baseline: expected,
          actual,
          allowed: roundMs(allowed),
          message: `Latency regressed beyond tolerance (${roundMs(allowed)}ms allowed)`,
        })
      }
      continue
    }

    if (rule.kind === 'memory') {
      const allowed = expected * (1 + rule.percentTolerance) + rule.absoluteToleranceBytes
      if (actual > allowed) {
        failures.push({
          metric: rule.path,
          baseline: expected,
          actual,
          allowed,
          message: `Memory regressed beyond tolerance (${Math.round(allowed / 1024 / 1024)} MiB allowed)`,
        })
      }
      continue
    }

    if (rule.kind === 'minimum') {
      if (actual < expected) {
        failures.push({
          metric: rule.path,
          baseline: expected,
          actual,
          message: 'Metric fell below the recorded baseline',
        })
      }
      continue
    }

    if (actual !== expected) {
      failures.push({
        metric: rule.path,
        baseline: expected,
        actual,
        message: 'Metric changed from the recorded baseline',
      })
    }
  }

  return failures
}

function getMetric(record: BenchmarkResults, path: string): unknown {
  return path
    .split('.')
    .reduce<unknown>((value, key) => {
      if (!value || typeof value !== 'object') {
        return undefined
      }
      return (value as Record<string, unknown>)[key]
    }, record)
}

function formatResults(result: BenchmarkResults): string {
  return [
    `React simple QA: first token ${result.scenarios.react_simple_qa.firstTokenMs}ms, wall ${result.scenarios.react_simple_qa.wallMs}ms`,
    `React prompt fallback: first token ${result.scenarios.react_prompt_fallback.firstTokenMs}ms, wall ${result.scenarios.react_prompt_fallback.wallMs}ms, provider calls ${result.scenarios.react_prompt_fallback.providerCallCount}`,
    `Graph coder stream: first token ${result.scenarios.graph_coder_stream.firstTokenMs}ms, wall ${result.scenarios.graph_coder_stream.wallMs}ms`,
    `Graph approval resume: first token ${result.scenarios.graph_tool_approval_resume.postResumeFirstTokenMs}ms, wall ${result.scenarios.graph_tool_approval_resume.wallMs}ms, success ${result.scenarios.graph_tool_approval_resume.resumeSuccessRate}`,
    `Semantic search: cold ${result.scenarios.semantic_search.coldMs}ms, warm ${result.scenarios.semantic_search.warmMs}ms`,
    `Concurrent 4-run saturation: wall ${result.scenarios.concurrent_4_run_saturation.wallMs}ms, accepted ${result.scenarios.concurrent_4_run_saturation.acceptedCount}, rejected ${result.scenarios.concurrent_4_run_saturation.rejectedCount}`,
    `Idle RSS: ${Math.round(result.idleMemoryRssBytes / 1024 / 1024)} MiB`,
  ].join('\n')
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  await writeFile(path, JSON.stringify(value, null, 2) + '\n', 'utf-8')
}

export async function main(): Promise<void> {
  const args = process.argv.slice(2)
  const check = args.includes('--check')
  const updateBaseline = args.includes('--update-baseline')
  const baselinePath = readFlagValue(args, '--baseline') ?? defaultBaselinePath
  const outputPath = readFlagValue(args, '--output') ?? defaultOutputPath

  const result = await runBenchmarks()
  await writeJson(outputPath, result)
  console.log(formatResults(result))
  console.log(`Saved benchmark report to ${outputPath}`)

  if (updateBaseline) {
    await writeJson(baselinePath, result)
    console.log(`Updated benchmark baseline at ${baselinePath}`)
  }

  if (!check) {
    return
  }

  const baseline = JSON.parse(
    await readFile(baselinePath, 'utf-8'),
  ) as BenchmarkResults
  const failures = compareWithBaseline(result, baseline)
  result.check = {
    ok: failures.length === 0,
    baselinePath,
    failures,
  }
  await writeJson(outputPath, result)

  if (failures.length === 0) {
    console.log(`Benchmark check passed against ${baselinePath}`)
    return
  }

  console.error(`Benchmark check failed against ${baselinePath}`)
  for (const failure of failures) {
    const allowed = typeof failure.allowed === 'number'
      ? `, allowed ${failure.allowed}`
      : ''
    console.error(
      `- ${failure.metric}: baseline ${failure.baseline}, actual ${failure.actual}${allowed} (${failure.message})`,
    )
  }
  process.exitCode = 1
}

function readFlagValue(args: string[], flag: string): string | undefined {
  const index = args.indexOf(flag)
  if (index < 0) {
    return undefined
  }
  return args[index + 1]
}

function isEntrypoint(): boolean {
  const entry = process.argv[1]
  if (!entry) {
    return false
  }
  return import.meta.url === pathToFileURL(entry).href
}

export const __testables = {
  collectRunMetrics,
  consumeStepsAsChatResponse,
  extractToolMessageText,
  formatResults,
  getMetric,
  isEntrypoint,
  readFlagValue,
}

if (isEntrypoint()) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
