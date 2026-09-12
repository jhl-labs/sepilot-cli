import { memoryEvidenceSchema } from './evidence.js'
import { retrieveRelevantMemory } from './auto-retrieve.js'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { homedir, tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { z } from 'zod'
import { GraphAgentRegistry } from '../agent/graph/registry.js'
import { parseConfig } from '../config/loader.js'
import { validateConfig } from '../config/validator.js'
import { loadManagedEnvFile } from '../config/env-file.js'
import { HookRegistry } from '../hook/registry.js'
import {
  closeProviderHttpDispatcher,
  configureProviderHttpTimeout,
  getProviderDispatcherPlan,
  planProviderDispatcher,
} from '../providers/http-timeout.js'
import { createPluginLoader } from '../generated/feature-registration.js'
import { createChannelFactoryRegistry } from '../server/runtime/channels.js'
import {
  buildProviderRegistry,
  createProviderFactoryRegistry,
} from '../server/runtime/providers.js'
import { FileSkillRegistry } from '../skills/registry.js'
import { ToolRegistry } from '../tools/registry.js'
import { findDedupCandidates } from './dedup.js'
import { SqliteSemanticIndex, type MemoryEmbedder } from './semantic-index.js'
import { selectRelevantMemories, tokenizeMemoryText } from './relevance.js'

const memorySourceSchema = z.enum(['conversation', 'document', 'skill', 'user'])

const fixtureMemorySchema = z.object({
  evidence: memoryEvidenceSchema.optional(),
  id: z.string().min(1),
  content: z.string().min(1),
  source: memorySourceSchema.default('user'),
  tags: z.array(z.string().min(1)).default([]),
})

const fixtureDocumentSchema = z.object({
  id: z.string().min(1),
  title: z.string().min(1),
  content: z.string().min(1),
  path: z.string().min(1).optional(),
  mimeType: z.string().min(1).optional(),
  sourceFileId: z.string().min(1).optional(),
  tags: z.array(z.string().min(1)).default([]),
})

const retrievalCaseSchema = z.object({
  scopeTags: z.array(z.string()).optional(),
  asOf: z.string().datetime({ offset: true }).optional(),
  useAutoRetrieve: z.boolean().optional(),
  maxChars: z.number().int().min(200).max(8000).optional(),
  expectedText: z.array(z.string()).optional(),
  forbiddenText: z.array(z.string()).optional(),
  operations: z.array(z.discriminatedUnion('action', [
    z.object({ action: z.literal('update'), id: z.string(), content: z.string(), evidence: memoryEvidenceSchema.optional() }),
    z.object({ action: z.literal('forget'), id: z.string() }),
  ])).optional(),
  name: z.string().min(1).optional(),
  query: z.string().min(1),
  expectedIds: z.array(z.string().min(1)).default([]),
  forbiddenIds: z.array(z.string().min(1)).default([]),
  type: z.enum(['semantic', 'keyword', 'hybrid']).default('hybrid'),
  limit: z.number().int().positive().max(20).default(3),
})

const documentRetrievalCaseSchema = z.object({
  name: z.string().min(1).optional(),
  query: z.string().min(1),
  expectedDocumentIds: z.array(z.string().min(1)).default([]),
  forbiddenDocumentIds: z.array(z.string().min(1)).default([]),
  expectedCitationIncludes: z.array(z.string().min(1)).default([]),
  expectedSnippetIncludes: z.array(z.string().min(1)).default([]),
  type: z.enum(['semantic', 'keyword', 'hybrid']).default('hybrid'),
  limit: z.number().int().positive().max(20).default(3),
  documentId: z.string().min(1).optional(),
})

const dedupCaseSchema = z.object({
  name: z.string().min(1).optional(),
  primaryId: z.string().min(1),
  expectedDuplicateIds: z.array(z.string().min(1)).default([]),
  forbiddenIds: z.array(z.string().min(1)).default([]),
})

const memoryEvalFixtureSchema = z
  .object({
    version: z.literal(1),
    memories: z.array(fixtureMemorySchema).default([]),
    documents: z.array(fixtureDocumentSchema).default([]),
    retrieval: z.array(retrievalCaseSchema).default([]),
    documentRetrieval: z.array(documentRetrievalCaseSchema).default([]),
    dedup: z.array(dedupCaseSchema).default([]),
  })
  .refine((fixture) => fixture.memories.length + fixture.documents.length > 0, {
    message: 'At least one memory or document fixture is required',
    path: ['memories'],
  })

export type MemoryEvalFixture = z.infer<typeof memoryEvalFixtureSchema>

export interface RetrievalEvalCaseResult {
  injectedChars?: number
  estimatedContextTokens?: number
  latencyMs?: number
  name: string
  query: string
  type: 'semantic' | 'keyword' | 'hybrid'
  expectedIds: string[]
  forbiddenIds: string[]
  rawSearchIds: string[]
  selectedIds: string[]
  hitIds: string[]
  missedIds: string[]
  forbiddenHitIds: string[]
  recall: number
  precision: number
  top1Hit: boolean
  passed: boolean
}

export interface DedupEvalCaseResult {
  name: string
  primaryId: string
  expectedDuplicateIds: string[]
  forbiddenIds: string[]
  candidateIds: string[]
  hitIds: string[]
  missedIds: string[]
  forbiddenHitIds: string[]
  recall: number
  precision: number
  passed: boolean
}

export interface DocumentRetrievalEvalCaseResult {
  name: string
  query: string
  type: 'semantic' | 'keyword' | 'hybrid'
  documentId?: string
  expectedDocumentIds: string[]
  forbiddenDocumentIds: string[]
  expectedCitationIncludes: string[]
  expectedSnippetIncludes: string[]
  rawChunkIds: string[]
  selectedChunkIds: string[]
  selectedDocumentIds: string[]
  hitDocumentIds: string[]
  missedDocumentIds: string[]
  forbiddenHitDocumentIds: string[]
  citationMatches: string[]
  snippetMatches: string[]
  citationCoverage: number
  snippetCoverage: number
  top1Hit: boolean
  passed: boolean
}

export interface MemoryEvalSummary {
  ok: boolean
  retrievalCaseCount: number
  retrievalPassCount: number
  retrievalRecall: number
  retrievalPrecision: number
  retrievalTop1Accuracy: number
  wrongMemoryInjectionRate: number
  documentRetrievalCaseCount: number
  documentRetrievalPassCount: number
  documentRetrievalRecall: number
  documentRetrievalPrecision: number
  documentRetrievalTop1Accuracy: number
  wrongDocumentInjectionRate: number
  documentCitationCoverage: number
  documentSnippetCoverage: number
  dedupCaseCount: number
  dedupPassCount: number
  dedupRecall: number
  dedupPrecision: number
  dedupFalsePositiveRate: number
}

export interface MemoryEvalReport {
  version: number
  generatedAt: string
  fixturePath?: string
  embedder: {
    mode: 'lexical' | 'live'
    providerId: string
    model: string
    vocabularySize?: number
  }
  runtime: {
    status: string
    vecAvailable: boolean
    dimensions?: number
  }
  summary: MemoryEvalSummary
  retrieval: RetrievalEvalCaseResult[]
  documentRetrieval: DocumentRetrievalEvalCaseResult[]
  dedup: DedupEvalCaseResult[]
}

const EVAL_EMBEDDING_MODEL = 'memory-eval-lexical-v1'

export interface MemoryEvalRunOptions {
  fixturePath?: string
  embedder?: MemoryEmbedder
  embeddingModel?: string
  configuredProviderId?: string
  embedderMode?: 'lexical' | 'live'
  vocabularySize?: number
}

export interface MemoryEvalLiveOptions {
  dataDir?: string
  configPath?: string
  providerId?: string
  model?: string
}

interface ResolvedMemoryEvalEmbedder {
  mode: 'lexical' | 'live'
  providerId: string
  model: string
  vocabularySize?: number
  embedder: MemoryEmbedder
  dispose?: () => Promise<void>
}

export async function loadMemoryEvalFixture(path: string): Promise<MemoryEvalFixture> {
  const content = await readFile(path, 'utf-8')
  const parsed = memoryEvalFixtureSchema.parse(JSON.parse(content))
  validateFixtureReferences(parsed)
  return parsed
}

export async function runMemoryEvalFixture(
  fixture: MemoryEvalFixture,
  options: MemoryEvalRunOptions = {},
): Promise<MemoryEvalReport> {
  const normalizedFixture = memoryEvalFixtureSchema.parse(fixture)
  validateFixtureReferences(normalizedFixture)

  const dir = await mkdtemp(join(tmpdir(), 'sepilot-memory-eval-'))
  const embedderConfig = resolveMemoryEvalEmbedder(normalizedFixture, options)
  let index: SqliteSemanticIndex | null = null

  try {
    index = await SqliteSemanticIndex.create(join(dir, 'memory-eval.db'), {
      embedder: embedderConfig.embedder,
      embeddingModel: embedderConfig.model,
      configuredProviderId: embedderConfig.providerId,
    })

    for (const memory of normalizedFixture.memories) {
      await index.add(memory)
    }
    for (const document of normalizedFixture.documents) {
      await index.ingestDocument(document)
    }
    await index.backfill()

    const retrieval = await evaluateRetrievalCases(index, normalizedFixture)
    const documentRetrieval = await evaluateDocumentRetrievalCases(index, normalizedFixture)
    const dedup = await evaluateDedupCases(index, normalizedFixture)
    const runtime = index.getStatus()
    const summary = summarizeMemoryEval(retrieval, documentRetrieval, dedup)

    return {
      version: 1,
      generatedAt: new Date().toISOString(),
      fixturePath: options.fixturePath,
      embedder: {
        mode: embedderConfig.mode,
        providerId: embedderConfig.providerId,
        model: embedderConfig.model,
        vocabularySize: embedderConfig.vocabularySize,
      },
      runtime: {
        status: runtime.status,
        vecAvailable: runtime.vecAvailable,
        dimensions: runtime.dimensions,
      },
      summary,
      retrieval,
      documentRetrieval,
      dedup,
    }
  } finally {
    index?.close()
    await rm(dir, { recursive: true, force: true })
  }
}

export function formatMemoryEvalReport(report: MemoryEvalReport): string {
  const embedderDetail = report.embedder.vocabularySize
    ? `${report.embedder.providerId}/${report.embedder.model} (${report.embedder.vocabularySize} tokens)`
    : `${report.embedder.providerId}/${report.embedder.model}`
  const lines = [
    'Memory eval',
    report.fixturePath ? `Fixture: ${report.fixturePath}` : undefined,
    `Runtime: ${report.runtime.status}${report.runtime.vecAvailable ? '' : ' (sqlite-vec unavailable)'}`,
    `Embedder: ${report.embedder.mode} ${embedderDetail}`,
    `Retrieval: ${report.summary.retrievalPassCount}/${report.summary.retrievalCaseCount} passed, recall ${formatRatio(report.summary.retrievalRecall)}, precision ${formatRatio(report.summary.retrievalPrecision)}, top1 ${formatRatio(report.summary.retrievalTop1Accuracy)}, wrong-injection ${formatRatio(report.summary.wrongMemoryInjectionRate)}`,
    report.summary.documentRetrievalCaseCount > 0
      ? `Document retrieval: ${report.summary.documentRetrievalPassCount}/${report.summary.documentRetrievalCaseCount} passed, recall ${formatRatio(report.summary.documentRetrievalRecall)}, precision ${formatRatio(report.summary.documentRetrievalPrecision)}, top1 ${formatRatio(report.summary.documentRetrievalTop1Accuracy)}, wrong-doc ${formatRatio(report.summary.wrongDocumentInjectionRate)}, citation ${formatRatio(report.summary.documentCitationCoverage)}, snippet ${formatRatio(report.summary.documentSnippetCoverage)}`
      : undefined,
    `Dedup candidates: ${report.summary.dedupPassCount}/${report.summary.dedupCaseCount} passed, recall ${formatRatio(report.summary.dedupRecall)}, precision ${formatRatio(report.summary.dedupPrecision)}, false-positive ${formatRatio(report.summary.dedupFalsePositiveRate)}`,
  ].filter(Boolean) as string[]

  const failures = [
    ...report.retrieval
      .filter((item) => !item.passed)
      .map(
        (item) =>
          `retrieval/${item.name}: selected [${item.selectedIds.join(', ')}], missed [${item.missedIds.join(', ')}], forbidden [${item.forbiddenHitIds.join(', ')}]`,
      ),
    ...report.documentRetrieval
      .filter((item) => !item.passed)
      .map(
        (item) =>
          `document/${item.name}: documents [${item.selectedDocumentIds.join(', ')}], missed [${item.missedDocumentIds.join(', ')}], forbidden [${item.forbiddenHitDocumentIds.join(', ')}], citation ${formatRatio(item.citationCoverage)}, snippet ${formatRatio(item.snippetCoverage)}`,
      ),
    ...report.dedup
      .filter((item) => !item.passed)
      .map(
        (item) =>
          `dedup/${item.name}: candidates [${item.candidateIds.join(', ')}], missed [${item.missedIds.join(', ')}], forbidden [${item.forbiddenHitIds.join(', ')}]`,
      ),
  ]

  if (failures.length > 0) {
    lines.push('', 'Failing cases:')
    for (const failure of failures) {
      lines.push(`- ${failure}`)
    }
  }

  return lines.join('\n')
}

export async function main(args = process.argv.slice(2)): Promise<void> {
  const json = args.includes('--json')
  const live = args.includes('--live')
  const outputPath = readFlagValue(args, '--output')
  const configPath = readFlagValue(args, '--config')
  const dataDir = readFlagValue(args, '--data-dir')
  const providerId = readFlagValue(args, '--provider')
  const model = readFlagValue(args, '--model')
  const fixturePath = readPositionalArg(args)

  if (!fixturePath) {
    throw new Error(
      'Usage: sepilotd memory-eval <fixture.json> [--live] [--data-dir <path>] [--config <path>] [--provider <id>] [--model <id>] [--json] [--output <path>]',
    )
  }

  const absoluteFixturePath = resolve(fixturePath)
  const fixture = await loadMemoryEvalFixture(absoluteFixturePath)
  const liveEmbedder = live
    ? await loadConfiguredMemoryEvalEmbedder({
        dataDir,
        configPath,
        providerId,
        model,
      })
    : undefined
  try {
    const report = await runMemoryEvalFixture(fixture, {
      fixturePath: absoluteFixturePath,
      embedder: liveEmbedder?.embedder,
      embeddingModel: liveEmbedder?.model,
      configuredProviderId: liveEmbedder?.providerId,
      embedderMode: liveEmbedder?.mode,
      vocabularySize: liveEmbedder?.vocabularySize,
    })

    if (outputPath) {
      await writeJson(resolve(outputPath), report)
    }

    if (json) {
      console.log(JSON.stringify(report, null, 2))
    } else {
      console.log(formatMemoryEvalReport(report))
      if (outputPath) {
        console.log(`Saved memory eval report to ${resolve(outputPath)}`)
      }
    }

    if (!report.summary.ok) {
      process.exitCode = 1
    }
  } finally {
    await liveEmbedder?.dispose?.()
  }
}

async function evaluateRetrievalCases(
  index: SqliteSemanticIndex,
  fixture: MemoryEvalFixture,
): Promise<RetrievalEvalCaseResult[]> {
  const results: RetrievalEvalCaseResult[] = []

  for (const [caseIndex, scenario] of fixture.retrieval.entries()) {
    for (const operation of scenario.operations ?? []) {
      if (operation.action === 'forget') await index.delete(operation.id)
      else {
        const existing = await index.get(operation.id)
        if (!existing) throw new Error(`Update references missing memory: ${operation.id}`)
        await index.add({ ...existing, content: operation.content, evidence: operation.evidence ?? existing.evidence })
      }
    }
    const started = performance.now()
    const rawResults = scenario.useAutoRetrieve ? [] : await index.search(scenario.query, {
      scopeTags: scenario.scopeTags,
      asOf: scenario.asOf,
      type: scenario.type,
      limit: Math.max(scenario.limit, 6),
      minScore: scenario.type === 'keyword' ? undefined : 0.08,
    })
    const injected = scenario.useAutoRetrieve ? await retrieveRelevantMemory({
      semanticIndex: index, query: scenario.query, scopeTags: scenario.scopeTags, asOf: scenario.asOf,
      limit: scenario.limit, maxChars: scenario.maxChars, searchType: scenario.type, minScore: 0,
    }) : undefined
    const selected = injected?.hits ?? selectRelevantMemories(scenario.query, rawResults, { limit: scenario.limit })
    const selectedIds = selected.map((entry) => entry.id)
    const hitIds = intersectIds(selectedIds, scenario.expectedIds)
    const missedIds = subtractIds(scenario.expectedIds, selectedIds)
    const forbiddenHitIds = intersectIds(selectedIds, scenario.forbiddenIds)

    results.push({
      name: scenario.name ?? `retrieval-${caseIndex + 1}`,
      injectedChars: injected?.block.length,
      estimatedContextTokens: injected ? Math.ceil(injected.block.length / 4) : undefined,
      latencyMs: performance.now() - started,
      query: scenario.query,
      type: scenario.type,
      expectedIds: scenario.expectedIds,
      forbiddenIds: scenario.forbiddenIds,
      rawSearchIds: rawResults.map((entry) => entry.id),
      selectedIds,
      hitIds,
      missedIds,
      forbiddenHitIds,
      recall: ratio(hitIds.length, scenario.expectedIds.length, 1),
      precision: ratio(
        hitIds.length,
        selectedIds.length,
        scenario.expectedIds.length === 0 ? 1 : 0,
      ),
      top1Hit: selectedIds.length > 0 && scenario.expectedIds.includes(selectedIds[0]),
      passed: missedIds.length === 0 && forbiddenHitIds.length === 0
        && (scenario.expectedText ?? []).every((text) => (injected?.block ?? selected.map((entry) => entry.content).join('\n')).includes(text))
        && !(scenario.forbiddenText ?? []).some((text) => (injected?.block ?? selected.map((entry) => entry.content).join('\n')).includes(text)),
    })
  }

  return results
}

async function evaluateDocumentRetrievalCases(
  index: SqliteSemanticIndex,
  fixture: MemoryEvalFixture,
): Promise<DocumentRetrievalEvalCaseResult[]> {
  const results: DocumentRetrievalEvalCaseResult[] = []

  for (const [caseIndex, scenario] of fixture.documentRetrieval.entries()) {
    const rawResults = await index.searchDocuments(scenario.query, {
      type: scenario.type,
      limit: scenario.limit,
      minScore: scenario.type === 'keyword' ? undefined : 0.08,
      documentId: scenario.documentId,
    })
    const selectedChunkIds = rawResults.map((entry) => entry.id)
    const selectedDocumentIds = uniqueIds(rawResults.map((entry) => entry.documentId))
    const hitDocumentIds = intersectIds(selectedDocumentIds, scenario.expectedDocumentIds)
    const missedDocumentIds = subtractIds(scenario.expectedDocumentIds, selectedDocumentIds)
    const forbiddenHitDocumentIds = intersectIds(selectedDocumentIds, scenario.forbiddenDocumentIds)
    const citationMatches = collectTextExpectationMatches(
      rawResults.map((entry) => entry.citationLabel ?? ''),
      scenario.expectedCitationIncludes,
    )
    const snippetMatches = collectTextExpectationMatches(
      rawResults.map((entry) => entry.snippet ?? ''),
      scenario.expectedSnippetIncludes,
    )

    results.push({
      name: scenario.name ?? `document-retrieval-${caseIndex + 1}`,
      query: scenario.query,
      type: scenario.type,
      documentId: scenario.documentId,
      expectedDocumentIds: scenario.expectedDocumentIds,
      forbiddenDocumentIds: scenario.forbiddenDocumentIds,
      expectedCitationIncludes: scenario.expectedCitationIncludes,
      expectedSnippetIncludes: scenario.expectedSnippetIncludes,
      rawChunkIds: rawResults.map((entry) => entry.id),
      selectedChunkIds,
      selectedDocumentIds,
      hitDocumentIds,
      missedDocumentIds,
      forbiddenHitDocumentIds,
      citationMatches,
      snippetMatches,
      citationCoverage: evaluateCoverage(
        scenario.expectedCitationIncludes,
        citationMatches,
        rawResults,
        (entry) => entry.citationLabel,
      ),
      snippetCoverage: evaluateCoverage(
        scenario.expectedSnippetIncludes,
        snippetMatches,
        rawResults,
        (entry) => entry.snippet,
      ),
      top1Hit:
        rawResults.length > 0 && scenario.expectedDocumentIds.includes(rawResults[0].documentId),
      passed:
        missedDocumentIds.length === 0 &&
        forbiddenHitDocumentIds.length === 0 &&
        evaluateCoverage(
          scenario.expectedCitationIncludes,
          citationMatches,
          rawResults,
          (entry) => entry.citationLabel,
        ) === 1 &&
        evaluateCoverage(
          scenario.expectedSnippetIncludes,
          snippetMatches,
          rawResults,
          (entry) => entry.snippet,
        ) === 1,
    })
  }

  return results
}

async function evaluateDedupCases(
  index: SqliteSemanticIndex,
  fixture: MemoryEvalFixture,
): Promise<DedupEvalCaseResult[]> {
  const memoryById = new Map(fixture.memories.map((memory) => [memory.id, memory]))
  const results: DedupEvalCaseResult[] = []

  for (const [caseIndex, scenario] of fixture.dedup.entries()) {
    const primary = memoryById.get(scenario.primaryId)
    if (!primary) {
      throw new Error(`Dedup case references missing primary memory: ${scenario.primaryId}`)
    }

    const candidates = await findDedupCandidates(index, primary)
    const candidateIds = candidates.map((entry) => entry.id)
    const hitIds = intersectIds(candidateIds, scenario.expectedDuplicateIds)
    const missedIds = subtractIds(scenario.expectedDuplicateIds, candidateIds)
    const forbiddenHitIds = intersectIds(candidateIds, scenario.forbiddenIds)

    results.push({
      name: scenario.name ?? `dedup-${caseIndex + 1}`,
      primaryId: scenario.primaryId,
      expectedDuplicateIds: scenario.expectedDuplicateIds,
      forbiddenIds: scenario.forbiddenIds,
      candidateIds,
      hitIds,
      missedIds,
      forbiddenHitIds,
      recall: ratio(hitIds.length, scenario.expectedDuplicateIds.length, 1),
      precision: ratio(
        hitIds.length,
        candidateIds.length,
        scenario.expectedDuplicateIds.length === 0 ? 1 : 0,
      ),
      passed: missedIds.length === 0 && forbiddenHitIds.length === 0,
    })
  }

  return results
}

function summarizeMemoryEval(
  retrieval: RetrievalEvalCaseResult[],
  documentRetrieval: DocumentRetrievalEvalCaseResult[],
  dedup: DedupEvalCaseResult[],
): MemoryEvalSummary {
  return {
    ok:
      retrieval.every((item) => item.passed) &&
      documentRetrieval.every((item) => item.passed) &&
      dedup.every((item) => item.passed),
    retrievalCaseCount: retrieval.length,
    retrievalPassCount: retrieval.filter((item) => item.passed).length,
    retrievalRecall: average(retrieval.map((item) => item.recall)),
    retrievalPrecision: average(retrieval.map((item) => item.precision)),
    retrievalTop1Accuracy: average(retrieval.map((item) => (item.top1Hit ? 1 : 0))),
    wrongMemoryInjectionRate: average(
      retrieval.map((item) => ratio(item.forbiddenHitIds.length, item.selectedIds.length, 0)),
    ),
    documentRetrievalCaseCount: documentRetrieval.length,
    documentRetrievalPassCount: documentRetrieval.filter((item) => item.passed).length,
    documentRetrievalRecall: average(
      documentRetrieval.map((item) =>
        ratio(item.hitDocumentIds.length, item.expectedDocumentIds.length, 1),
      ),
    ),
    documentRetrievalPrecision: average(
      documentRetrieval.map((item) =>
        ratio(
          item.hitDocumentIds.length,
          item.selectedDocumentIds.length,
          item.expectedDocumentIds.length === 0 ? 1 : 0,
        ),
      ),
    ),
    documentRetrievalTop1Accuracy: average(documentRetrieval.map((item) => (item.top1Hit ? 1 : 0))),
    wrongDocumentInjectionRate: average(
      documentRetrieval.map((item) =>
        ratio(item.forbiddenHitDocumentIds.length, item.selectedDocumentIds.length, 0),
      ),
    ),
    documentCitationCoverage: average(documentRetrieval.map((item) => item.citationCoverage)),
    documentSnippetCoverage: average(documentRetrieval.map((item) => item.snippetCoverage)),
    dedupCaseCount: dedup.length,
    dedupPassCount: dedup.filter((item) => item.passed).length,
    dedupRecall: average(dedup.map((item) => item.recall)),
    dedupPrecision: average(dedup.map((item) => item.precision)),
    dedupFalsePositiveRate: average(
      dedup.map((item) => ratio(item.forbiddenHitIds.length, item.candidateIds.length, 0)),
    ),
  }
}

export async function loadConfiguredMemoryEvalEmbedder(
  options: MemoryEvalLiveOptions = {},
): Promise<ResolvedMemoryEvalEmbedder> {
  const dataDir = resolve(
    options.dataDir ?? process.env.SEPILOTD_DATA_DIR ?? join(homedir(), '.sepilotd'),
  )
  const configPath = resolve(options.configPath ?? join(dataDir, 'config.yaml'))
  await loadManagedEnvFile(dataDir)
  const configText = await readFile(configPath, 'utf-8')
  const config = parseConfig(configText)
  const validation = validateConfig(config)
  if (!validation.valid) {
    throw new Error(`Invalid config: ${validation.errors.join(' ')}`)
  }

  // Install the process-wide egress policy before plugin imports or provider
  // factories can perform network I/O. A programmatic caller may already own
  // the daemon dispatcher: reuse an identical policy without closing it, and
  // refuse to replace a different live policy that cannot be restored safely.
  const requestedNetworkPlan = planProviderDispatcher(config.network)
  const existingNetworkPlan = getProviderDispatcherPlan()
  if (
    existingNetworkPlan
    && existingNetworkPlan.fingerprint !== requestedNetworkPlan.fingerprint
  ) {
    throw new Error(
      'Memory eval network policy differs from the active daemon policy; run memory-eval in a separate process.',
    )
  }
  const ownsNetworkDispatcher = existingNetworkPlan === null
    && configureProviderHttpTimeout(config.network) !== null
  let dispatcherOwnershipTransferred = false
  try {
    const providerFactoryRegistry = createProviderFactoryRegistry()
    const pluginLoader = createPluginLoader(dataDir)
    const skillRegistry = new FileSkillRegistry(join(dataDir, 'skills'))
    await skillRegistry.init()
    // pluginLoader is null when the `plugins` feature is disabled in this build;
    // the eval embedder still works without plugin-contributed providers.
    if (pluginLoader) {
      await pluginLoader.loadAll({
        providers: providerFactoryRegistry,
        channels: createChannelFactoryRegistry(),
        tools: new ToolRegistry(),
        hooks: new HookRegistry(),
        skills: skillRegistry,
        graphs: new GraphAgentRegistry(),
      })
    }

    const providerRegistry = buildProviderRegistry(config, providerFactoryRegistry)
    const providerId = options.providerId ?? config.memory.embeddingProvider
    const model = options.model ?? config.memory.embeddingModel

    if (!providerId || !model) {
      throw new Error(
        'Semantic memory is not configured. Set memory.embeddingProvider and memory.embeddingModel or pass --provider and --model.',
      )
    }

    const provider = providerRegistry.get(providerId)
    if (!provider) {
      throw new Error(`Embedding provider "${providerId}" is not available from the current config.`)
    }
    if (typeof provider.embed !== 'function') {
      throw new Error(`Provider "${providerId}" does not expose embed().`)
    }

    dispatcherOwnershipTransferred = true
    return {
      mode: 'live',
      providerId,
      model,
      embedder: {
        providerId,
        embed: (texts, requestedModel) => provider.embed!(texts, requestedModel ?? model),
      },
      ...(ownsNetworkDispatcher ? { dispose: closeProviderHttpDispatcher } : {}),
    }
  } finally {
    if (!dispatcherOwnershipTransferred && ownsNetworkDispatcher) {
      await closeProviderHttpDispatcher()
    }
  }
}

function createLexicalEvalEmbedder(
  fixture: MemoryEvalFixture,
): MemoryEmbedder & { vocabularySize: number } {
  const vocabulary = buildVocabulary(fixture)
  const vocabularyIndex = new Map(vocabulary.map((token, index) => [token, index]))
  const dimensions = vocabulary.length

  return {
    providerId: 'memory-eval-lexical',
    vocabularySize: dimensions,
    async embed(texts: string[]) {
      return texts.map((text) => {
        const vector = new Array<number>(dimensions).fill(0)
        for (const token of tokenizeMemoryText(text)) {
          const index = vocabularyIndex.get(token)
          if (index !== undefined) {
            vector[index] += 1
          }
        }

        let sumSquares = 0
        for (const value of vector) {
          sumSquares += value * value
        }
        if (sumSquares === 0) {
          vector[0] = 1
          return vector
        }

        const norm = Math.sqrt(sumSquares)
        return vector.map((value) => value / norm)
      })
    },
  }
}

function resolveMemoryEvalEmbedder(
  fixture: MemoryEvalFixture,
  options: MemoryEvalRunOptions,
): ResolvedMemoryEvalEmbedder {
  if (options.embedder) {
    return {
      mode: options.embedderMode ?? 'live',
      providerId: options.configuredProviderId ?? options.embedder.providerId,
      model: options.embeddingModel ?? EVAL_EMBEDDING_MODEL,
      vocabularySize: options.vocabularySize,
      embedder: options.embedder,
    }
  }

  const embedder = createLexicalEvalEmbedder(fixture)
  return {
    mode: 'lexical',
    providerId: embedder.providerId,
    model: EVAL_EMBEDDING_MODEL,
    vocabularySize: embedder.vocabularySize,
    embedder,
  }
}

function buildVocabulary(fixture: MemoryEvalFixture): string[] {
  const tokens = new Set<string>(['__empty__'])

  for (const memory of fixture.memories) {
    for (const token of tokenizeMemoryText(memory.content)) {
      tokens.add(token)
    }
  }
  for (const document of fixture.documents) {
    for (const value of [document.title, document.content, document.path, ...document.tags]) {
      if (!value) continue
      for (const token of tokenizeMemoryText(value)) {
        tokens.add(token)
      }
    }
  }
  for (const scenario of fixture.retrieval) {
    for (const token of tokenizeMemoryText(scenario.query)) {
      tokens.add(token)
    }
  }
  for (const scenario of fixture.documentRetrieval) {
    for (const value of [
      scenario.query,
      scenario.documentId,
      ...scenario.expectedCitationIncludes,
      ...scenario.expectedSnippetIncludes,
    ]) {
      if (!value) continue
      for (const token of tokenizeMemoryText(value)) {
        tokens.add(token)
      }
    }
  }

  return Array.from(tokens).sort((left, right) => left.localeCompare(right))
}

function validateFixtureReferences(fixture: MemoryEvalFixture): void {
  const ids = new Set(fixture.memories.map((memory) => memory.id))
  const documentIds = new Set(fixture.documents.map((document) => document.id))
  for (const scenario of fixture.retrieval) {
    for (const id of [...scenario.expectedIds, ...scenario.forbiddenIds]) {
      if (!ids.has(id)) {
        throw new Error(`Retrieval case references unknown memory id: ${id}`)
      }
    }
  }
  for (const scenario of fixture.documentRetrieval) {
    if (scenario.documentId && !documentIds.has(scenario.documentId)) {
      throw new Error(
        `Document retrieval case references unknown document id: ${scenario.documentId}`,
      )
    }
    for (const id of [...scenario.expectedDocumentIds, ...scenario.forbiddenDocumentIds]) {
      if (!documentIds.has(id)) {
        throw new Error(`Document retrieval case references unknown document id: ${id}`)
      }
    }
  }
  for (const scenario of fixture.dedup) {
    if (!ids.has(scenario.primaryId)) {
      throw new Error(`Dedup case references unknown primary memory id: ${scenario.primaryId}`)
    }
    for (const id of [...scenario.expectedDuplicateIds, ...scenario.forbiddenIds]) {
      if (!ids.has(id)) {
        throw new Error(`Dedup case references unknown memory id: ${id}`)
      }
    }
  }
}

function intersectIds(left: string[], right: string[]): string[] {
  const rightSet = new Set(right)
  return left.filter((id) => rightSet.has(id))
}

function subtractIds(left: string[], right: string[]): string[] {
  const rightSet = new Set(right)
  return left.filter((id) => !rightSet.has(id))
}

function uniqueIds(values: string[]): string[] {
  const seen = new Set<string>()
  const unique: string[] = []
  for (const value of values) {
    if (seen.has(value)) continue
    seen.add(value)
    unique.push(value)
  }
  return unique
}

function collectTextExpectationMatches(haystacks: string[], expectations: string[]): string[] {
  return expectations.filter((expectation) => {
    const expected = expectation.trim().toLowerCase()
    if (!expected) return false
    return haystacks.some((value) => value.toLowerCase().includes(expected))
  })
}

function evaluateCoverage<T>(
  expectations: string[],
  matches: string[],
  results: T[],
  readField: (result: T) => string | undefined,
): number {
  if (expectations.length > 0) {
    return ratio(matches.length, expectations.length, 1)
  }
  return results.every((result) => Boolean(readField(result)?.trim())) ? 1 : 0
}

function ratio(numerator: number, denominator: number, fallback: number): number {
  if (denominator === 0) return fallback
  return numerator / denominator
}

function average(values: number[]): number {
  if (values.length === 0) return 1
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function formatRatio(value: number): string {
  return `${Math.round(value * 100)}%`
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  await writeFile(path, JSON.stringify(value, null, 2) + '\n', 'utf-8')
}

function readFlagValue(args: string[], flag: string): string | undefined {
  const index = args.indexOf(flag)
  if (index < 0) return undefined
  return args[index + 1]
}

function readPositionalArg(args: string[]): string | undefined {
  const valueFlags = new Set(['--output', '--config', '--data-dir', '--provider', '--model'])
  for (let index = 0; index < args.length; index++) {
    const arg = args[index]
    if (arg === '--json' || arg === '--live') continue
    if (valueFlags.has(arg)) {
      index++
      continue
    }
    if (!arg.startsWith('--')) {
      return arg
    }
  }
  return undefined
}

function isEntrypoint(): boolean {
  const entry = process.argv[1]
  if (!entry) {
    return false
  }
  return import.meta.url === pathToFileURL(entry).href
}

export const __testables = {
  average,
  buildVocabulary,
  collectTextExpectationMatches,
  createLexicalEvalEmbedder,
  evaluateCoverage,
  formatRatio,
  intersectIds,
  isEntrypoint,
  ratio,
  readFlagValue,
  readPositionalArg,
  resolveMemoryEvalEmbedder,
  subtractIds,
  summarizeMemoryEval,
  uniqueIds,
}

if (isEntrypoint()) {
  main(process.argv.slice(2)).catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
