import { randomUUID } from 'node:crypto'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, isAbsolute, join, relative, resolve } from 'node:path'
import type { HealthExportSnapshot } from '../server/health-support.js'
import { sepilotdHome } from '../storage/home.js'
import type {
  ObservabilityExportBundle,
  ObservabilityExportResult,
  ObservabilityRange,
} from './events.js'
import {
  TRACE_REDACTION_RULES,
  createTraceRedactionContext,
  redactSecretKeys,
  type TraceRedactionContext,
} from './trace-redaction.js'

export interface ObservabilitySupportBundleInput {
  range: ObservabilityRange
  exportResult: ObservabilityExportResult
  health?: HealthExportSnapshot
  persist?: boolean
  outputDir?: string
  generatedAt?: string
  env?: NodeJS.ProcessEnv
  cwd?: string
  stateDir?: string
}

export interface ObservabilitySupportBundleFile {
  path: string
  mediaType: string
  bytes: number
}

export interface ObservabilitySupportBundleRedaction {
  level: 'support'
  appliedAt: string
  rules: string[]
  pathAliases: Record<string, string>
}

export interface ObservabilitySupportBundleManifest {
  schemaVersion: 1
  kind: 'sepilotd-observability-support-bundle'
  generatedAt: string
  range: ObservabilityRange
  counts: {
    events: number
    crashes: number
    feedback: number
  }
  healthIncluded: boolean
  redaction: ObservabilitySupportBundleRedaction
  contents: ObservabilitySupportBundleFile[]
}

export interface ObservabilitySupportBundleResult {
  path?: string
  fileCount: number
  totalBytes: number
  eventCount: number
  crashCount: number
  feedbackCount: number
  files: ObservabilitySupportBundleFile[]
  manifest: ObservabilitySupportBundleManifest
  redaction: ObservabilitySupportBundleRedaction
}

interface BundleFileDraft {
  path: string
  mediaType: string
  content: string
}

function byteLength(content: string): number {
  return Buffer.byteLength(content, 'utf8')
}

function assertSafeBundleRelativePath(pathName: string): string {
  const normalized = pathName.replaceAll('\\', '/')
  if (
    !normalized
    || normalized.startsWith('/')
    || normalized.split('/').some((part) => !part || part === '.' || part === '..')
  ) {
    throw new Error(`Invalid support bundle file path: ${pathName}`)
  }
  return normalized
}

function resolveBundleFilePath(outputDir: string, pathName: string): string {
  const base = resolve(outputDir)
  const resolved = resolve(base, assertSafeBundleRelativePath(pathName))
  const rel = relative(base, resolved)
  if (!rel || rel.startsWith('..') || isAbsolute(rel)) {
    throw new Error(`Support bundle path escaped output directory: ${pathName}`)
  }
  return resolved
}

function fileSummary(file: BundleFileDraft): ObservabilitySupportBundleFile {
  return {
    path: file.path,
    mediaType: file.mediaType,
    bytes: byteLength(file.content),
  }
}

function jsonFile(pathName: string, value: unknown): BundleFileDraft {
  return {
    path: assertSafeBundleRelativePath(pathName),
    mediaType: 'application/json',
    content: `${JSON.stringify(value, null, 2)}\n`,
  }
}

function jsonlFile(pathName: string, values: readonly unknown[]): BundleFileDraft {
  return {
    path: assertSafeBundleRelativePath(pathName),
    mediaType: 'application/x-ndjson',
    content: values.map((value) => JSON.stringify(value)).join('\n') + (values.length ? '\n' : ''),
  }
}

function textFile(pathName: string, content: string): BundleFileDraft {
  return {
    path: assertSafeBundleRelativePath(pathName),
    mediaType: 'text/plain; charset=utf-8',
    content: content.endsWith('\n') ? content : `${content}\n`,
  }
}

function writeBundleDirectory(outputDir: string, files: readonly BundleFileDraft[]): void {
  mkdirSync(dirname(outputDir), { recursive: true, mode: 0o700 })
  mkdirSync(outputDir, { mode: 0o700 })
  for (const file of files) {
    const filePath = resolveBundleFilePath(outputDir, file.path)
    mkdirSync(dirname(filePath), { recursive: true, mode: 0o700 })
    writeFileSync(filePath, file.content, { encoding: 'utf8', mode: 0o600, flag: 'wx' })
  }
}

function defaultOutputDir(generatedAt: string): string {
  const stamp = generatedAt.replace(/[^0-9A-Za-zTZ-]/g, '-')
  return join(
    sepilotdHome(),
    'observability',
    'support-bundles',
    `support-${stamp}-${randomUUID().slice(0, 8)}`,
  )
}

function redactionMetadata(
  generatedAt: string,
  context: TraceRedactionContext,
): ObservabilitySupportBundleRedaction {
  return {
    level: 'support',
    appliedAt: generatedAt,
    rules: TRACE_REDACTION_RULES,
    pathAliases: context.pathAliases,
  }
}

function redactExportBundle(
  bundle: ObservabilityExportBundle,
  context: TraceRedactionContext,
): ObservabilityExportBundle {
  return redactSecretKeys(bundle, context) as ObservabilityExportBundle
}

export function buildObservabilitySupportBundle(
  input: ObservabilitySupportBundleInput,
): ObservabilitySupportBundleResult {
  const generatedAt = input.generatedAt ?? new Date().toISOString()
  const context = createTraceRedactionContext(input)
  const redaction = redactionMetadata(generatedAt, context)
  const redacted = redactExportBundle(input.exportResult.bundle, context)
  const health = input.health
    ? redactSecretKeys(input.health, context) as HealthExportSnapshot
    : undefined
  const dataFiles: BundleFileDraft[] = [
    textFile(
      'README.txt',
      [
        'sepilotd diagnostic support bundle',
        '',
        'This bundle is generated from local observability data.',
        'It applies export-time redaction in addition to ingest-time sanitization.',
      ].join('\n'),
    ),
    jsonFile('observability/snapshot.json', redacted.snapshot),
    jsonlFile('observability/events.jsonl', redacted.events),
    jsonlFile('observability/crashes.jsonl', redacted.crashes),
    jsonlFile('observability/feedback.jsonl', redacted.feedback),
  ]
  if (health) {
    dataFiles.push(jsonFile('system/health.json', health))
  }

  const contents = dataFiles.map(fileSummary)
  const manifest: ObservabilitySupportBundleManifest = {
    schemaVersion: 1,
    kind: 'sepilotd-observability-support-bundle',
    generatedAt,
    range: input.range,
    counts: {
      events: input.exportResult.eventCount,
      crashes: input.exportResult.crashCount,
      feedback: input.exportResult.feedbackCount,
    },
    healthIncluded: Boolean(health),
    redaction,
    contents,
  }
  const files = [jsonFile('manifest.json', manifest), ...dataFiles]
  const summaries = files.map(fileSummary)
  const totalBytes = summaries.reduce((sum, file) => sum + file.bytes, 0)
  const path = input.persist === false
    ? undefined
    : input.outputDir ?? defaultOutputDir(generatedAt)
  if (path) {
    writeBundleDirectory(path, files)
  }

  return {
    path,
    fileCount: summaries.length,
    totalBytes,
    eventCount: input.exportResult.eventCount,
    crashCount: input.exportResult.crashCount,
    feedbackCount: input.exportResult.feedbackCount,
    files: summaries,
    manifest,
    redaction,
  }
}
