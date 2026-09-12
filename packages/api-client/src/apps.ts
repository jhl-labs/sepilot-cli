export const SEPILOT_APPS_FRAMEWORK_VERSION = 1 as const
export const SEPILOT_APPS_SCHEMA_VERSION = 1 as const
export const SEPILOT_APPS_ROOT_DIR = 'apps'
export const SEPILOT_APP_MANIFEST_FILE = 'manifest.json'
export const SEPILOT_APP_ENTRY_FILE = 'index.html'
export const SEPILOT_APP_DATA_FILE = 'data.json'
export const SEPILOT_APP_DEFAULT_READ_LIMIT = 12

export const SEPILOT_APP_SENSITIVE_KEY_PATTERN =
  /(?:password|passphrase|secret|token|api[-_]?key|private[-_]?key|credential|cipher|salt|iv|vault)/iu

export type SepilotAppDataPrimitive = string | number | boolean | null

export interface SepilotAppDataPolicy {
  /**
   * Whether daemon-side agent tools may read this app's data. Sensitive apps
   * such as vaults should set this to false unless they expose redacted views.
   */
  aiReadable?: boolean
  /**
   * Whether daemon-side agent tools may mutate this app's data.json. Sensitive
   * apps should keep this disabled and expose explicit app-owned flows instead.
   */
  aiWritable?: boolean
  /**
   * Whether sandboxed apps may read this app through the inter-app broker.
   */
  interAppReadable?: boolean
  sensitivePaths?: string[]
  redactedKeys?: string[]
  encrypted?: boolean
  backup?: 'included' | 'excluded'
  retentionDays?: number
}

export interface SepilotAppCapabilities {
  fields?: boolean
  collections?: boolean
  timeSeries?: boolean
  semanticSearch?: boolean
  scheduler?: boolean
  notifications?: boolean
  backup?: boolean
  encryption?: boolean
  interAppRead?: boolean
  agentTools?: boolean
  workflow?: boolean
  customComponents?: boolean
}

export interface SepilotAppHostRequestPermission {
  op: string
  reason?: string
}

export interface SepilotAppFieldSchema {
  type?: 'string' | 'number' | 'boolean' | 'date' | 'datetime' | 'json'
  label?: string
  description?: string
  required?: boolean
}

export interface SepilotAppCollectionSchema {
  label?: string
  itemSchema?: Record<string, SepilotAppFieldSchema>
}

export interface SepilotAppTimeSeriesSchema {
  label?: string
  valueType?: 'number' | 'string' | 'json'
  dimensions?: string[]
}

export interface SepilotAppDataSchema {
  fields?: Record<string, SepilotAppFieldSchema>
  collections?: Record<string, SepilotAppCollectionSchema>
  timeSeries?: Record<string, SepilotAppTimeSeriesSchema>
}

export type SepilotAppMutation =
  | {
      op: 'fields.set'
      field: string
      value: unknown
    }
  | {
      op: 'collection.upsert'
      collection: string
      item: Record<string, unknown>
    }
  | {
      op: 'collection.remove'
      collection: string
      id: string
    }
  | {
      op: 'timeSeries.append'
      series: string
      point: Record<string, unknown>
    }

export interface SepilotAppMutationIssue {
  severity: 'error' | 'warning'
  code: string
  message: string
  mutationIndex?: number
}

export interface SepilotAppMutationApplyResult {
  ok: boolean
  data: Record<string, unknown>
  applied: string[]
  issues: SepilotAppMutationIssue[]
}

export interface SepilotAppRedactionOptions {
  policy?: SepilotAppDataPolicy
  maxDepth?: number
  maxArrayItems?: number
  maxStringLength?: number
}

export interface SepilotReadableAppSummary {
  id: string
  title: string
  kind: string
  summary?: string
  category?: string
  version?: number
  updatedAt?: number
  capabilities?: SepilotAppCapabilities
}

export interface SepilotReadableAppRecord extends SepilotReadableAppSummary {
  data: unknown
  dataSchema?: SepilotAppDataSchema
}

export interface SepilotAppSearchMatch {
  path: string
  label?: string
  text: string
  score: number
}

export interface SepilotAppSearchScore {
  score: number
  matches: SepilotAppSearchMatch[]
}

export type SepilotReadableAppSearchResult = SepilotReadableAppRecord & SepilotAppSearchScore

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function cloneRecord(value: Record<string, unknown>): Record<string, unknown> {
  try {
    return JSON.parse(JSON.stringify(value)) as Record<string, unknown>
  } catch {
    return { ...value }
  }
}

function safeRecord(value: unknown): Record<string, unknown> {
  return isRecord(value) ? { ...value } : {}
}

function safeArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value.slice() : []
}

function isSafeAppDataKey(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    /^[\p{L}\p{N}_.-]{1,80}$/u.test(value) &&
    value !== '__proto__' &&
    value !== 'constructor' &&
    value !== 'prototype'
  )
}

function schemaHasEntries(value: unknown): boolean {
  return isRecord(value) && Object.keys(value).length > 0
}

function validateAppFieldValue(
  value: unknown,
  schema: SepilotAppFieldSchema | undefined,
  label: string,
): string | null {
  if (!schema) return null
  if (schema.required && (value == null || value === '')) {
    return `${label} is required.`
  }
  if (value == null || value === '') return null
  if (schema.type === 'string' && typeof value !== 'string') return `${label} must be a string.`
  if (schema.type === 'number' && typeof value !== 'number') return `${label} must be a number.`
  if (schema.type === 'boolean' && typeof value !== 'boolean') {
    return `${label} must be a boolean.`
  }
  if (schema.type === 'date' || schema.type === 'datetime') {
    if (typeof value !== 'string' || Number.isNaN(Date.parse(value))) {
      return `${label} must be a valid ${schema.type}.`
    }
  }
  return null
}

function pushAppMutationIssue(
  issues: SepilotAppMutationIssue[],
  severity: SepilotAppMutationIssue['severity'],
  code: string,
  message: string,
  mutationIndex: number,
): void {
  issues.push({ severity, code, message, mutationIndex })
}

export function applySepilotAppMutations(input: {
  data: Record<string, unknown>
  dataSchema?: SepilotAppDataSchema
  mutations: SepilotAppMutation[]
}): SepilotAppMutationApplyResult {
  const data = cloneRecord(input.data)
  const schema = input.dataSchema
  const issues: SepilotAppMutationIssue[] = []
  const applied: string[] = []
  const fieldSchemas = schema?.fields ?? {}
  const collectionSchemas = schema?.collections ?? {}
  const timeSeriesSchemas = schema?.timeSeries ?? {}
  const hasFieldSchemas = schemaHasEntries(fieldSchemas)
  const hasCollectionSchemas = schemaHasEntries(collectionSchemas)
  const hasTimeSeriesSchemas = schemaHasEntries(timeSeriesSchemas)

  input.mutations.forEach((mutation, index) => {
    if (!isRecord(mutation) || typeof mutation.op !== 'string') {
      pushAppMutationIssue(
        issues,
        'error',
        'invalid_mutation',
        'Mutation must include an op.',
        index,
      )
      return
    }

    if (mutation.op === 'fields.set') {
      if (!isSafeAppDataKey(mutation.field)) {
        pushAppMutationIssue(issues, 'error', 'invalid_field', 'Field name is invalid.', index)
        return
      }
      const schemaField = fieldSchemas[mutation.field]
      if (hasFieldSchemas && !schemaField) {
        pushAppMutationIssue(
          issues,
          'warning',
          'unknown_field',
          `Field is not declared in dataSchema: ${mutation.field}`,
          index,
        )
      }
      const validation = validateAppFieldValue(
        mutation.value,
        schemaField,
        schemaField?.label ?? mutation.field,
      )
      if (validation) {
        pushAppMutationIssue(issues, 'error', 'invalid_field_value', validation, index)
        return
      }
      const fields = safeRecord(data.fields)
      fields[mutation.field] = mutation.value
      data.fields = fields
      applied.push(`fields.set:${mutation.field}`)
      return
    }

    if (mutation.op === 'collection.upsert') {
      if (!isSafeAppDataKey(mutation.collection)) {
        pushAppMutationIssue(
          issues,
          'error',
          'invalid_collection',
          'Collection name is invalid.',
          index,
        )
        return
      }
      if (!isRecord(mutation.item)) {
        pushAppMutationIssue(
          issues,
          'error',
          'invalid_item',
          'Collection item must be an object.',
          index,
        )
        return
      }
      const id = mutation.item.id
      if (!isSafeAppDataKey(id)) {
        pushAppMutationIssue(
          issues,
          'error',
          'invalid_item_id',
          'Collection item id is required.',
          index,
        )
        return
      }
      const collectionSchema = collectionSchemas[mutation.collection]
      if (hasCollectionSchemas && !collectionSchema) {
        pushAppMutationIssue(
          issues,
          'warning',
          'unknown_collection',
          `Collection is not declared in dataSchema: ${mutation.collection}`,
          index,
        )
      }
      const itemSchema = collectionSchema?.itemSchema ?? {}
      for (const [field, fieldSchema] of Object.entries(itemSchema)) {
        const validation = validateAppFieldValue(
          mutation.item[field],
          fieldSchema,
          fieldSchema.label ?? `${mutation.collection}.${field}`,
        )
        if (validation) {
          pushAppMutationIssue(issues, 'error', 'invalid_item_value', validation, index)
          return
        }
      }
      const collections = safeRecord(data.collections)
      const rows = safeArray(collections[mutation.collection]).filter(isRecord)
      const nextItem = { ...mutation.item, id }
      const existingIndex = rows.findIndex((row) => row.id === id)
      collections[mutation.collection] =
        existingIndex >= 0
          ? rows.map((row, rowIndex) => (rowIndex === existingIndex ? nextItem : row))
          : [...rows, nextItem]
      data.collections = collections
      applied.push(`collection.upsert:${mutation.collection}.${id}`)
      return
    }

    if (mutation.op === 'collection.remove') {
      if (!isSafeAppDataKey(mutation.collection) || !isSafeAppDataKey(mutation.id)) {
        pushAppMutationIssue(
          issues,
          'error',
          'invalid_collection_remove',
          'Collection and id are required.',
          index,
        )
        return
      }
      const collections = safeRecord(data.collections)
      const rows = safeArray(collections[mutation.collection]).filter(isRecord)
      collections[mutation.collection] = rows.filter((row) => row.id !== mutation.id)
      data.collections = collections
      applied.push(`collection.remove:${mutation.collection}.${mutation.id}`)
      return
    }

    if (mutation.op === 'timeSeries.append') {
      if (!isSafeAppDataKey(mutation.series)) {
        pushAppMutationIssue(
          issues,
          'error',
          'invalid_series',
          'Time series name is invalid.',
          index,
        )
        return
      }
      if (!isRecord(mutation.point)) {
        pushAppMutationIssue(
          issues,
          'error',
          'invalid_point',
          'Time series point must be an object.',
          index,
        )
        return
      }
      if (hasTimeSeriesSchemas && !timeSeriesSchemas[mutation.series]) {
        pushAppMutationIssue(
          issues,
          'warning',
          'unknown_series',
          `Time series is not declared in dataSchema: ${mutation.series}`,
          index,
        )
      }
      const timeSeries = safeRecord(data.timeSeries)
      const points = safeArray(timeSeries[mutation.series]).filter(isRecord)
      const point = {
        ...mutation.point,
        ts: typeof mutation.point.ts === 'number' ? mutation.point.ts : Date.now(),
      }
      timeSeries[mutation.series] = [...points, point].slice(-5000)
      data.timeSeries = timeSeries
      applied.push(`timeSeries.append:${mutation.series}`)
      return
    }

    pushAppMutationIssue(
      issues,
      'error',
      'unsupported_mutation',
      `Unsupported mutation op: ${(mutation as { op?: string }).op ?? 'unknown'}`,
      index,
    )
  })

  return {
    ok: !issues.some((issue) => issue.severity === 'error'),
    data,
    applied,
    issues,
  }
}

function sensitiveKeySet(policy: SepilotAppDataPolicy | undefined): Set<string> {
  return new Set((policy?.redactedKeys ?? []).map((key) => key.toLowerCase()))
}

function sensitivePathSet(policy: SepilotAppDataPolicy | undefined): Set<string> {
  return new Set((policy?.sensitivePaths ?? []).filter(Boolean))
}

function pathMatches(path: string, sensitivePath: string): boolean {
  if (!path || !sensitivePath) return false
  if (path === sensitivePath) return true
  if (path.startsWith(`${sensitivePath}.`)) return true
  if (sensitivePath.endsWith('.*')) {
    const prefix = sensitivePath.slice(0, -2)
    return path === prefix || path.startsWith(`${prefix}.`)
  }
  return false
}

function shouldRedactKey(
  key: string,
  path: string,
  policy: SepilotAppDataPolicy | undefined,
): boolean {
  if (sensitiveKeySet(policy).has(key.toLowerCase())) return true
  if (SEPILOT_APP_SENSITIVE_KEY_PATTERN.test(key)) return true
  for (const sensitivePath of sensitivePathSet(policy)) {
    if (pathMatches(path, sensitivePath)) return true
  }
  return false
}

export function isAppDataReadableByAgent(policy?: SepilotAppDataPolicy): boolean {
  return policy?.aiReadable !== false
}

export function isAppDataWritableByAgent(policy?: SepilotAppDataPolicy): boolean {
  return policy?.aiWritable !== false
}

export function isAppDataReadableByApps(policy?: SepilotAppDataPolicy): boolean {
  return policy?.interAppReadable !== false
}

export function redactAppData(value: unknown, options: SepilotAppRedactionOptions = {}): unknown {
  const maxDepth = options.maxDepth ?? 8
  const maxArrayItems = options.maxArrayItems ?? 60
  const maxStringLength = options.maxStringLength ?? 1_200
  const seen = new WeakSet<object>()

  function redactNode(node: unknown, path: string, depth: number): unknown {
    if (node == null || typeof node === 'number' || typeof node === 'boolean') {
      return node as SepilotAppDataPrimitive
    }
    if (typeof node === 'string') {
      return node.length > maxStringLength ? `${node.slice(0, maxStringLength)}...` : node
    }
    if (typeof node !== 'object') {
      return String(node)
    }
    if (depth >= maxDepth) {
      return '[MaxDepth]'
    }
    if (seen.has(node)) {
      return '[Circular]'
    }
    seen.add(node)
    if (Array.isArray(node)) {
      const sliced = node
        .slice(0, maxArrayItems)
        .map((item, index) => redactNode(item, `${path}.${index}`, depth + 1))
      if (node.length > maxArrayItems) {
        sliced.push(`[${node.length - maxArrayItems} more]`)
      }
      return sliced
    }
    const out: Record<string, unknown> = {}
    for (const [key, child] of Object.entries(node)) {
      const childPath = path ? `${path}.${key}` : key
      out[key] = shouldRedactKey(key, childPath, options.policy)
        ? '[REDACTED]'
        : redactNode(child, childPath, depth + 1)
    }
    return out
  }

  return redactNode(value, '', 0)
}

export function sanitizeAppDataForAgent(value: unknown, policy?: SepilotAppDataPolicy): unknown {
  if (!isAppDataReadableByAgent(policy)) {
    return {
      redacted: true,
      reason: 'ai_read_disabled',
    }
  }
  return redactAppData(value, { policy })
}

export function stringifyAppSearchHaystack(value: unknown): string {
  try {
    return JSON.stringify(value).toLowerCase().replace(/\s+/gu, ' ').slice(0, 40_000)
  } catch {
    return String(value).toLowerCase()
  }
}

function appSearchTokens(query: string): string[] {
  const tokens = query
    .toLowerCase()
    .match(/[\p{L}\p{N}_-]+/gu)
    ?.map((token) => token.trim())
    .filter(Boolean)
  return Array.from(new Set(tokens ?? []))
}

function compactSearchText(value: unknown): string {
  if (value == null) return ''
  if (typeof value === 'string') return value.replace(/\s+/gu, ' ').trim()
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return ''
}

function truncateSearchText(value: string, maxLength = 220): string {
  const text = value.replace(/\s+/gu, ' ').trim()
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text
}

function schemaLabelForPath(
  path: string[],
  schema: SepilotAppDataSchema | undefined,
): string | undefined {
  if (path[0] !== 'data') return undefined
  if (path[1] === 'fields' && typeof path[2] === 'string') {
    return schema?.fields?.[path[2]]?.label
  }
  if (path[1] === 'collections' && typeof path[2] === 'string') {
    const collection = schema?.collections?.[path[2]]
    const field = typeof path[4] === 'string' ? collection?.itemSchema?.[path[4]] : undefined
    return [collection?.label, field?.label].filter(Boolean).join(' · ') || undefined
  }
  if (path[1] === 'timeSeries' && typeof path[2] === 'string') {
    return schema?.timeSeries?.[path[2]]?.label
  }
  return undefined
}

function appSearchPathWeight(path: string[]): number {
  if (path[0] === 'title') return 4
  if (path[0] === 'summary') return 3
  if (path[0] === 'category' || path[0] === 'kind') return 2.4
  if (path[0] !== 'data') return 1
  if (path[1] === 'fields') return 2.5
  if (path[1] === 'collections') {
    if (path.length <= 4) return 1.8
    return path[4] === 'title' || path[4] === 'name' ? 2.4 : 1.9
  }
  if (path[1] === 'timeSeries') return 1.4
  return 1
}

function scoreTextAgainstTokens(
  text: string,
  label: string | undefined,
  path: string,
  tokens: string[],
  phrase: string,
): { score: number; tokenHits: Set<string> } {
  const haystack = `${label ?? ''} ${path} ${text}`.toLowerCase()
  const valueText = text.toLowerCase()
  const tokenHits = new Set<string>()
  let score = 0
  if (phrase && valueText.includes(phrase)) score += Math.max(4, tokens.length * 2)
  for (const token of tokens) {
    const valueIndex = valueText.indexOf(token)
    const haystackIndex = haystack.indexOf(token)
    if (haystackIndex < 0) continue
    tokenHits.add(token)
    if (valueIndex >= 0) {
      const boundaryBefore =
        valueIndex === 0 || /[^\p{L}\p{N}_-]/u.test(valueText[valueIndex - 1] ?? '')
      const after = valueIndex + token.length
      const boundaryAfter =
        after >= valueText.length || /[^\p{L}\p{N}_-]/u.test(valueText[after] ?? '')
      score += boundaryBefore && boundaryAfter ? 2.2 : 1.2
    } else {
      score += 0.8
    }
  }
  return { score, tokenHits }
}

interface AppSearchEntry {
  path: string[]
  label?: string
  text: string
  weight: number
}

function collectAppSearchEntries(value: unknown): AppSearchEntry[] {
  const root = isRecord(value) ? value : {}
  const schema = isRecord(root.dataSchema) ? (root.dataSchema as SepilotAppDataSchema) : undefined
  const entries: AppSearchEntry[] = []
  const push = (path: string[], text: string, weight = appSearchPathWeight(path)) => {
    if (!text) return
    entries.push({
      path,
      label: schemaLabelForPath(path, schema),
      text,
      weight,
    })
  }
  for (const key of ['title', 'summary', 'category', 'kind', 'id'] as const) {
    push([key], compactSearchText(root[key]))
  }

  function visit(node: unknown, path: string[], depth: number): void {
    if (depth > 8) return
    const text = compactSearchText(node)
    if (text) {
      push(path, text)
      return
    }
    if (Array.isArray(node)) {
      node.forEach((item, index) => visit(item, [...path, String(index)], depth + 1))
      return
    }
    if (!isRecord(node)) return
    if (path[0] === 'data' && path[1] === 'collections' && path.length === 4) {
      const summary = Object.values(node).map(compactSearchText).filter(Boolean).join(' ')
      push(path, summary, appSearchPathWeight(path))
    }
    for (const [key, child] of Object.entries(node)) {
      visit(child, [...path, key], depth + 1)
    }
  }

  visit(root.data, ['data'], 0)
  return entries
}

export function scoreAppSearchRecord(
  value: unknown,
  query: string,
  options: { maxMatches?: number } = {},
): SepilotAppSearchScore {
  const tokens = appSearchTokens(query)
  if (tokens.length === 0) return { score: 0, matches: [] }
  const phrase = query.toLowerCase().replace(/\s+/gu, ' ').trim()
  const entries = collectAppSearchEntries(value)
  const allTokenHits = new Set<string>()
  const scoredMatches: SepilotAppSearchMatch[] = []
  let score = 0
  for (const entry of entries) {
    const entryPath = entry.path.join('.')
    const scored = scoreTextAgainstTokens(entry.text, entry.label, entryPath, tokens, phrase)
    if (scored.score <= 0) continue
    for (const token of scored.tokenHits) allTokenHits.add(token)
    const weighted = scored.score * entry.weight
    score += weighted
    scoredMatches.push({
      path: entryPath,
      label: entry.label,
      text: truncateSearchText(entry.text),
      score: Number(weighted.toFixed(3)),
    })
  }
  if (!tokens.every((token) => allTokenHits.has(token))) {
    const haystack = stringifyAppSearchHaystack(value)
    if (!tokens.every((token) => haystack.includes(token))) return { score: 0, matches: [] }
    score += 0.1
  }
  scoredMatches.sort(
    (left, right) => right.score - left.score || left.path.localeCompare(right.path),
  )
  return {
    score: Number(score.toFixed(3)),
    matches: scoredMatches.slice(0, Math.max(1, Math.min(10, options.maxMatches ?? 5))),
  }
}

export function rankAppSearchRecords<T extends object>(
  records: T[],
  query: string,
  options: { limit?: number; maxMatches?: number } = {},
): Array<T & SepilotAppSearchScore> {
  const limit = clampAppReadLimit(options.limit, records.length || SEPILOT_APP_DEFAULT_READ_LIMIT)
  return records
    .map((record) => ({
      ...record,
      ...scoreAppSearchRecord(record, query, { maxMatches: options.maxMatches }),
    }))
    .filter((record) => record.score > 0)
    .sort((left, right) => {
      if (right.score !== left.score) return right.score - left.score
      const rightRecord = right as { updatedAt?: unknown }
      const leftRecord = left as { updatedAt?: unknown }
      const rightUpdated = typeof rightRecord.updatedAt === 'number' ? rightRecord.updatedAt : 0
      const leftUpdated = typeof leftRecord.updatedAt === 'number' ? leftRecord.updatedAt : 0
      return rightUpdated - leftUpdated
    })
    .slice(0, limit)
}

export function matchesAppQuery(value: unknown, query: string): boolean {
  return scoreAppSearchRecord(value, query).score > 0
}

export function clampAppReadLimit(
  value: unknown,
  fallback = SEPILOT_APP_DEFAULT_READ_LIMIT,
): number {
  const parsed = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return Math.min(50, Math.max(1, Math.floor(parsed)))
}

export function buildReadableAppSummary(input: {
  id: string
  title: string
  kind: string
  summary?: string
  category?: string
  version?: number
  updatedAt?: number
  capabilities?: SepilotAppCapabilities
}): SepilotReadableAppSummary {
  return {
    id: input.id,
    title: input.title,
    kind: input.kind,
    summary: input.summary,
    category: input.category,
    version: input.version,
    updatedAt: input.updatedAt,
    capabilities: input.capabilities,
  }
}

export function buildReadableAppRecord(input: {
  id: string
  title: string
  kind: string
  summary?: string
  category?: string
  version?: number
  updatedAt?: number
  capabilities?: SepilotAppCapabilities
  dataSchema?: SepilotAppDataSchema
  data: unknown
  dataPolicy?: SepilotAppDataPolicy
}): SepilotReadableAppRecord {
  return {
    ...buildReadableAppSummary(input),
    dataSchema: input.dataSchema,
    data: sanitizeAppDataForAgent(input.data, input.dataPolicy),
  }
}

export function normalizeAppDataPolicy(
  value: unknown,
  fallback: SepilotAppDataPolicy = {},
): SepilotAppDataPolicy {
  if (!isRecord(value)) return { ...fallback }
  return {
    ...fallback,
    aiReadable: typeof value.aiReadable === 'boolean' ? value.aiReadable : fallback.aiReadable,
    aiWritable: typeof value.aiWritable === 'boolean' ? value.aiWritable : fallback.aiWritable,
    interAppReadable:
      typeof value.interAppReadable === 'boolean'
        ? value.interAppReadable
        : fallback.interAppReadable,
    encrypted: typeof value.encrypted === 'boolean' ? value.encrypted : fallback.encrypted,
    backup:
      value.backup === 'excluded' || value.backup === 'included' ? value.backup : fallback.backup,
    retentionDays:
      typeof value.retentionDays === 'number' && Number.isFinite(value.retentionDays)
        ? value.retentionDays
        : fallback.retentionDays,
    sensitivePaths: Array.isArray(value.sensitivePaths)
      ? value.sensitivePaths.filter((entry): entry is string => typeof entry === 'string')
      : fallback.sensitivePaths,
    redactedKeys: Array.isArray(value.redactedKeys)
      ? value.redactedKeys.filter((entry): entry is string => typeof entry === 'string')
      : fallback.redactedKeys,
  }
}
