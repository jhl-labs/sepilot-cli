import type {
  A2UIChart,
  A2UIComponent,
  A2UIForm,
  A2UIImage,
  A2UIPayload,
  A2UIProgress,
  A2UITable,
} from './types.js'
import { isSafeA2UIImageSrc } from './types.js'

const A2UI_BLOCK_PATTERN = /```a2ui\s*\n([\s\S]*?)```/g
const MAX_A2UI_BLOCK_BYTES = 256_000
const MAX_COMPONENTS = 50
const MAX_TABLE_ROWS = 500
const MAX_TABLE_COLUMNS = 50
const MAX_LIST_ITEMS = 200
const MAX_FORM_FIELDS = 50
const MAX_PROGRESS_STEPS = 100
const MAX_TEXT_LENGTH = 50_000
const MAX_CODE_LENGTH = 200_000

export interface A2UIBlockParseIssue {
  index: number
  message: string
}

export interface ExtractA2UIBlocksResult {
  text: string
  canvases: A2UIPayload[]
  invalidBlocks: A2UIBlockParseIssue[]
}

export function extractA2UI(content: string): {
  text: string
  canvas: A2UIPayload | null
} {
  const result = extractA2UIBlocks(content)
  return {
    text: result.text,
    canvas: result.canvases[0] ?? null,
  }
}

export function extractA2UIBlocks(content: string): ExtractA2UIBlocksResult {
  const canvases: A2UIPayload[] = []
  const invalidBlocks: A2UIBlockParseIssue[] = []
  let text = ''
  let cursor = 0
  let blockIndex = 0

  for (const match of content.matchAll(A2UI_BLOCK_PATTERN)) {
    const fullMatch = match[0] ?? ''
    const body = match[1] ?? ''
    const index = match.index ?? 0
    text += content.slice(cursor, index)
    cursor = index + fullMatch.length

    const payload = parseA2UIPayloadJson(body)
    if (payload) {
      canvases.push(payload)
    } else {
      invalidBlocks.push({ index: blockIndex, message: 'Invalid A2UI payload' })
      text += fullMatch
    }
    blockIndex += 1
  }

  if (cursor === 0) return { text: content, canvases, invalidBlocks }
  text += content.slice(cursor)
  return { text: text.trim(), canvases, invalidBlocks }
}

export function parseA2UIPayloadJson(json: string): A2UIPayload | null {
  if (json.length > MAX_A2UI_BLOCK_BYTES) return null
  try {
    return parseA2UIPayload(JSON.parse(json))
  } catch {
    return null
  }
}

export function parseA2UIPayload(value: unknown): A2UIPayload | null {
  if (!isRecord(value)) return null
  if (value.kind != null && value.kind !== 'a2ui') return null
  if (value.version != null && value.version !== 1) return null
  if (!Array.isArray(value.components) || value.components.length > MAX_COMPONENTS) return null

  const components: A2UIComponent[] = []
  for (const component of value.components) {
    const parsed = parseA2UIComponent(component)
    if (!parsed) return null
    components.push(parsed)
  }

  const payload: A2UIPayload = { components }
  if (value.kind === 'a2ui') payload.kind = 'a2ui'
  if (value.version === 1) payload.version = 1
  if (typeof value.interactive === 'boolean') payload.interactive = value.interactive
  if (isMetadata(value.metadata)) payload.metadata = value.metadata
  return payload
}

function parseA2UIComponent(value: unknown): A2UIComponent | null {
  if (!isRecord(value) || typeof value.type !== 'string') return null
  switch (value.type) {
    case 'text':
      return parseText(value)
    case 'table':
      return parseTable(value)
    case 'chart':
      return parseChart(value)
    case 'form':
      return parseForm(value)
    case 'code':
      return parseCode(value)
    case 'image':
      return parseImage(value)
    case 'list':
      return parseList(value)
    case 'progress':
      return parseProgress(value)
    default:
      return null
  }
}

function parseText(value: Record<string, unknown>): A2UIComponent | null {
  if (!isBoundedString(value.content, MAX_TEXT_LENGTH)) return null
  if (
    value.format != null &&
    value.format !== 'plain' &&
    value.format !== 'markdown' &&
    value.format !== 'html'
  ) {
    return null
  }
  return {
    type: 'text',
    content: value.content,
    ...(typeof value.format === 'string' ? { format: value.format } : {}),
  }
}

function parseTable(value: Record<string, unknown>): A2UITable | null {
  if (!isStringArray(value.headers, MAX_TABLE_COLUMNS, 500)) return null
  if (!Array.isArray(value.rows) || value.rows.length > MAX_TABLE_ROWS) return null
  const rows = value.rows.map((row) =>
    isStringArray(row, MAX_TABLE_COLUMNS, 5_000) ? row : null,
  )
  if (rows.some((row) => row == null)) return null
  return {
    type: 'table',
    ...(isBoundedString(value.title, 500) ? { title: value.title } : {}),
    headers: value.headers,
    rows: rows as string[][],
  }
}

function parseChart(value: Record<string, unknown>): A2UIChart | null {
  if (
    value.chartType !== 'bar' &&
    value.chartType !== 'line' &&
    value.chartType !== 'pie' &&
    value.chartType !== 'scatter'
  ) {
    return null
  }
  if (!isRecord(value.data)) return null
  if (!isStringArray(value.data.labels, 500, 500)) return null
  if (!Array.isArray(value.data.datasets) || value.data.datasets.length > 20) return null
  const datasets = value.data.datasets.map((dataset) => {
    if (!isRecord(dataset) || !isBoundedString(dataset.label, 500)) return null
    if (!Array.isArray(dataset.values) || dataset.values.length > 500) return null
    const values = dataset.values.map((item) => Number(item))
    if (values.some((item) => !Number.isFinite(item))) return null
    return { label: dataset.label, values }
  })
  if (datasets.some((dataset) => dataset == null)) return null
  return {
    type: 'chart',
    ...(isBoundedString(value.title, 500) ? { title: value.title } : {}),
    chartType: value.chartType,
    data: {
      labels: value.data.labels,
      datasets: datasets as Array<{ label: string; values: number[] }>,
    },
  }
}

function parseForm(value: Record<string, unknown>): A2UIForm | null {
  if (!Array.isArray(value.fields) || value.fields.length > MAX_FORM_FIELDS) return null
  const fields = value.fields.map((field) => {
    if (!isRecord(field)) return null
    if (!isBoundedString(field.name, 120) || !isBoundedString(field.label, 500)) return null
    if (
      field.type !== 'text' &&
      field.type !== 'number' &&
      field.type !== 'select' &&
      field.type !== 'checkbox' &&
      field.type !== 'textarea'
    ) {
      return null
    }
    if (field.options != null && !isStringArray(field.options, 100, 500)) return null
    return {
      name: field.name,
      label: field.label,
      type: field.type,
      ...(Array.isArray(field.options) ? { options: field.options } : {}),
      ...(typeof field.required === 'boolean' ? { required: field.required } : {}),
      ...(isBoundedString(field.defaultValue, 5_000)
        ? { defaultValue: field.defaultValue }
        : {}),
    }
  })
  if (fields.some((field) => field == null)) return null
  return {
    type: 'form',
    ...(isBoundedString(value.title, 500) ? { title: value.title } : {}),
    fields: fields as A2UIForm['fields'],
    ...(isBoundedString(value.submitLabel, 120) ? { submitLabel: value.submitLabel } : {}),
  }
}

function parseCode(value: Record<string, unknown>): A2UIComponent | null {
  if (!isBoundedString(value.language, 80)) return null
  if (!isBoundedString(value.code, MAX_CODE_LENGTH)) return null
  return {
    type: 'code',
    language: value.language,
    code: value.code,
    ...(isBoundedString(value.title, 500) ? { title: value.title } : {}),
  }
}

function parseImage(value: Record<string, unknown>): A2UIImage | null {
  if (!isBoundedString(value.src, 50_000) || !isSafeA2UIImageSrc(value.src)) return null
  return {
    type: 'image',
    src: value.src,
    ...(isBoundedString(value.alt, 1_000) ? { alt: value.alt } : {}),
    ...(isPositiveInteger(value.width) ? { width: value.width } : {}),
    ...(isPositiveInteger(value.height) ? { height: value.height } : {}),
    ...(isBoundedString(value.mime, 120) ? { mime: value.mime } : {}),
  }
}

function parseList(value: Record<string, unknown>): A2UIComponent | null {
  if (!Array.isArray(value.items) || value.items.length > MAX_LIST_ITEMS) return null
  const items: Array<{ text: string; description?: string; icon?: string } | null> =
    value.items.map((item) => {
      if (!isRecord(item) || !isBoundedString(item.text, 1_000)) return null
      return {
        text: item.text,
        ...(isBoundedString(item.description, 2_000)
          ? { description: item.description }
          : {}),
        ...(isBoundedString(item.icon, 80) ? { icon: item.icon } : {}),
      }
    })
  if (items.some((item) => item == null)) return null
  return {
    type: 'list',
    ...(isBoundedString(value.title, 500) ? { title: value.title } : {}),
    items: items as Array<{ text: string; description?: string; icon?: string }>,
    ...(typeof value.ordered === 'boolean' ? { ordered: value.ordered } : {}),
  }
}

function parseProgress(value: Record<string, unknown>): A2UIProgress | null {
  if (!Array.isArray(value.steps) || value.steps.length > MAX_PROGRESS_STEPS) return null
  const steps = value.steps.map((step) => {
    if (!isRecord(step) || !isBoundedString(step.label, 500)) return null
    if (step.status !== 'done' && step.status !== 'active' && step.status !== 'pending') {
      return null
    }
    return { label: step.label, status: step.status }
  })
  if (steps.some((step) => step == null)) return null
  return {
    type: 'progress',
    ...(isBoundedString(value.title, 500) ? { title: value.title } : {}),
    steps: steps as A2UIProgress['steps'],
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isBoundedString(value: unknown, maxLength: number): value is string {
  return typeof value === 'string' && value.length <= maxLength
}

function isStringArray(
  value: unknown,
  maxItems: number,
  maxItemLength: number,
): value is string[] {
  return (
    Array.isArray(value) &&
    value.length <= maxItems &&
    value.every((item) => isBoundedString(item, maxItemLength))
  )
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0
}

function isMetadata(value: unknown): value is A2UIPayload['metadata'] {
  if (!isRecord(value)) return false
  return Object.values(value).every(
    (item) =>
      item == null ||
      typeof item === 'string' ||
      typeof item === 'number' ||
      typeof item === 'boolean',
  )
}
