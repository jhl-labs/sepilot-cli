import { memoryEvidenceSchema } from './evidence.js'
import { createHash } from 'node:crypto'
import type { ISemanticIndex, MemoryEvidence } from '@sepilotd/core'
import { formatPortfolioHoldingsForMemory } from './portfolio-holdings.js'
import { looksSensitive, redactSensitive } from './sensitive.js'

export interface UserMemoryFileStore {
  mergeMemorySectionItems(sectionTitle: string, items: string[]): Promise<number>
}

export interface RememberUserMemoryOptions {
  content: string
  evidence?: MemoryEvidence
  fileMemory?: UserMemoryFileStore
  semanticIndex?: Pick<ISemanticIndex, 'add'>
  sectionTitle?: string
  tags?: string[]
}

export interface RememberUserMemoryResult {
  status: 'saved' | 'partial' | 'empty' | 'sensitive' | 'unavailable' | 'too-long'
  id?: string
  content?: string
  fileMemoryAdded?: number
  semanticMemorySaved?: boolean
  errors?: string[]
}

const DEFAULT_SECTION_TITLE = 'User Memory'
const MAX_MEMORY_CHARS = 1200

export function extractExplicitMemoryDirective(text: string): { content: string } | null {
  const command = text.trim().match(/^\/remember(?:@\S+)?\s+([\s\S]+)$/i)
  if (!command?.[1]) return null
  return normalizeDirective(command[1])
}

export async function rememberUserMemory(
  options: RememberUserMemoryOptions,
): Promise<RememberUserMemoryResult> {
  const content = normalizeMemoryContent(options.content)
  if (!content) {
    return { status: 'empty' }
  }
  if (content.length > MAX_MEMORY_CHARS) return { status: 'too-long' }
  if (options.evidence) memoryEvidenceSchema.parse(options.evidence)
  if (looksSensitive(content)) {
    return { status: 'sensitive', content: redactSensitive(content).redacted }
  }
  if (!options.fileMemory && !options.semanticIndex) {
    return { status: 'unavailable', content }
  }

  const errors: string[] = []
  let fileMemoryAdded = 0
  let semanticMemorySaved = false

  const suppliedTags = uniqueTags(['explicit-memory', ...(options.tags ?? [])])
  const hasDurableScope = suppliedTags.some((tag) => tag.toLowerCase().startsWith('scope:') && !tag.toLowerCase().startsWith('scope:session:'))
  const tags = hasDurableScope ? suppliedTags.filter((tag) => !tag.toLowerCase().startsWith('scope:session:')) : suppliedTags
  const id = `user-${stableMemoryHash(memoryIdentitySeed(options.evidence?.subject || options.evidence?.reality ? JSON.stringify([content, options.evidence.subject, options.evidence.reality]) : content, tags))}`
  const projectToFile = !options.evidence || (options.evidence.status === 'active'
    && options.evidence.origin === 'user' && !options.evidence.subject && !options.evidence.reality && !options.evidence.validFrom && !options.evidence.validUntil)
  if (!projectToFile && !options.semanticIndex) return { status: 'unavailable', errors: ['Evidence-managed memory requires a semantic store.'] }

  if (options.semanticIndex) {
    try {
      await options.semanticIndex.add({ id, content, source: 'user', evidence: options.evidence ?? {
        kind: 'semantic', origin: 'user', observedAt: new Date().toISOString(), status: 'active', sourceIds: [],
      }, tags })
      semanticMemorySaved = true
    } catch (error) {
      return { status: 'unavailable', id, semanticMemorySaved: false, fileMemoryAdded: 0,
        errors: [`semantic:${error instanceof Error ? error.message : String(error)}`] }
    }
  }

  if (options.fileMemory && projectToFile) {
    try {
      fileMemoryAdded = await options.fileMemory.mergeMemorySectionItems(options.sectionTitle ?? DEFAULT_SECTION_TITLE, [content])
    } catch (error) {
      errors.push(`file:${error instanceof Error ? error.message : String(error)}`)
    }
  }

  if (fileMemoryAdded === 0 && !semanticMemorySaved && errors.length > 0) {
    return {
      status: 'unavailable',
      content,
      fileMemoryAdded,
      semanticMemorySaved,
      errors,
    }
  }

  return {
    status: errors.length ? 'partial' : 'saved',
    id: semanticMemorySaved ? id : undefined,
    content,
    fileMemoryAdded,
    semanticMemorySaved,
    errors: errors.length > 0 ? errors : undefined,
  }
}

function normalizeDirective(value: string): { content: string } | null {
  const content = normalizeMemoryContent(value)
  if (!content || /^(이거|이것|이걸|그거|그것|그걸|this|that|it)$/i.test(content)) {
    return null
  }
  const portfolioMemory = formatPortfolioHoldingsForMemory(value)
  return { content: portfolioMemory ? normalizeMemoryContent(portfolioMemory) : content }
}

function normalizeMemoryContent(value: string): string {
  const normalized = value
    .trim()
    .replace(/^["'`“”‘’]+|["'`“”‘’]+$/g, '')
    .replace(/\s+/g, ' ')
    .trim()
  return normalized
}

function stableMemoryHash(content: string): string {
  return createHash('sha256')
    .update(content.trim().replace(/\s+/g, ' ').toLowerCase())
    .digest('hex')
    .slice(0, 32)
}

function memoryIdentitySeed(content: string, tags: string[]): string {
  let scopeTags = tags
    .map((tag) => tag.trim().toLowerCase())
    .filter((tag) => tag.startsWith('scope:'))
    .sort()
  if (scopeTags.some((tag) => !tag.startsWith('scope:session:'))) {
    scopeTags = scopeTags.filter((tag) => !tag.startsWith('scope:session:'))
  }
  return scopeTags.length > 0
    ? `${scopeTags.join('|')}\n${content}`
    : content
}

function uniqueTags(tags: string[]): string[] {
  return Array.from(new Set(
    tags
      .map((tag) => tag.trim())
      .filter(Boolean),
  ))
}
