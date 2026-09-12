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

/**
 * Parse the explicit `/remember <text>` slash command into a memory directive.
 *
 * Only the slash command is parsed here — it is an unambiguous command, like
 * `/approve`. Natural-language "remember X" / "기억해줘 X" / "X 라고 기록해놔" is
 * NOT regex-detected anymore: those messages flow through the normal agent
 * turn, and the agent saves them via the `memory.remember` tool (the system
 * prompt instructs it to do so on any explicit "remember/save/store" request,
 * and memory.remember is auto-approved as a user-memory write). This keeps the
 * channel free of intent-guessing heuristics.
 */
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
    // Do not echo the raw secret/PII back to the caller (it may be logged or
    // surfaced in a UI) — return the redacted form.
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
  // A new session is provenance, not an ownership transfer of a durable fact.
  const tags = hasDurableScope ? suppliedTags.filter((tag) => !tag.toLowerCase().startsWith('scope:session:')) : suppliedTags
  const id = `user-${stableMemoryHash(memoryIdentitySeed(options.evidence?.subject || options.evidence?.reality ? JSON.stringify([content, options.evidence.subject, options.evidence.reality]) : content, tags))}`
  // Plain markdown has no expiry/provenance enforcement. Keep managed facts in
  // the authoritative index so future, temporary and inferred claims cannot
  // silently turn into permanent user instructions in the prompt.
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
      // Never create an untracked file copy after the authoritative store rejects
      // a write (for example scope, evidence, or reset validation).
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
