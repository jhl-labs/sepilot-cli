import { parse as parseToml } from 'smol-toml'
import { parse as parseYaml } from 'yaml'
import type { SkillMetadata } from '@sepilotd/core'

export type SkillFormat = 'toml' | 'yaml'

export interface ParsedSkill {
  metadata: SkillMetadata
  content: string
  format: SkillFormat
}

const TOML_FRONTMATTER = /^\+\+\+\r?\n([\s\S]*?)\r?\n\+\+\+\r?\n([\s\S]*)$/
const YAML_FRONTMATTER = /^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/

function isSafeSlug(value: string): boolean {
  return Boolean(value) && !/[/\\\0]/.test(value) && !/^\.+$/.test(value)
}

export function slugify(input: string): string {
  const slug = input
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/\.+/g, '.')
    .replace(/^[.-]+|[.-]+$/g, '')
  return isSafeSlug(slug) ? slug : 'skill'
}

export function parseSkillMd(raw: string, id: string): ParsedSkill {
  let format: SkillFormat
  let frontmatter: string
  let markdownBody: string

  const tomlMatch = raw.match(TOML_FRONTMATTER)
  const yamlMatch = raw.match(YAML_FRONTMATTER)
  if (tomlMatch) {
    format = 'toml'
    frontmatter = tomlMatch[1]
    markdownBody = tomlMatch[2]
  } else if (yamlMatch) {
    format = 'yaml'
    frontmatter = yamlMatch[1]
    markdownBody = yamlMatch[2]
  } else {
    throw new Error('Invalid SKILL.md format: missing frontmatter delimiters (--- or +++)')
  }

  const parsed =
    format === 'toml'
      ? (parseToml(frontmatter) as Record<string, unknown>)
      : (parseYaml(frontmatter) as Record<string, unknown>)

  const body = markdownBody.trim()
  const metadataMap = isRecord(parsed.metadata) ? parsed.metadata : {}
  const name = asString(parsed.name) ?? (slugify(id) || 'skill')
  const description =
    asString(parsed.description)
    ?? firstMarkdownParagraph(body)
    ?? ''
  const whenToUse = asString(parsed.when_to_use) ?? asString(parsed.whenToUse)
  const combinedDescription = [description, whenToUse].filter(Boolean).join('\n\n')
  const tools = parseToolDeclarations(parsed.tools)
  const allowedTools = parseToolDeclarations(parsed['allowed-tools'])
  const metadata: SkillMetadata = {
    id: slugify(name) || slugify(id) || 'skill',
    name,
    version: asString(parsed.version) ?? asString(metadataMap.version) ?? '0.0.0',
    description: combinedDescription,
    author: asString(parsed.author) ?? asString(metadataMap.author),
    tags: parseStringList(parsed.tags),
    tools: tools.length ? tools : allowedTools,
    autonomy_required: parsed.autonomy_required as SkillMetadata['autonomy_required'],
    created: asString(parsed.created),
    enabled: asBool(parsed.enabled),
  }

  const source = parseSourceRecord(parsed)
  if (source) metadata.source = source
  const provenance = parseProvenanceRecord(parsed.provenance)
  if (provenance) metadata.provenance = provenance
  const riskTier = asString(parsed.risk_tier)
  if (riskTier === 'low' || riskTier === 'medium' || riskTier === 'high') {
    metadata.risk_tier = riskTier
  }
  const permissions = parsePermissionManifest(parsed.permissions)
  if (permissions) metadata.permissions = permissions
  const execution = parseExecutionPolicy(parsed.execution)
  if (execution) metadata.execution = execution

  return { metadata, content: body, format }
}

function parseExecutionPolicy(value: unknown): SkillMetadata['execution'] | undefined {
  if (value === undefined) return undefined
  if (!isRecord(value) || !Array.isArray(value.stages)) {
    throw new Error('Invalid SKILL.md execution policy: stages must be an array')
  }
  const stages = value.stages.map((entry, index) => {
    if (!isRecord(entry)) {
      throw new Error(`Invalid SKILL.md execution policy: stage ${index + 1} must be an object`)
    }
    const id = asString(entry.id)
    const tools = parseStringList(entry.tools)
    if (!id) {
      throw new Error(`Invalid SKILL.md execution policy: stage ${index + 1} requires an id`)
    }
    if (!tools?.length) {
      throw new Error(`Invalid SKILL.md execution policy: stage ${id} requires tools`)
    }
    const maxCallsPerTurn = parseExecutionInteger(entry.maxCallsPerTurn, `${id}.maxCallsPerTurn`)
    const requires = parseExecutionStringList(entry.requires, `${id}.requires`)
    const requiredForCompletion = parseExecutionBoolean(
      entry.requiredForCompletion,
      `${id}.requiredForCompletion`,
    )
    const satisfyOn = parseExecutionSatisfyOn(entry.satisfyOn, `${id}.satisfyOn`)
    return {
      id,
      tools,
      ...(maxCallsPerTurn !== undefined ? { maxCallsPerTurn } : {}),
      ...(requires?.length ? { requires } : {}),
      ...(requiredForCompletion !== undefined ? { requiredForCompletion } : {}),
      ...(satisfyOn !== undefined ? { satisfyOn } : {}),
    }
  })
  const maxCompletionRetries = parseExecutionInteger(
    value.maxCompletionRetries,
    'maxCompletionRetries',
  )
  const argumentBindings = parseExecutionArgumentBindings(value.argumentBindings)
  return {
    stages,
    ...(argumentBindings ? { argumentBindings } : {}),
    ...(maxCompletionRetries !== undefined ? { maxCompletionRetries } : {}),
  }
}

function parseExecutionSatisfyOn(
  value: unknown,
  field: string,
): 'success' | 'executed-outcome' | undefined {
  if (value === undefined) return undefined
  if (value === 'success' || value === 'executed-outcome') return value
  throw new Error(
    `Invalid SKILL.md execution policy: ${field} must be "success" or "executed-outcome"`,
  )
}

function parseExecutionArgumentBindings(
  value: unknown,
): NonNullable<SkillMetadata['execution']>['argumentBindings'] | undefined {
  if (value === undefined) return undefined
  if (!Array.isArray(value)) {
    throw new Error('Invalid SKILL.md execution policy: argumentBindings must be an array')
  }
  return value.map((entry, index) => {
    if (!isRecord(entry)) {
      throw new Error(
        `Invalid SKILL.md execution policy: argument binding ${index + 1} must be an object`,
      )
    }
    const id = asString(entry.id)
    if (!id || !Array.isArray(entry.targets)) {
      throw new Error(
        `Invalid SKILL.md execution policy: argument binding ${index + 1} requires id and targets`,
      )
    }
    const targets = entry.targets.map((target, targetIndex) => {
      if (!isRecord(target)) {
        throw new Error(
          `Invalid SKILL.md execution policy: binding ${id} target ${targetIndex + 1} must be an object`,
        )
      }
      const stage = asString(target.stage)
      const argument = asString(target.argument)
      if (!stage || !argument) {
        throw new Error(
          `Invalid SKILL.md execution policy: binding ${id} targets require stage and argument`,
        )
      }
      return { stage, argument }
    })
    const allowMissing = parseExecutionBoolean(entry.allowMissing, `${id}.allowMissing`)
    return {
      id,
      targets,
      ...(allowMissing !== undefined ? { allowMissing } : {}),
    }
  })
}

function parseExecutionInteger(value: unknown, field: string): number | undefined {
  if (value === undefined) return undefined
  const parsed = Number(value)
  if (!Number.isInteger(parsed)) {
    throw new Error(`Invalid SKILL.md execution policy: ${field} must be an integer`)
  }
  return parsed
}

function parseExecutionBoolean(value: unknown, field: string): boolean | undefined {
  if (value === undefined) return undefined
  const parsed = asBool(value)
  if (parsed === undefined) {
    throw new Error(`Invalid SKILL.md execution policy: ${field} must be a boolean`)
  }
  return parsed
}

function parseExecutionStringList(value: unknown, field: string): string[] | undefined {
  if (value === undefined) return undefined
  const parsed = parseStringList(value)
  if (!parsed && !(Array.isArray(value) && value.length === 0)) {
    throw new Error(`Invalid SKILL.md execution policy: ${field} must be a string list`)
  }
  return parsed
}

function firstMarkdownParagraph(body: string): string | undefined {
  const paragraph = body
    .split(/\n\s*\n/)
    .map((part) => part.trim())
    .find((part) => part && !part.startsWith('#'))
  return paragraph
}

function asString(value: unknown): string | undefined {
  if (typeof value === 'string') return value.trim() || undefined
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return undefined
}

function asBool(value: unknown): boolean | undefined {
  if (typeof value === 'boolean') return value
  if (typeof value === 'string') {
    const v = value.trim().toLowerCase()
    if (v === 'true') return true
    if (v === 'false') return false
  }
  return undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parseSourceType(value: unknown): NonNullable<SkillMetadata['source']>['type'] | undefined {
  const type = asString(value)
  if (type === 'marketplace' || type === 'git' || type === 'url') return type
  return undefined
}

function parseSourceRecord(parsed: Record<string, unknown>): SkillMetadata['source'] | undefined {
  const sourceRecord = isRecord(parsed.source) ? parsed.source : undefined
  const type = parseSourceType(parsed.source_type) ?? parseSourceType(sourceRecord?.type)
  const ref = asString(parsed.source_ref) ?? asString(sourceRecord?.ref)
  if (!type || !ref) return undefined
  return { type, ref }
}

function parseProvenanceRecord(value: unknown): SkillMetadata['provenance'] | undefined {
  if (!isRecord(value)) return undefined
  const source = parseSourceRecord(value)
  const digest = asString(value.digest)
  const verification = asString(value.verification)
  const verified = asBool(value.verified)
  if (!source || !digest || verified === undefined) return undefined
  if (
    verification !== 'builtin'
    && verification !== 'digest'
    && verification !== 'signature'
    && verification !== 'manual'
    && verification !== 'unverified'
  ) {
    return undefined
  }

  const signatureInput = isRecord(value.signature) ? value.signature : undefined
  const signatureValue = asString(signatureInput?.value)
  const signature = signatureValue
    ? {
        algorithm: 'ed25519' as const,
        keyId: asString(signatureInput?.keyId),
        value: signatureValue,
        verified: asBool(signatureInput?.verified),
      }
    : undefined

  const scanInput = isRecord(value.scan) ? value.scan : undefined
  const scanResult = asString(scanInput?.result)
  const checkedAt = asString(scanInput?.checkedAt)
  let scan: NonNullable<SkillMetadata['provenance']>['scan'] | undefined
  if (
    (scanResult === 'pass' || scanResult === 'warn' || scanResult === 'fail' || scanResult === 'unknown')
    && checkedAt
  ) {
    scan = {
      result: scanResult,
      checkedAt,
      errors: parseStringList(scanInput?.errors),
      warnings: parseStringList(scanInput?.warnings),
    }
  }

  return {
    source,
    digest,
    installedAt: asString(value.installedAt),
    verified,
    verification,
    publisher: asString(value.publisher),
    sourceRef: asString(value.sourceRef),
    ...(signature ? { signature } : {}),
    ...(scan ? { scan } : {}),
  }
}

function parsePermissionManifest(value: unknown): SkillMetadata['permissions'] | undefined {
  if (!isRecord(value)) return undefined
  const manifest = {
    tools: parseStringList(value.tools),
    network: parseStringList(value.network),
    files: parseStringList(value.files),
  }
  return manifest.tools || manifest.network || manifest.files ? manifest : undefined
}

function parseStringList(value: unknown): string[] | undefined {
  if (Array.isArray(value)) {
    const items = value.map(asString).filter((item): item is string => Boolean(item))
    return items.length ? items : undefined
  }
  const single = asString(value)
  if (!single) return undefined
  const items = single.split(/[,\s]+/).map((item) => item.trim()).filter(Boolean)
  return items.length ? items : undefined
}

function parseToolDeclarations(value: unknown): string[] {
  const raw = Array.isArray(value)
    ? value.map(asString).filter((item): item is string => Boolean(item))
    : tokenizeToolString(asString(value))
  return Array.from(new Set(raw.flatMap(mapExternalToolName)))
}

function tokenizeToolString(value: string | undefined): string[] {
  if (!value) return []
  return value.match(/[A-Za-z0-9_.:-]+(?:\([^)]*\))?/g) ?? []
}

function mapExternalToolName(raw: string): string[] {
  const normalized = raw.trim()
  if (!normalized) return []
  const base = normalized.replace(/\(.*\)$/, '').toLowerCase()
  const direct = normalized.includes('.') ? normalized : null
  const mapped: Record<string, string[]> = {
    read: ['fs.read'],
    grep: ['fs.search'],
    glob: ['fs.glob'],
    edit: ['fs.edit'],
    multiedit: ['fs.edit'],
    write: ['fs.write'],
    append: ['fs.append'],
    bash: ['terminal.run'],
    command: ['terminal.run'],
    browser: ['browser.navigate'],
    webfetch: ['webfetch'],
    web_fetch: ['webfetch'],
    websearch: ['web.search'],
    web_search: ['web.search'],
  }
  return mapped[base] ?? (direct ? [direct] : [normalized])
}
