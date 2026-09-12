import { createHash, verify as verifySignature } from 'node:crypto'
import type { ToolDefinitionRuntime } from '../tools/registry.js'
import type { McpMarketplace } from './marketplace-catalog.js'
import type { McpServerTemplate } from './marketplace-source.js'

export interface McpProvenanceSignature {
  algorithm: 'ed25519'
  keyId?: string
  value: string
  verified: boolean
}

export interface McpServerProvenance {
  source: 'builtin' | 'marketplace' | 'manual'
  marketplace?: string
  publisher?: string
  homepage?: string
  repository?: string
  sourceRef?: string
  templateDigest?: string
  declaredDigest?: string
  verified: boolean
  verification: 'builtin' | 'signature' | 'digest' | 'manual' | 'unverified'
  signature?: McpProvenanceSignature
  installedAt?: string
}

export interface McpToolManifestEntry {
  name: string
  digest: string
}

export interface McpToolManifest {
  version: 1
  generatedAt: string
  digest: string
  tools: McpToolManifestEntry[]
}

export interface McpSecurityAlert {
  code:
    | 'MCP_PROVENANCE_MISSING'
    | 'MCP_PROVENANCE_UNVERIFIED'
    | 'MCP_TOOL_MANIFEST_BASELINE_UNPERSISTED'
    | 'MCP_TOOL_MANIFEST_CHANGED'
  severity: 'warning' | 'critical'
  message: string
  detectedAt: string
  details?: Record<string, unknown>
}

export interface McpManifestComparison {
  changed: boolean
  added: string[]
  removed: string[]
  changedTools: string[]
}

type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue }

function stableJson(value: unknown): string {
  return JSON.stringify(stableValue(value))
}

function stableValue(value: unknown): JsonValue {
  if (value === undefined) return null
  if (value === null || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value
  }
  if (Array.isArray(value)) return value.map(stableValue)
  if (typeof value === 'object') {
    const out: Record<string, JsonValue> = {}
    for (const key of Object.keys(value as Record<string, unknown>).sort()) {
      const entry = (value as Record<string, unknown>)[key]
      if (entry !== undefined) out[key] = stableValue(entry)
    }
    return out
  }
  return String(value)
}

function sha256(value: string): string {
  return `sha256:${createHash('sha256').update(value).digest('hex')}`
}

export function mcpTemplateDigestInput(template: McpServerTemplate): Record<string, unknown> {
  return {
    name: template.name,
    description: template.description,
    transport: template.transport,
    command: template.command,
    args: template.args ?? [],
    url: template.url,
    env: template.env ?? {},
    headers: template.headers ?? {},
    tags: template.tags ?? [],
    homepage: template.homepage,
    variables: template.variables ?? [],
    publisher: template.publisher,
  }
}

export function computeMcpTemplateDigest(template: McpServerTemplate): string {
  return sha256(stableJson(mcpTemplateDigestInput(template)))
}

export function verifyMcpTemplateSignature(args: {
  digest: string
  signature?: string
  publicKey?: string
}): boolean {
  if (!args.signature || !args.publicKey) return false
  try {
    return verifySignature(
      null,
      Buffer.from(args.digest, 'utf-8'),
      args.publicKey,
      Buffer.from(args.signature, 'base64'),
    )
  } catch {
    return false
  }
}

export function buildMcpServerProvenance(args: {
  template: McpServerTemplate
  marketplace?: McpMarketplace | null
  sourceRef?: string
  installedAt?: string
  allowUnverified?: boolean
}): McpServerProvenance {
  const templateDigest = computeMcpTemplateDigest(args.template)
  const declaredDigest = args.template.digest
  if (declaredDigest && declaredDigest !== templateDigest) {
    throw new Error(
      `MCP template digest mismatch for ${args.template.name}: declared ${declaredDigest}, computed ${templateDigest}`,
    )
  }

  const source = args.template.marketplace === 'builtin' ? 'builtin' : 'marketplace'
  const signatureVerified = verifyMcpTemplateSignature({
    digest: templateDigest,
    signature: args.template.signature,
    publicKey: args.marketplace?.publicKey ?? args.template.marketplacePublicKey,
  })
  const digestVerified = !!declaredDigest && declaredDigest === templateDigest
  const verified = source === 'builtin' || signatureVerified || digestVerified

  if (!verified && !args.allowUnverified) {
    throw new Error(
      `MCP template "${args.template.name}" is not verified. Require a matching digest or marketplace signature, or pass allowUnverified for a manual-risk install.`,
    )
  }

  return {
    source,
    marketplace: args.template.marketplace,
    publisher: args.template.publisher ?? args.marketplace?.publisher ?? args.template.marketplace,
    homepage: args.template.homepage,
    repository: args.template.repository ?? args.marketplace?.url ?? args.template.marketplaceUrl,
    sourceRef: args.sourceRef ?? args.template.sourceRef,
    templateDigest,
    declaredDigest,
    verified,
    verification: source === 'builtin'
      ? 'builtin'
      : signatureVerified
        ? 'signature'
        : digestVerified
          ? 'digest'
          : 'unverified',
    signature: args.template.signature
      ? {
          algorithm: 'ed25519',
          keyId: args.template.signatureKeyId,
          value: args.template.signature,
          verified: signatureVerified,
        }
      : undefined,
    installedAt: args.installedAt ?? new Date().toISOString(),
  }
}

export function buildMcpToolManifest(tools: ToolDefinitionRuntime[]): McpToolManifest {
  const entries = tools
    .map((tool) => ({
      name: tool.name,
      digest: sha256(stableJson({
        name: tool.name,
        description: tool.description,
        inputSchema: tool.inputSchema,
      })),
    }))
    .sort((a, b) => a.name.localeCompare(b.name))
  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    digest: sha256(stableJson(entries)),
    tools: entries,
  }
}

export function compareMcpToolManifest(
  expected: McpToolManifest | undefined,
  actual: McpToolManifest,
): McpManifestComparison {
  if (!expected) {
    return { changed: false, added: [], removed: [], changedTools: [] }
  }
  const expectedByName = new Map(expected.tools.map((tool) => [tool.name, tool.digest]))
  const actualByName = new Map(actual.tools.map((tool) => [tool.name, tool.digest]))
  const added = actual.tools
    .map((tool) => tool.name)
    .filter((name) => !expectedByName.has(name))
    .sort()
  const removed = expected.tools
    .map((tool) => tool.name)
    .filter((name) => !actualByName.has(name))
    .sort()
  const changedTools = actual.tools
    .filter((tool) => expectedByName.has(tool.name) && expectedByName.get(tool.name) !== tool.digest)
    .map((tool) => tool.name)
    .sort()

  return {
    changed: expected.digest !== actual.digest || added.length > 0 || removed.length > 0 || changedTools.length > 0,
    added,
    removed,
    changedTools,
  }
}
