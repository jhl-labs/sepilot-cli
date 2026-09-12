import { createHash, verify } from 'node:crypto'
import type { SkillMetadata, SkillProvenanceRecord, SkillSignatureRecord } from '@sepilotd/core'
import type { FileSkillRegistry } from './registry.js'
import type { SkillSource, SkillRef, FetchedSkill } from './sources/types.js'
import { parseSkillSource } from './source-parser.js'
import { SkillAlreadyExistsError, SkillDigestMismatchError, SkillDigestRequiredError } from './errors.js'

export interface InstallPipelineDeps {
  registry: FileSkillRegistry
  urlSource: SkillSource
  gitSource: SkillSource
  marketplaceSource: SkillSource
  // Trusted ed25519 public keys (keyId -> base64/PEM public key) used to verify
  // a skill's advertised signature. Empty by default: no scanner or trust
  // anchor exists yet, so signatures are recorded but never claimed verified.
  trustedSignatureKeys?: ReadonlyMap<string, string>
}

export interface InstallInput {
  source: string
  force?: boolean
  allowOverwrite?: boolean
  expectedDigest?: string
  requireExpectedDigest?: boolean
}

export interface InstallRefInput {
  ref: SkillRef
  force?: boolean
  allowOverwrite?: boolean
  expectedDigest?: string
  requireExpectedDigest?: boolean
}

export interface InstallResult {
  installed: SkillMetadata[]
  digest: string
}

export interface InstallPreviewResult {
  fetched: FetchedSkill[]
  digest: string
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue)
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([, item]) => item !== undefined)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, item]) => [key, stableValue(item)]),
    )
  }
  return value
}

function metadataForDigest(item: FetchedSkill): SkillMetadata {
  const {
    source: _source,
    provenance: _provenance,
    enabled: _enabled,
    ...metadata
  } = item.metadata
  return { ...metadata, source: item.source }
}

export function skillInstallDigest(items: FetchedSkill[]): string {
  const normalized = items
    .map((item) => ({
      source: item.source,
      metadata: metadataForDigest(item),
      content: item.content,
    }))
    .sort((a, b) => {
      const aKey = `${a.metadata.id}\n${a.source.type}\n${a.source.ref}`
      const bKey = `${b.metadata.id}\n${b.source.type}\n${b.source.ref}`
      return aKey.localeCompare(bKey)
    })
  return `sha256:${createHash('sha256').update(JSON.stringify(stableValue(normalized))).digest('hex')}`
}

function classifySkillRiskTier(metadata: SkillMetadata): SkillMetadata['risk_tier'] {
  const tools = metadata.tools ?? []
  if (tools.some((tool) => (
    tool === 'terminal.run'
    || tool === 'process.start'
    || tool === 'process.stop'
    || tool === 'process.signal'
    || tool === 'service.start'
    || tool === 'service.stop'
    || tool === 'service.restart'
    || tool === 'service.remove'
    || tool === 'service.healthcheck'
    || tool === 'fs.write'
    || tool === 'fs.append'
    || tool === 'fs.edit'
    || tool === 'apply_patch'
    || tool.startsWith('computer.')
    || tool.startsWith('mcp.')
  ))) {
    return 'high'
  }
  if (tools.some((tool) => (
    tool.startsWith('browser.')
    || tool === 'webfetch'
    || tool === 'memory.remember'
    || tool === 'memory.update'
    || tool === 'memory.documents.ingest'
    || tool === 'memory.documents.update'
  ))) {
    return 'medium'
  }
  return 'low'
}

function hostFromRef(ref: string): string | undefined {
  try {
    const url = new URL(ref)
    return url.host || undefined
  } catch {
    return undefined
  }
}

function decodeSignature(signature: string): Buffer {
  const value = signature.startsWith('base64:') ? signature.slice('base64:'.length) : signature
  return Buffer.from(value, 'base64')
}

// Verify an advertised ed25519 signature over the install digest against a
// trusted key. Returns false on any failure (unknown key, bad signature,
// malformed input) so a claimed-but-unverifiable signature is never trusted.
function verifySkillSignature(
  signature: SkillSignatureRecord | undefined,
  digest: string,
  trustedKeys: ReadonlyMap<string, string> | undefined,
): boolean {
  if (!signature?.value || !signature.keyId || !trustedKeys) return false
  const publicKey = trustedKeys.get(signature.keyId)
  if (!publicKey) return false
  try {
    return verify(null, Buffer.from(digest), publicKey, decodeSignature(signature.value))
  } catch {
    return false
  }
}

function securityMetadata(
  item: FetchedSkill,
  digest: string,
  expectedDigest: string | undefined,
  installedAt: string,
  trustedKeys: ReadonlyMap<string, string> | undefined,
): SkillMetadata {
  const metadata: SkillMetadata = { ...item.metadata, source: item.source }
  const network = hostFromRef(item.source.ref)
  const advertisedSignature = item.signature ?? item.metadata.provenance?.signature
  const signatureVerified = verifySkillSignature(advertisedSignature, digest, trustedKeys)
  // The digest is a trust-on-first-use transit integrity check (does the
  // fetched bytes match what the caller pinned?), NOT proof of authorship.
  // Authenticity only comes from a verified signature.
  const digestMatches = expectedDigest === digest
  const verification: SkillProvenanceRecord['verification'] = signatureVerified
    ? 'signature'
    : digestMatches
      ? 'digest'
      : 'unverified'
  const signature: SkillSignatureRecord | undefined = advertisedSignature
    ? { ...advertisedSignature, verified: signatureVerified }
    : undefined
  return {
    ...metadata,
    risk_tier: metadata.risk_tier ?? classifySkillRiskTier(metadata),
    permissions: {
      ...(metadata.permissions ?? {}),
      tools: metadata.permissions?.tools ?? metadata.tools,
      ...(network ? { network: metadata.permissions?.network ?? [network] } : {}),
    },
    provenance: {
      source: item.source,
      digest,
      installedAt,
      verified: signatureVerified || digestMatches,
      verification,
      ...(item.publisher ? { publisher: item.publisher } : {}),
      ...(item.sourceRef ? { sourceRef: item.sourceRef } : {}),
      ...(signature ? { signature } : {}),
      // No real content scanner ships yet — record 'unknown' rather than
      // falsely stamping 'pass'.
      scan: {
        result: 'unknown',
        checkedAt: installedAt,
      },
    },
  }
}

export function createInstallPipeline(deps: InstallPipelineDeps) {
  const pickSource = (ref: SkillRef): SkillSource => {
    if (ref.type === 'url') return deps.urlSource
    if (ref.type === 'git') return deps.gitSource
    return deps.marketplaceSource
  }

  const registerAll = async (
    items: FetchedSkill[],
    force: boolean,
    allowOverwrite: boolean,
    expectedDigest?: string,
    requireExpectedDigest?: boolean,
  ): Promise<InstallResult> => {
    const digest = skillInstallDigest(items)
    if (requireExpectedDigest && !expectedDigest) {
      throw new SkillDigestRequiredError(digest)
    }
    if (expectedDigest && digest !== expectedDigest) {
      throw new SkillDigestMismatchError(expectedDigest, digest)
    }
    const installed: SkillMetadata[] = []
    const installedAt = new Date().toISOString()
    for (const item of items) {
      const metadata = securityMetadata(item, digest, expectedDigest, installedAt, deps.trustedSignatureKeys)
      if (!force && !allowOverwrite && await deps.registry.get(metadata.id)) {
        throw new SkillAlreadyExistsError(metadata.id)
      }
      await deps.registry.register(metadata, item.content, { force: false })
      installed.push(metadata)
    }
    return { installed, digest }
  }

  return {
    async preview(input: InstallInput): Promise<InstallPreviewResult> {
      const ref = parseSkillSource(input.source)
      const source = pickSource(ref)
      const fetched = await source.fetch(ref)
      return { fetched, digest: skillInstallDigest(fetched) }
    },
    async previewRef(input: InstallRefInput): Promise<InstallPreviewResult> {
      const source = pickSource(input.ref)
      const fetched = await source.fetch(input.ref)
      return { fetched, digest: skillInstallDigest(fetched) }
    },
    async install(input: InstallInput): Promise<InstallResult> {
      const ref = parseSkillSource(input.source)
      const source = pickSource(ref)
      const fetched = await source.fetch(ref)
      return registerAll(
        fetched,
        input.force === true,
        input.allowOverwrite === true,
        input.expectedDigest,
        input.requireExpectedDigest === true,
      )
    },
    async installRef(input: InstallRefInput): Promise<InstallResult> {
      const source = pickSource(input.ref)
      const fetched = await source.fetch(input.ref)
      return registerAll(
        fetched,
        input.force === true,
        input.allowOverwrite === true,
        input.expectedDigest,
        input.requireExpectedDigest === true,
      )
    },
  }
}

export type InstallPipeline = ReturnType<typeof createInstallPipeline>
