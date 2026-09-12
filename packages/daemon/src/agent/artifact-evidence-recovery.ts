const PATH_EVIDENCE_BLOCK_PATTERN =
  /required artifact "([^"]+)" cites repository paths without prior read\/search\/glob\/write evidence:\s*([\s\S]+?)(?:\.\s+(?:Do not retry|Verify|Next valid step)|$)/i

export const ARTIFACT_EVIDENCE_RECOVERY_TOOL_NAMES = new Set([
  'fs.read',
  'fs.glob',
  'fs.search',
  'code.symbols',
  'code.dependencies',
  'code.diagnostics',
  'lsp',
])

export interface ArtifactEvidenceRecoveryToolResult {
  toolName?: string
  output: string
  status: 'success' | 'error'
}

export interface RequiredArtifactPathEvidenceBlock {
  artifact: string
  missing: string[]
}

export function parseRequiredArtifactPathEvidenceBlock(
  output: string,
): RequiredArtifactPathEvidenceBlock | null {
  const match = output.match(PATH_EVIDENCE_BLOCK_PATTERN)
  if (!match) {
    return null
  }
  const artifact = (match[1] ?? 'the requested artifact').trim()
  const missingText = (match[2] ?? '').trim().replace(/\.$/, '')
  const missing = missingText
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
  return { artifact, missing }
}

export function isRequiredArtifactPathEvidenceBlock(output: string): boolean {
  return PATH_EVIDENCE_BLOCK_PATTERN.test(output)
}

export function isArtifactEvidenceRecoveryToolName(toolName: string | undefined): boolean {
  return Boolean(toolName && ARTIFACT_EVIDENCE_RECOVERY_TOOL_NAMES.has(toolName))
}

export function shouldEnterArtifactEvidenceRecovery(
  recentToolResults: readonly ArtifactEvidenceRecoveryToolResult[] | undefined,
): boolean {
  if (!recentToolResults || recentToolResults.length === 0) {
    return false
  }

  for (let i = recentToolResults.length - 1; i >= 0; i -= 1) {
    const result = recentToolResults[i]
    if (!result) continue

    if (result.status === 'success' && isArtifactEvidenceRecoveryToolName(result.toolName)) {
      return false
    }

    if (result.status === 'error' && isRequiredArtifactPathEvidenceBlock(result.output)) {
      return true
    }
  }

  return false
}

export function buildRequiredArtifactPathEvidenceRepairMessage(output: string): string {
  const block = parseRequiredArtifactPathEvidenceBlock(output)
  const artifact = block?.artifact ?? 'the requested artifact'
  const missing = block?.missing.join(', ')
  return [
    '[Tool recovery]',
    `The previous required artifact write for ${artifact} was blocked because it cited repository paths without read/search/glob evidence.`,
    missing ? `Missing evidence: ${missing}.` : '',
    'Next turn must gather evidence before another artifact write: call fs.glob for missing inventories/directories, fs.search for symbols/routes, or fs.read for concrete files.',
    'After evidence is gathered, reuse the blocked draft from the previous file-edit tool call and retry only after verifying, removing, or marking those claims as explicit coverage gaps.',
    'Do not call fs.write/fs.append/fs.edit/apply_patch for that artifact again until the missing path claims are verified or removed. If no valid evidence step remains, answer INCOMPLETE with the blocker.',
  ].filter(Boolean).join(' ')
}

export function buildArtifactEvidenceRecoveryToolRestrictionMessage(
  toolNames: readonly string[],
): string {
  const names = toolNames.length > 0
    ? toolNames.map((toolName) => `\`${toolName}\``).join(', ')
    : 'the available read/search/inventory tools'
  return [
    '[Artifact evidence recovery]',
    'The previous required artifact write was blocked by the structural evidence guard.',
    `For this turn, use only evidence-gathering tools: ${names}.`,
    'Gather the missing path evidence first; after a successful evidence tool result, continue the normal write and validation cycle.',
  ].join(' ')
}
