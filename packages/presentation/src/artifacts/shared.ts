export interface ArtifactLike {
  id: string
  type: string
  title?: string
  language?: string
  content: string
}

export function artifactLabel(artifact: ArtifactLike): string {
  return artifact.title || artifact.language || artifact.type
}

export function artifactPreview(
  artifact: ArtifactLike,
  limit = 160,
): string {
  if (artifact.type === 'image') {
    return artifact.title || 'Image artifact'
  }

  const normalized = artifact.content.replace(/\s+/g, ' ').trim()
  if (!normalized) {
    return 'No preview available'
  }
  return normalized.length > limit
    ? `${normalized.slice(0, limit - 1)}…`
    : normalized
}
