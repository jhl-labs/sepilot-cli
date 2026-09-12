import { join } from 'node:path'
import type { Artifact } from '../../agent/artifacts.js'
import { extractArtifacts, extractImageArtifacts } from '../../agent/artifacts.js'
import { ArtifactStore } from '../../memory/artifact-store.js'
import { getRuntimeDataDir } from './utils.js'

export async function extractAndStoreArtifacts(
  runtime: { dataDir?: string },
  sessionId: string,
  content: string,
): Promise<Artifact[]> {
  if (!content) return []

  const { artifacts } = extractArtifacts(content, { sessionId })
  if (artifacts.length === 0) return []

  try {
    const store = new ArtifactStore(join(getRuntimeDataDir(runtime), 'artifacts'))
    return await Promise.all(
      artifacts.map((artifact) => store.save(sessionId, artifact)),
    )
  } catch {
    // Artifact persistence should not fail the parent run.
  }

  return artifacts
}

export async function extractAndStoreUserImageArtifacts(
  runtime: { dataDir?: string },
  sessionId: string,
  content: string,
): Promise<Artifact[]> {
  if (!content) return []

  const artifacts = extractImageArtifacts(content, { sessionId })
  if (artifacts.length === 0) return []

  try {
    const store = new ArtifactStore(join(getRuntimeDataDir(runtime), 'artifacts'))
    return await Promise.all(
      artifacts.map((artifact) => store.save(sessionId, artifact)),
    )
  } catch {
    // Artifact persistence should not fail the parent run.
  }

  return artifacts
}
