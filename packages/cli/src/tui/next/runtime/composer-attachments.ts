import { access } from 'node:fs/promises'
import { extractAttachmentReferences, resolveAttachmentPath } from '../../utils/attachments.js'

export async function resolveComposerAttachments(
  text: string,
  cwd = process.cwd(),
  exists: (path: string) => Promise<unknown> = access,
): Promise<Array<{ path: string }>> {
  const unique = [...new Set(extractAttachmentReferences(text).map(({ path }) => path))]
  const resolved = await Promise.all(unique.map(async (path) => {
    try {
      await exists(resolveAttachmentPath(path, cwd))
      return { path }
    } catch {
      return null
    }
  }))
  return resolved.filter((entry): entry is { path: string } => entry !== null)
}
