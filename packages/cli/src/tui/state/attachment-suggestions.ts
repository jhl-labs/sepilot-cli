import {
  resolveAttachmentCandidates,
  type AttachmentCandidate,
} from '../utils/attachments.js'

export type AttachmentSuggestionsResult =
  | { ok: true; items: AttachmentCandidate[]; cancelled?: false }
  | { ok: true; cancelled: true }
  | { ok: false }

export async function loadAttachmentSuggestions(opts: {
  referencePath: string
  rootDir: string
  cancelled: () => boolean
}): Promise<AttachmentSuggestionsResult> {
  try {
    const items = await resolveAttachmentCandidates(
      opts.referencePath,
      opts.rootDir,
    )
    if (opts.cancelled()) return { ok: true, cancelled: true }
    return { ok: true, items }
  } catch {
    if (opts.cancelled()) return { ok: true, cancelled: true }
    return { ok: false }
  }
}
