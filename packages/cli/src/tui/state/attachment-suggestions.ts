// Pulled out of App.tsx so the @path attachment-completion lookup
// can return a tagged result instead of forcing the effect to run
// setters from inside try/catch + cancellation guards.

import {
  resolveAttachmentCandidates,
  type AttachmentCandidate,
} from '../utils/attachments.js'

export type AttachmentSuggestionsResult =
  | { ok: true; items: AttachmentCandidate[]; cancelled?: false }
  | { ok: true; cancelled: true }
  | { ok: false }

/**
 * Resolve completion candidates for a partially-typed attachment
 * reference. The underlying lookup walks the file index against the
 * caller-supplied workspace root; failures (no matches, IO error)
 * collapse to ok=false so the caller renders an empty suggestions
 * list rather than surfacing a stack trace.
 */
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
