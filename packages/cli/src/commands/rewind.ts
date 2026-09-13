import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export interface RewindOptions { url?: string; scope?: 'files' | 'conversation' | 'both'; apply?: boolean; turns?: string }
export type RewindClient = Pick<DaemonClient, 'listSessionCheckpoints' | 'previewSessionRewind' | 'rewindSessionFiles' | 'session' | 'branchSession'>

/** Conversation history is branched, never destroyed; file mutation requires explicit apply. */
export async function runRewind(client: RewindClient, sessionId: string, checkpointId: string | undefined, options: RewindOptions = {}): Promise<unknown> {
  const scope = options.scope ?? 'files'
  if (scope !== 'conversation' && !checkpointId) return { scope, checkpoints: await client.listSessionCheckpoints(sessionId), next: 'Choose a checkpoint to preview, then use --apply to restore files.' }
  const source = await client.session(sessionId)
  const preview = scope !== 'conversation' ? await client.previewSessionRewind(sessionId, checkpointId!) : undefined
  if (preview?.files.some(file => file.conflict) || preview?.incompleteHistory) {
    if (options.apply) throw new Error('Rewind refused: conflicting files or incomplete checkpoint history. Inspect the preview; no files were changed.')
  }
  let fromEventIndex: number | undefined
  if (scope !== 'files') {
    const indices = source.events.flatMap((event, index) => event.type === 'user_message' && (!preview || event.timestamp <= preview.checkpoint.createdAt) ? [index] : [])
    const turns = options.turns === undefined ? 1 : Number(options.turns)
    if (!Number.isSafeInteger(turns) || turns < 1) throw new Error('turns must be a positive integer')
    fromEventIndex = indices[Math.max(0, indices.length - turns)]
    if (fromEventIndex === undefined) throw new Error('No user turn exists before this rewind point')
  }
  if (!options.apply) return { scope, preview, fromEventIndex, next: 'Preview only. Use --apply to create the conversation branch and/or restore files.' }
  // Preserve the original conversation even if the subsequent file operation
  // fails. Never switch the UI to this branch until all requested work succeeds.
  const branch = fromEventIndex === undefined ? undefined : await client.branchSession(sessionId, { fromEventIndex })
  try {
    const files = preview ? await client.rewindSessionFiles(sessionId, checkpointId!, preview.checkpointIds) : undefined
    return { scope, branch, files, originalSessionId: sessionId }
  } catch (error) {
    if (branch) throw new Error(`File rewind failed. Original conversation is intact; unused recovery branch ${branch.branchId} is retained. ${error instanceof Error ? error.message : String(error)}`)
    throw error
  }
}

export async function rewindCommand(sessionId: string, checkpointId: string | undefined, options: RewindOptions) {
  output(await runRewind(new DaemonClient(options.url), sessionId, checkpointId, options), value => JSON.stringify(value, null, 2))
}
