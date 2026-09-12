import type {
  DaemonRagFolder,
  DaemonRagSyncResult,
  DaemonRagVectorDbInfo,
} from '@sepilotd/api-client'

export const RAG_COMMAND_USAGE = [
  'Usage: /rag [open|current]',
  '       /rag search <query>',
  '       /rag add <path> [name...]',
  '       /rag sync',
  '       /rag close',
].join('\n')

export function formatRagSource(source: DaemonRagFolder): string {
  const details = [
    source.sourceType,
    source.path ? `path=${source.path}` : null,
    `${source.documents} docs`,
    source.lastSyncStatus ? `sync=${source.lastSyncStatus}` : null,
  ].filter(Boolean)
  return `${source.name}  ${source.id}\n  ${details.join('  ')}${source.lastSyncError ? `\n  ${source.lastSyncError}` : ''}`
}

export function formatRagSyncResult(result: DaemonRagSyncResult): string {
  const headline = result.ok
    ? 'RAG sync completed.'
    : 'RAG sync completed with errors.'
  const lines = [
    headline,
    `folders=${result.folders} indexed=${result.indexed} deleted=${result.deleted} skipped=${result.skipped}`,
  ]
  for (const error of result.errors) {
    lines.push(`${error.folderId}: ${error.error}`)
  }
  return lines.join('\n')
}

export function formatRagVectorInfo(
  info: DaemonRagVectorDbInfo | null,
): string {
  if (!info) return 'vector backend unavailable'
  const details = [
    `engine=${info.engine}`,
    info.status ? `status=${info.status}` : null,
    info.backendAvailable != null ? `available=${String(info.backendAvailable)}` : null,
    info.dimension ? `dim=${info.dimension}` : null,
    `documents=${info.documents}`,
  ].filter(Boolean)
  return details.join('  ')
}
