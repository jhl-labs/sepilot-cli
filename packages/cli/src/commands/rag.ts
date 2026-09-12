import { basename, resolve } from 'node:path'
import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { getOutputFormat, output } from '../output/formatter.js'
import { detectCliLocale } from '../utils/locale.js'

const RAG_COPY = {
  en: {
    noSources: 'No RAG sources.',
    pathPrefix: 'path',
    docsSuffix: (n: number) => `${n} docs`,
    syncPrefix: 'sync',
    sourceAddedPrefix: (name: string) => `RAG source added: ${name}`,
    idPrefix: 'ID:',
    pathLabelPrefix: 'Path:',
    noResults: 'No RAG results.',
    folderPrefix: 'folder',
    scorePrefix: 'score',
    noDocuments: 'No RAG documents.',
    bytesSuffix: (n: number) => `${n} bytes`,
    localRag: 'Local RAG',
    enginePrefix: 'Engine:',
    documentsPrefix: 'Documents:',
    statusPrefix: 'Status:',
    dimensionsPrefix: 'Dimensions:',
    backendAvailablePrefix: 'Backend available:',
    lastErrorPrefix: 'Last error:',
    syncCompleted: 'RAG sync completed',
    syncCompletedWithErrors: 'RAG sync completed with errors',
    syncDetails: (folders: number, indexed: number, deleted: number, skipped: number) =>
      `Folders: ${folders} • indexed ${indexed} • deleted ${deleted} • skipped ${skipped}`,
  },
  ko: {
    noSources: 'RAG 소스가 없습니다.',
    pathPrefix: '경로',
    docsSuffix: (n: number) => `문서 ${n}개`,
    syncPrefix: '동기화',
    sourceAddedPrefix: (name: string) => `RAG 소스 추가됨: ${name}`,
    idPrefix: 'ID:',
    pathLabelPrefix: '경로:',
    noResults: 'RAG 결과가 없습니다.',
    folderPrefix: '폴더',
    scorePrefix: '점수',
    noDocuments: 'RAG 문서가 없습니다.',
    bytesSuffix: (n: number) => `${n} 바이트`,
    localRag: '로컬 RAG',
    enginePrefix: '엔진:',
    documentsPrefix: '문서:',
    statusPrefix: '상태:',
    dimensionsPrefix: '차원:',
    backendAvailablePrefix: '백엔드 사용 가능:',
    lastErrorPrefix: '마지막 오류:',
    syncCompleted: 'RAG 동기화 완료',
    syncCompletedWithErrors: 'RAG 동기화 완료 (오류 있음)',
    syncDetails: (folders: number, indexed: number, deleted: number, skipped: number) =>
      `폴더: ${folders}개 • 인덱싱 ${indexed} • 삭제 ${deleted} • 건너뜀 ${skipped}`,
  },
} as const

function ragCopy() {
  return RAG_COPY[detectCliLocale()] ?? RAG_COPY.en
}

export async function ragSourcesListCommand(options: { url?: string }) {
  const copy = ragCopy()
  const client = new DaemonClient(options.url)
  const folders = await client.ragFolders()

  if (getOutputFormat() === 'json') {
    output(folders)
    return
  }

  if (folders.length === 0) {
    console.log(copy.noSources)
    return
  }

  for (const folder of folders) {
    console.log(chalk.cyan(folder.name))
    const details = [
      folder.id,
      folder.sourceType,
      folder.path ? `${copy.pathPrefix} ${folder.path}` : null,
      copy.docsSuffix(folder.documents),
      folder.lastSyncStatus ? `${copy.syncPrefix} ${folder.lastSyncStatus}` : null,
    ].filter((part): part is string => Boolean(part))
    console.log(chalk.gray(`  ${details.join(' • ')}`))
    if (folder.lastSyncError) {
      console.log(chalk.yellow(`  ${folder.lastSyncError}`))
    }
  }
}

export async function ragSourceAddCommand(
  sourcePath: string,
  options: {
    url?: string
    id?: string
    name?: string
    include?: string
    exclude?: string
    sync?: boolean
  },
) {
  const client = new DaemonClient(options.url)
  const absolutePath = resolve(sourcePath)
  const folder = await client.upsertRagFolder({
    id: options.id?.trim() || undefined,
    name: options.name?.trim() || basename(absolutePath) || absolutePath,
    sourceType: 'git',
    path: absolutePath,
    include: splitCsv(options.include),
    exclude: splitCsv(options.exclude),
  })
  const syncResult = options.sync === false ? null : await client.syncRag()

  if (getOutputFormat() === 'json') {
    output({ folder, sync: syncResult })
    return
  }

  const copy = ragCopy()
  console.log(chalk.green(copy.sourceAddedPrefix(folder.name)))
  console.log(chalk.gray(`${copy.idPrefix} ${folder.id}`))
  console.log(chalk.gray(`${copy.pathLabelPrefix} ${absolutePath}`))
  if (syncResult) {
    printSyncResult(syncResult)
  }
}

export async function ragSyncCommand(options: { url?: string }) {
  const client = new DaemonClient(options.url)
  const result = await client.syncRag()

  if (getOutputFormat() === 'json') {
    output(result)
    return
  }

  printSyncResult(result)
}

export async function ragSearchCommand(
  query: string,
  options: { url?: string; limit?: string },
) {
  const client = new DaemonClient(options.url)
  const hits = await client.searchRag(query, parseLimit(options.limit))

  if (getOutputFormat() === 'json') {
    output(hits)
    return
  }

  const copy = ragCopy()
  if (hits.length === 0) {
    console.log(copy.noResults)
    return
  }

  for (const hit of hits) {
    console.log(chalk.cyan(hit.title))
    console.log(`  ${hit.snippet}`)
    const details = [
      hit.path ? `${copy.pathPrefix} ${hit.path}` : null,
      `${copy.folderPrefix} ${hit.folderId}`,
      `${copy.scorePrefix} ${hit.score.toFixed(2)}`,
    ].filter((part): part is string => Boolean(part))
    console.log(chalk.gray(`  ${details.join(' • ')}`))
  }
}

export async function ragDocumentsListCommand(
  folderId: string,
  options: { url?: string },
) {
  const client = new DaemonClient(options.url)
  const documents = await client.ragDocuments(folderId)

  if (getOutputFormat() === 'json') {
    output(documents)
    return
  }

  const copy = ragCopy()
  if (documents.length === 0) {
    console.log(copy.noDocuments)
    return
  }

  for (const document of documents) {
    console.log(chalk.cyan(document.title))
    const details = [
      document.id,
      document.path ? `${copy.pathPrefix} ${document.path}` : null,
      document.size != null ? copy.bytesSuffix(document.size) : null,
    ].filter((part): part is string => Boolean(part))
    console.log(chalk.gray(`  ${details.join(' • ')}`))
  }
}

export async function ragStatusCommand(options: { url?: string }) {
  const client = new DaemonClient(options.url)
  const info = await client.ragVectorDbInfo()

  if (getOutputFormat() === 'json') {
    output(info)
    return
  }

  const copy = ragCopy()
  console.log(chalk.green(copy.localRag))
  console.log(chalk.gray(`${copy.enginePrefix} ${info.engine}`))
  console.log(chalk.gray(`${copy.documentsPrefix} ${info.documents}`))
  if (info.status) console.log(chalk.gray(`${copy.statusPrefix} ${info.status}`))
  if (info.dimension) console.log(chalk.gray(`${copy.dimensionsPrefix} ${info.dimension}`))
  if (info.backendAvailable != null) {
    console.log(chalk.gray(`${copy.backendAvailablePrefix} ${String(info.backendAvailable)}`))
  }
  if (info.lastError) console.log(chalk.yellow(`${copy.lastErrorPrefix} ${info.lastError}`))
}

function printSyncResult(result: {
  ok: boolean
  folders: number
  indexed: number
  deleted: number
  skipped: number
  errors: Array<{ folderId: string; path?: string; error: string }>
}) {
  const copy = ragCopy()
  const color = result.ok ? chalk.green : chalk.yellow
  console.log(color(result.ok ? copy.syncCompleted : copy.syncCompletedWithErrors))
  console.log(
    chalk.gray(copy.syncDetails(result.folders, result.indexed, result.deleted, result.skipped)),
  )
  for (const error of result.errors) {
    console.log(chalk.yellow(`  ${error.folderId}: ${error.error}`))
  }
}

function splitCsv(value: string | undefined): string[] | undefined {
  const parts = value
    ?.split(',')
    .map((part) => part.trim())
    .filter(Boolean)
  return parts?.length ? parts : undefined
}

function parseLimit(limit: string | undefined): number | undefined {
  if (!limit) return undefined
  const parsed = Number.parseInt(limit, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined
}
