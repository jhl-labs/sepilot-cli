import {
  chmod,
  copyFile,
  lstat,
  mkdir,
  open,
  readFile,
  readdir,
  readlink,
  realpath,
  rename,
  rm,
  symlink,
  utimes,
  writeFile,
} from 'node:fs/promises'
import type { Dirent } from 'node:fs'
import { randomUUID } from 'node:crypto'
import { homedir } from 'node:os'
import {
  dirname,
  isAbsolute,
  join,
  relative,
  resolve,
  sep,
} from 'node:path'
import { openDatabase } from '../db/sqlite.js'

const STORAGE_LAYOUT_VERSION = 1 as const
const LAYOUT_DIRNAME = '.storage-layout-v1'
const ACTIVE_MARKER_FILENAME = 'active.json'
const TRANSACTIONS_DIRNAME = 'transactions'
const JOURNAL_FILENAME = 'journal.json'
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

// These paths historically resolved through SEPILOTD_HOME even when the
// daemon itself was started with a different SEPILOTD_DATA_DIR. They are the
// only roots eligible for automatic import. Canonical profile state such as
// config, sessions, memory, skills, security tokens, logs, and cache is never
// overwritten by this compatibility migration.
const LEGACY_OWNED_ENTRIES = [
  'backup',
  'channel-ingress',
  'github',
  'image-gen',
  'jobs',
  'message-subscription',
  'monitoring',
  'notifications',
  'observability',
  'persona',
  'personal-docs',
  'prompt-templates',
  'rag',
  'scheduler',
  'security/github-oauth.json',
  'snippets',
  'team-docs',
  'wiki',
] as const

const RESERVED_CANONICAL_ROOTS = new Set([
  LAYOUT_DIRNAME,
  'agents',
  'approval-checkpoints',
  'artifacts',
  'cache',
  'channel-origins',
  'channel-pipeline',
  'channel-replays',
  'channel-sessions',
  'commands',
  'history',
  'logs',
  'memory',
  'plans',
  'projects',
  'run-checkpoints',
  'security',
  'services',
  'sessions',
  'skills',
  'state',
  'tool-executions',
  'uploads',
  'user-agents',
])

type MigrationMode = 'fresh-profile' | 'empty-legacy' | 'migrated'
type TransactionStatus = 'copying' | 'prepared' | 'installing' | 'complete'

interface StorageLayoutMarker {
  version: typeof STORAGE_LAYOUT_VERSION
  mode: MigrationMode
  sourceRoot: string
  activatedAt: string
  entries: string[]
  transactionId?: string
  rollbackRoot?: string
}

interface StorageMigrationJournal {
  version: typeof STORAGE_LAYOUT_VERSION
  id: string
  sourceRoot: string
  targetRoot: string
  createdAt: string
  status: TransactionStatus
  entries: string[]
  installed: string[]
}

export interface PrepareDaemonStorageInput {
  dataDir: string
  /** Whether config.yaml existed before bootstrap created any profile files. */
  profileExisted: boolean
  legacyHome?: string
  explicitHome?: string
}

export interface PreparedDaemonStorage {
  dataDir: string
  domainHome: string
  mode: 'default-root' | 'explicit-home' | MigrationMode | 'already-active'
  migratedEntries: string[]
  rollbackRoot?: string
}

export class StorageLayoutError extends Error {
  constructor(
    readonly code:
      | 'STORAGE_LAYOUT_INVALID'
      | 'STORAGE_LAYOUT_CONFLICT'
      | 'STORAGE_LAYOUT_COPY_FAILED',
    message: string,
  ) {
    super(message)
    this.name = 'StorageLayoutError'
  }
}

function within(root: string, candidate: string): boolean {
  const rel = relative(root, candidate)
  return rel === '' || (!rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel))
}

function assertSafeEntry(entry: string): string {
  const segments = entry.split(/[\\/]+/)
  if (
    !entry
    || entry.includes('\0')
    || entry.includes('\\')
    || isAbsolute(entry)
    || segments.some((segment) => segment === '..' || segment === '')
  ) {
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_INVALID',
      `Unsafe storage migration entry: ${entry}`,
    )
  }
  return entry
}

async function exists(path: string): Promise<boolean> {
  try {
    await lstat(path)
    return true
  } catch (error) {
    if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') return false
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_INVALID',
      `Cannot inspect storage migration path ${path}: ${String(error)}`,
    )
  }
}

async function ensurePrivateDirectory(path: string): Promise<void> {
  await mkdir(path, { recursive: true, mode: 0o700 })
  const stat = await lstat(path)
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_INVALID',
      `Storage migration metadata path is not a real directory: ${path}`,
    )
  }
  await chmod(path, 0o700)
}

function layoutRoot(dataDir: string): string {
  return join(dataDir, LAYOUT_DIRNAME)
}

function markerPath(dataDir: string): string {
  return join(layoutRoot(dataDir), ACTIVE_MARKER_FILENAME)
}

function transactionRoot(dataDir: string, id: string): string {
  return join(layoutRoot(dataDir), TRANSACTIONS_DIRNAME, id)
}

function journalPath(dataDir: string, id: string): string {
  return join(transactionRoot(dataDir, id), JOURNAL_FILENAME)
}

function parseMarker(value: unknown): StorageLayoutMarker {
  if (!value || typeof value !== 'object') {
    throw new StorageLayoutError('STORAGE_LAYOUT_INVALID', 'Storage layout marker is not an object')
  }
  const marker = value as Partial<StorageLayoutMarker>
  if (
    marker.version !== STORAGE_LAYOUT_VERSION
    || !['fresh-profile', 'empty-legacy', 'migrated'].includes(marker.mode ?? '')
    || typeof marker.sourceRoot !== 'string'
    || typeof marker.activatedAt !== 'string'
    || !Array.isArray(marker.entries)
    || marker.entries.some((entry) => typeof entry !== 'string')
  ) {
    throw new StorageLayoutError('STORAGE_LAYOUT_INVALID', 'Storage layout marker has an invalid schema')
  }
  for (const entry of marker.entries) assertSafeEntry(entry)
  return marker as StorageLayoutMarker
}

function parseJournal(value: unknown): StorageMigrationJournal {
  if (!value || typeof value !== 'object') {
    throw new StorageLayoutError('STORAGE_LAYOUT_INVALID', 'Storage migration journal is not an object')
  }
  const journal = value as Partial<StorageMigrationJournal>
  if (
    journal.version !== STORAGE_LAYOUT_VERSION
    || typeof journal.id !== 'string'
    || !UUID_PATTERN.test(journal.id)
    || typeof journal.sourceRoot !== 'string'
    || !isAbsolute(journal.sourceRoot)
    || typeof journal.targetRoot !== 'string'
    || !isAbsolute(journal.targetRoot)
    || typeof journal.createdAt !== 'string'
    || !['copying', 'prepared', 'installing', 'complete'].includes(journal.status ?? '')
    || !Array.isArray(journal.entries)
    || journal.entries.some((entry) => typeof entry !== 'string')
    || !Array.isArray(journal.installed)
    || journal.installed.some((entry) => typeof entry !== 'string')
  ) {
    throw new StorageLayoutError('STORAGE_LAYOUT_INVALID', 'Storage migration journal has an invalid schema')
  }
  for (const entry of [...journal.entries, ...journal.installed]) {
    assertSafeEntry(entry)
  }
  const entries = new Set(journal.entries)
  if (
    entries.size !== journal.entries.length
    || new Set(journal.installed).size !== journal.installed.length
    || journal.installed.some((entry) => !entries.has(entry))
  ) {
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_INVALID',
      'Storage migration journal has inconsistent entry identities',
    )
  }
  return journal as StorageMigrationJournal
}

async function readJson(path: string): Promise<unknown> {
  try {
    return JSON.parse(await readFile(path, 'utf8'))
  } catch (error) {
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_INVALID',
      `Cannot read storage migration metadata at ${path}: ${String(error)}`,
    )
  }
}

async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 })
  const temp = `${path}.${randomUUID()}.tmp`
  try {
    await writeFile(temp, `${JSON.stringify(value, null, 2)}\n`, {
      encoding: 'utf8',
      mode: 0o600,
      flag: 'wx',
    })
    await rename(temp, path)
  } finally {
    await rm(temp, { force: true })
  }
}

async function readMarker(dataDir: string): Promise<StorageLayoutMarker | null> {
  const path = markerPath(dataDir)
  if (!(await exists(path))) return null
  const marker = parseMarker(await readJson(path))
  if (marker.mode === 'migrated') {
    if (
      !marker.transactionId
      || !UUID_PATTERN.test(marker.transactionId)
      || typeof marker.rollbackRoot !== 'string'
      || resolve(marker.rollbackRoot)
        !== resolve(transactionRoot(dataDir, marker.transactionId), 'rollback')
    ) {
      throw new StorageLayoutError(
        'STORAGE_LAYOUT_INVALID',
        'Migrated storage layout marker has invalid rollback metadata',
      )
    }
  }
  return marker
}

async function writeMarker(dataDir: string, marker: StorageLayoutMarker): Promise<void> {
  await writeJsonAtomic(markerPath(dataDir), marker)
}

async function readJournal(dataDir: string, id: string): Promise<StorageMigrationJournal> {
  const journal = parseJournal(await readJson(journalPath(dataDir, id)))
  if (journal.id !== id || resolve(journal.targetRoot) !== resolve(dataDir)) {
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_INVALID',
      'Storage migration journal does not belong to this data root',
    )
  }
  return journal
}

async function writeJournal(dataDir: string, journal: StorageMigrationJournal): Promise<void> {
  await writeJsonAtomic(journalPath(dataDir, journal.id), journal)
}

async function hasImmediateSqliteDb(path: string): Promise<boolean> {
  let entries
  try {
    entries = await readdir(path, { withFileTypes: true })
  } catch (error) {
    if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') return false
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_INVALID',
      `Cannot inspect legacy domain directory ${path}: ${String(error)}`,
    )
  }
  return entries.some((entry) => entry.isFile() && entry.name.endsWith('.db'))
}

async function discoverLegacyEntries(sourceRoot: string): Promise<string[]> {
  const entries = new Set<string>()
  for (const entry of LEGACY_OWNED_ENTRIES) {
    if (await exists(join(sourceRoot, entry))) entries.add(entry)
  }

  // Preserve forward compatibility for domain repositories added after this
  // migration shipped. A top-level directory with an immediate SQLite DB is a
  // stable structural signal; canonical/mixed roots are explicitly excluded.
  let roots: Dirent[] = []
  try {
    roots = await readdir(sourceRoot, { withFileTypes: true })
  } catch (error) {
    if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') return [...entries].sort()
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_INVALID',
      `Cannot enumerate legacy storage root ${sourceRoot}: ${String(error)}`,
    )
  }
  for (const entry of roots) {
    if (!entry.isDirectory() || RESERVED_CANONICAL_ROOTS.has(entry.name)) continue
    if (await hasImmediateSqliteDb(join(sourceRoot, entry.name))) {
      entries.add(entry.name)
    }
  }
  return [...entries].sort()
}

async function isSqliteFile(path: string): Promise<boolean> {
  if (!path.endsWith('.db')) return false
  const handle = await open(path, 'r')
  try {
    const header = Buffer.alloc(16)
    const { bytesRead } = await handle.read(header, 0, header.length, 0)
    return bytesRead === header.length && header.equals(Buffer.from('SQLite format 3\0'))
  } finally {
    await handle.close()
  }
}

async function snapshotSqlite(source: string, destination: string): Promise<void> {
  const db = openDatabase(source, { readonly: true, fileMustExist: true })
  try {
    const escapedDestination = destination.replaceAll("'", "''")
    db.exec(`VACUUM INTO '${escapedDestination}'`)
  } finally {
    db.close()
  }

  const snapshot = openDatabase(destination, { readonly: true, fileMustExist: true })
  try {
    const integrity = snapshot.pragma('integrity_check', { simple: true })
    if (integrity !== 'ok') {
      throw new StorageLayoutError(
        'STORAGE_LAYOUT_COPY_FAILED',
        `SQLite snapshot failed integrity check: ${destination}`,
      )
    }
  } finally {
    snapshot.close()
  }
  await chmod(destination, 0o600)
}

function shouldSkipCopy(sourceRoot: string, path: string): boolean {
  const rel = relative(sourceRoot, path).split(sep).join('/')
  if (rel === 'image-gen/local/venv' || rel.startsWith('image-gen/local/venv/')) {
    return true
  }
  if (rel.endsWith('.db-wal') || rel.endsWith('.db-shm')) {
    return true
  }
  return false
}

async function copyOwnedTree(
  sourceRoot: string,
  source: string,
  destination: string,
  entrySourceRoot: string,
): Promise<void> {
  if (shouldSkipCopy(sourceRoot, source)) return
  const stat = await lstat(source)
  if (stat.isSymbolicLink()) {
    const target = await readlink(source)
    if (isAbsolute(target) || !within(entrySourceRoot, resolve(dirname(source), target))) {
      throw new StorageLayoutError(
        'STORAGE_LAYOUT_COPY_FAILED',
        `Refusing external symlink in legacy storage: ${source}`,
      )
    }
    await mkdir(dirname(destination), { recursive: true, mode: 0o700 })
    await symlink(target, destination)
    return
  }
  if (stat.isDirectory()) {
    await mkdir(destination, { recursive: true, mode: stat.mode & 0o777 })
    for (const child of await readdir(source)) {
      await copyOwnedTree(
        sourceRoot,
        join(source, child),
        join(destination, child),
        entrySourceRoot,
      )
    }
    await chmod(destination, stat.mode & 0o777)
    await utimes(destination, stat.atime, stat.mtime)
    return
  }
  if (!stat.isFile()) {
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_COPY_FAILED',
      `Unsupported file type in legacy storage: ${source}`,
    )
  }
  await mkdir(dirname(destination), { recursive: true, mode: 0o700 })
  if (await isSqliteFile(source)) {
    await snapshotSqlite(source, destination)
  } else {
    await copyFile(source, destination)
    await chmod(destination, stat.mode & 0o777)
  }
  await utimes(destination, stat.atime, stat.mtime)
}

async function resetAndCopyStage(
  dataDir: string,
  journal: StorageMigrationJournal,
): Promise<StorageMigrationJournal> {
  const root = transactionRoot(dataDir, journal.id)
  const stage = join(root, 'stage')
  await rm(stage, { recursive: true, force: true })
  await mkdir(stage, { recursive: true, mode: 0o700 })
  const entries = await discoverLegacyEntries(journal.sourceRoot)
  const physicalDataDir = await realpath(dataDir)
  for (const entry of entries) {
    const safeEntry = assertSafeEntry(entry)
    const source = join(journal.sourceRoot, safeEntry)
    const physicalSource = await realpath(source)
    if (
      within(physicalSource, physicalDataDir)
      || within(physicalDataDir, physicalSource)
    ) {
      throw new StorageLayoutError(
        'STORAGE_LAYOUT_CONFLICT',
        `Storage migration roots overlap at entry: ${safeEntry}`,
      )
    }
    await copyOwnedTree(
      journal.sourceRoot,
      source,
      join(stage, safeEntry),
      source,
    )
  }
  const prepared: StorageMigrationJournal = {
    ...journal,
    status: 'prepared',
    entries,
    installed: [],
  }
  await writeJournal(dataDir, prepared)
  return prepared
}

async function installTransaction(
  dataDir: string,
  initial: StorageMigrationJournal,
): Promise<StorageMigrationJournal> {
  let journal = initial.status === 'prepared'
    ? { ...initial, status: 'installing' as const }
    : initial
  if (initial.status === 'prepared') await writeJournal(dataDir, journal)

  const root = transactionRoot(dataDir, journal.id)
  const installed = new Set(journal.installed)
  for (const entry of journal.entries) {
    if (installed.has(entry)) continue
    const stage = join(root, 'stage', entry)
    const target = join(dataDir, entry)
    const rollback = join(root, 'rollback', entry)
    const stageExists = await exists(stage)
    const targetExists = await exists(target)
    const rollbackExists = await exists(rollback)

    if (!stageExists) {
      if (!targetExists) {
        throw new StorageLayoutError(
          'STORAGE_LAYOUT_CONFLICT',
          `Interrupted storage migration lost both staged and active entry: ${entry}`,
        )
      }
      installed.add(entry)
    } else {
      if (targetExists) {
        if (rollbackExists) {
          throw new StorageLayoutError(
            'STORAGE_LAYOUT_CONFLICT',
            `Storage migration target changed during recovery: ${entry}`,
          )
        }
        await mkdir(dirname(rollback), { recursive: true, mode: 0o700 })
        await rename(target, rollback)
      }
      await mkdir(dirname(target), { recursive: true, mode: 0o700 })
      await rename(stage, target)
      installed.add(entry)
    }

    journal = { ...journal, installed: [...installed] }
    await writeJournal(dataDir, journal)
  }

  journal = { ...journal, status: 'complete', installed: [...installed] }
  await writeJournal(dataDir, journal)
  return journal
}

async function activeTransaction(dataDir: string): Promise<StorageMigrationJournal | null> {
  const root = join(layoutRoot(dataDir), TRANSACTIONS_DIRNAME)
  let ids: string[]
  try {
    ids = (await readdir(root, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
  } catch (error) {
    if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') return null
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_INVALID',
      `Cannot enumerate storage migration transactions: ${String(error)}`,
    )
  }
  const active: StorageMigrationJournal[] = []
  for (const id of ids) {
    const journalFile = journalPath(dataDir, id)
    if (!(await exists(journalFile))) continue
    const journal = await readJournal(dataDir, id)
    // A process can stop after the last entry and completed journal were
    // committed but before active.json was renamed into place. With no marker
    // present, that completed transaction is still the only authoritative
    // recovery candidate.
    active.push(journal)
  }
  if (active.length > 1) {
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_CONFLICT',
      'Multiple unfinished storage migration transactions require operator review',
    )
  }
  return active[0] ?? null
}

async function completeMigration(
  dataDir: string,
  journal: StorageMigrationJournal,
): Promise<PreparedDaemonStorage> {
  const completed = journal.status === 'complete'
    ? journal
    : await installTransaction(dataDir, journal)
  if (
    completed.entries.length !== completed.installed.length
    || completed.entries.some((entry) => !completed.installed.includes(entry))
  ) {
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_INVALID',
      'Completed storage migration journal is missing installed entries',
    )
  }
  for (const entry of completed.entries) {
    if (!(await exists(join(dataDir, entry)))) {
      throw new StorageLayoutError(
        'STORAGE_LAYOUT_CONFLICT',
        `Completed storage migration is missing active entry: ${entry}`,
      )
    }
  }
  const rollbackRoot = join(transactionRoot(dataDir, completed.id), 'rollback')
  await writeMarker(dataDir, {
    version: STORAGE_LAYOUT_VERSION,
    mode: 'migrated',
    sourceRoot: completed.sourceRoot,
    activatedAt: new Date().toISOString(),
    entries: completed.entries,
    transactionId: completed.id,
    rollbackRoot,
  })
  return {
    dataDir,
    domainHome: dataDir,
    mode: 'migrated',
    migratedEntries: completed.entries,
    rollbackRoot,
  }
}

async function beginMigration(
  dataDir: string,
  sourceRoot: string,
): Promise<StorageMigrationJournal> {
  const id = randomUUID()
  const root = transactionRoot(dataDir, id)
  await mkdir(root, { recursive: false, mode: 0o700 })
  const journal: StorageMigrationJournal = {
    version: STORAGE_LAYOUT_VERSION,
    id,
    sourceRoot,
    targetRoot: dataDir,
    createdAt: new Date().toISOString(),
    status: 'copying',
    entries: [],
    installed: [],
  }
  await writeJournal(dataDir, journal)
  return resetAndCopyStage(dataDir, journal)
}

async function assertLegacyDaemonInactive(legacyHome: string): Promise<void> {
  const pidPath = join(legacyHome, 'sepilotd.pid')
  let raw: string
  try {
    raw = (await readFile(pidPath, 'utf8')).trim()
  } catch (error) {
    if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') return
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_CONFLICT',
      `Cannot verify whether legacy storage is active: ${String(error)}`,
    )
  }
  if (!/^[1-9][0-9]*$/.test(raw) || !Number.isSafeInteger(Number(raw))) {
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_CONFLICT',
      'Legacy storage has an indeterminate PID owner; refusing migration',
    )
  }
  try {
    process.kill(Number(raw), 0)
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_CONFLICT',
      'Legacy storage is owned by a running daemon; stop it before migration',
    )
  } catch (error) {
    if (error instanceof StorageLayoutError) throw error
    if ((error as NodeJS.ErrnoException)?.code !== 'ESRCH') {
      throw new StorageLayoutError(
        'STORAGE_LAYOUT_CONFLICT',
        'Legacy storage ownership could not be proven inactive',
      )
    }
  }
}

/**
 * Resolve and, when required, migrate the legacy SEPILOTD_HOME-owned state
 * before any domain repository is opened. Source state is never removed.
 * Existing target entries are atomically archived inside the transaction's
 * rollback tree before the consistent source snapshot becomes active.
 */
export async function prepareDaemonStorage(
  input: PrepareDaemonStorageInput,
): Promise<PreparedDaemonStorage> {
  const dataDir = resolve(input.dataDir)
  const legacyHome = resolve(input.legacyHome ?? join(homedir(), '.sepilotd'))
  const explicitHomeValue = input.explicitHome ?? process.env.SEPILOTD_HOME
  const explicitHome = explicitHomeValue?.trim()

  if (explicitHome) {
    return {
      dataDir,
      domainHome: resolve(explicitHome),
      mode: 'explicit-home',
      migratedEntries: [],
    }
  }
  if (dataDir === legacyHome) {
    return {
      dataDir,
      domainHome: dataDir,
      mode: 'default-root',
      migratedEntries: [],
    }
  }
  await mkdir(dataDir, { recursive: true, mode: 0o700 })
  const physicalDataDir = await realpath(dataDir)
  if (await exists(legacyHome)) {
    let physicalLegacyHome: string
    try {
      physicalLegacyHome = await realpath(legacyHome)
    } catch (error) {
      throw new StorageLayoutError(
        'STORAGE_LAYOUT_INVALID',
        `Cannot resolve legacy storage root ${legacyHome}: ${String(error)}`,
      )
    }
    if (physicalDataDir === physicalLegacyHome) {
      return {
        dataDir,
        domainHome: dataDir,
        mode: 'default-root',
        migratedEntries: [],
      }
    }
  }
  await ensurePrivateDirectory(layoutRoot(dataDir))
  await ensurePrivateDirectory(join(layoutRoot(dataDir), TRANSACTIONS_DIRNAME))

  const marker = await readMarker(dataDir)
  if (marker) {
    return {
      dataDir,
      domainHome: dataDir,
      mode: 'already-active',
      migratedEntries: marker.entries,
      rollbackRoot: marker.rollbackRoot,
    }
  }

  let transaction = await activeTransaction(dataDir)
  if (transaction) {
    if (resolve(transaction.sourceRoot) !== legacyHome) {
      throw new StorageLayoutError(
        'STORAGE_LAYOUT_CONFLICT',
        'Unfinished storage migration references a different legacy root',
      )
    }
    if (transaction.status === 'copying') {
      await assertLegacyDaemonInactive(legacyHome)
      transaction = await resetAndCopyStage(dataDir, transaction)
    }
    return completeMigration(dataDir, transaction)
  }

  if (!input.profileExisted) {
    const freshMarker: StorageLayoutMarker = {
      version: STORAGE_LAYOUT_VERSION,
      mode: 'fresh-profile',
      sourceRoot: legacyHome,
      activatedAt: new Date().toISOString(),
      entries: [],
    }
    await writeMarker(dataDir, freshMarker)
    return {
      dataDir,
      domainHome: dataDir,
      mode: 'fresh-profile',
      migratedEntries: [],
    }
  }

  await assertLegacyDaemonInactive(legacyHome)
  const entries = await discoverLegacyEntries(legacyHome)
  if (entries.length === 0) {
    const emptyMarker: StorageLayoutMarker = {
      version: STORAGE_LAYOUT_VERSION,
      mode: 'empty-legacy',
      sourceRoot: legacyHome,
      activatedAt: new Date().toISOString(),
      entries: [],
    }
    await writeMarker(dataDir, emptyMarker)
    return {
      dataDir,
      domainHome: dataDir,
      mode: 'empty-legacy',
      migratedEntries: [],
    }
  }

  try {
    transaction = await beginMigration(dataDir, legacyHome)
    return await completeMigration(dataDir, transaction)
  } catch (error) {
    if (error instanceof StorageLayoutError) throw error
    throw new StorageLayoutError(
      'STORAGE_LAYOUT_COPY_FAILED',
      `Legacy storage migration failed: ${String(error)}`,
    )
  }
}

export const storageLayoutPaths = {
  layoutDirname: LAYOUT_DIRNAME,
  markerFilename: ACTIVE_MARKER_FILENAME,
  transactionsDirname: TRANSACTIONS_DIRNAME,
  journalFilename: JOURNAL_FILENAME,
} as const
