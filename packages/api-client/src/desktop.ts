import type {
  ChatOptions,
  ChatStreamOptions,
  DaemonListSessionsOptions,
} from './daemon/http.js'
import type { ApprovalDecisionStatus, ApprovalRule, ApprovalScope } from '@sepilotd/core'
import type {
  SepilotAppCapabilities,
  SepilotAppDataPolicy,
  SepilotAppDataSchema,
  SepilotAppHostRequestPermission,
} from './apps.js'

export type { ApprovalDecisionStatus, ApprovalRule, ApprovalScope } from '@sepilotd/core'
import type {
  DaemonAgentDescriptor,
  DaemonArtifact,
  DaemonApprovalResponseResult,
  DaemonChatResult,
  DaemonChatStreamPayload,
  DaemonConfig,
  DaemonConfigUpdateInput,
  DaemonConfigUpdateResult,
  DaemonDoctorReport,
  DaemonHealth,
  DaemonHealthReportSnapshot,
  DaemonFileMemorySectionUpdateResult,
  DaemonFileMemorySnapshot,
  DaemonMemoryAddOptions,
  DaemonMemoryAddResult,
  DaemonMemoryAuditEntry,
  DaemonMemoryAuditOptions,
  DaemonMemoryDeleteOptions,
  DaemonMemoryDocument,
  DaemonMemoryDocumentChunk,
  DaemonMemoryDocumentIngestInput,
  DaemonMemoryDocumentListOptions,
  DaemonMemoryDocumentSearchOptions,
  DaemonMemoryEntry,
  DaemonMemoryLifecycleOptions,
  DaemonMemoryLifecycleStatus,
  DaemonMemoryMaintenanceInput,
  DaemonMemoryMaintenanceResult,
  DaemonMemoryRecentEntry,
  DaemonMemoryRecentOptions,
  DaemonMemoryReindexResult,
  DaemonMemoryScopes,
  DaemonMemorySearchOptions,
  DaemonMemoryScopeTransferInput,
  DaemonMemoryScopeTransferResult,
  DaemonMemorySecurityAuditOptions,
  DaemonMemorySecurityAuditResult,
  DaemonMemorySemanticStatus,
  DaemonMemoryUpdateInput,
  DaemonPersona,
  DaemonProviderInfo,
  DaemonProject,
  DaemonSessionCompactResult,
  DaemonSessionDetail,
  DaemonSessionList,
  DaemonSessionsWatchPayload,
  DaemonSessionWatchPayload,
  DaemonSkill,
  DaemonUsageSummary,
  MarketplaceSkillSearchResult,
} from './daemon/types.js'
import type { DaemonDevice } from './daemon/types.js'
import type {
  GatewayTicketComment,
  GatewayTicketCommentsWatchPayload,
  GatewayTicketsWatchPayload,
} from './gateway/http.js'
import type { PaginatedResult, Ticket } from '@sepilotd/core'

export type DesktopShellKind = 'local' | 'ssh'

export interface DesktopShellStartInput {
  /**
   * Optional UUID reserved by the caller before startup so data/exit IPC
   * listeners can be attached before a short-lived process is spawned.
   */
  id?: string
  kind?: DesktopShellKind
  cwd?: string
  /**
   * Optional local command to run as the PTY process instead of the default
   * shell. Used by desktop-only integrations that need a real TTY-backed
   * process while rendering output somewhere other than the terminal pane.
   */
  command?: string[]
  title?: string
  env?: Record<string, string>
  cols?: number
  rows?: number
  terminalType?: string
  encoding?: string
  ssh?: {
    host?: string
    user?: string
    port?: number
    identityFile?: string
    jumpHost?: string
    proxyCommand?: string
    localForwards?: string[]
    remoteEnv?: string[]
    keepAlive?: boolean
    agentForwarding?: boolean
    x11Forwarding?: boolean
    compression?: boolean
    terminalType?: string
    encoding?: string
  }
}

export interface DesktopShellSessionInfo {
  id: string
  kind: DesktopShellKind
  pid: number
  title: string
  command: string
  cwd: string
}

export interface DesktopShellExitEvent {
  exitCode: number
  signal?: number
}

export interface DesktopShellDoctorCheck {
  id: string
  label: string
  status: 'pass' | 'warn' | 'fail'
  message: string
  detail?: string
}

export interface DesktopShellDoctorReport {
  target: string
  generatedAt: string
  checks: DesktopShellDoctorCheck[]
}

export interface SepilotDesktopShellApi {
  start(input?: DesktopShellStartInput): Promise<DesktopShellSessionInfo>
  write(id: string, data: string): Promise<void>
  resize(id: string, cols: number, rows: number): Promise<void>
  kill(id: string): Promise<void>
  readSshConfig?(): Promise<string | null>
  diagnose?(input: DesktopShellStartInput): Promise<DesktopShellDoctorReport>
  onData(id: string, handler: (data: string) => void): () => void
  onExit(id: string, handler: (event: DesktopShellExitEvent) => void): () => void
}

export interface DesktopAppPermissions {
  storage: boolean
  network: boolean
  clipboard: boolean
  fileSystem: boolean
  shell: boolean
}

export interface DesktopAppValidationIssue {
  severity: 'error' | 'warning'
  code: string
  message: string
}

export interface DesktopAppValidationResult {
  ok: boolean
  issues: DesktopAppValidationIssue[]
}

export interface DesktopAppVersionInfo {
  version: number
  createdAt: number
  file: string
  sha256: string
  prompt?: string
}

export interface DesktopAppManifest {
  schemaVersion: 1
  frameworkVersion?: 1
  id: string
  title: string
  kind: string
  summary?: string
  category?: string
  accent?: string
  origin?: 'template' | 'code-agent'
  entry: 'index.html'
  dataFile: 'data.json'
  createdAt: number
  updatedAt: number
  version: number
  permissions: DesktopAppPermissions
  dataPolicy?: SepilotAppDataPolicy
  capabilities?: SepilotAppCapabilities
  hostRequests?: SepilotAppHostRequestPermission[]
  dataSchema?: SepilotAppDataSchema
  sandbox: string
  versions: DesktopAppVersionInfo[]
  validation: DesktopAppValidationResult
}

export interface DesktopAppBundle {
  manifest: DesktopAppManifest
  html: string
  data: Record<string, unknown>
  path: string
}

export interface DesktopAppStorageDiagnostic {
  directoryName: string
  path: string
  status: 'valid' | 'invalid'
  canDelete: boolean
  manifest?: Partial<DesktopAppManifest> & {
    id?: string
    title?: string
    kind?: string
    version?: number
    updatedAt?: number
  }
  validation: DesktopAppValidationResult
}

export type DesktopAppStorageNodeType = 'file' | 'directory' | 'symlink'

export type DesktopAppStorageFileKind =
  | 'manifest'
  | 'html'
  | 'data'
  | 'chat'
  | 'version'
  | 'audit'
  | 'backup'
  | 'unknown'

export interface DesktopAppStorageFileEntry {
  id: string
  appId: string | null
  appTitle?: string
  appKind?: string
  name: string
  path: string
  relativePath: string
  virtualPath: string
  nodeType: DesktopAppStorageNodeType
  fileKind: DesktopAppStorageFileKind
  sizeBytes: number | null
  updatedAt: number | null
  depth: number
  readable: boolean
  editable: boolean
  sensitive: boolean
}

export interface DesktopAppStorageOwner {
  id: string
  title: string
  kind: string
  path: string
  status: 'valid' | 'invalid' | 'unknown'
  validation?: DesktopAppValidationResult
  metrics: {
    fileCount: number
    directoryCount: number
    totalBytes: number
    editableCount: number
    sensitiveCount: number
  }
  files: DesktopAppStorageFileEntry[]
}

export interface DesktopAppStorageSnapshot {
  rootPath: string
  collectedAt: number
  apps: DesktopAppStorageOwner[]
  looseFiles: DesktopAppStorageFileEntry[]
  diagnostics: DesktopAppStorageDiagnostic[]
  truncated?: boolean
}

export interface DesktopAppStorageReadInput {
  relativePath: string
}

export interface DesktopAppStorageReadResult {
  file: DesktopAppStorageFileEntry
  content: string
  encoding: 'utf-8'
  sha256: string
  language: 'json' | 'html' | 'markdown' | 'yaml' | 'text'
}

export interface DesktopAppStorageWriteInput {
  relativePath: string
  content: string
  expectedSha256?: string | null
}

export interface DesktopAppStorageWriteResult extends DesktopAppStorageReadResult {
  savedAt: number
}

export interface DesktopAppChatTurn {
  id: string
  role: 'user' | 'assistant'
  content: string
  createdAt: number
  dataChanged?: boolean
  statusLabel?: string
}

export interface DesktopAppAuditEntry {
  id: string
  appId: string
  operation: string
  capability: string
  status: 'success' | 'error'
  source: 'sandbox' | 'agent' | 'manager'
  message?: string
  createdAt: number
}

export interface DesktopAppAuditInput {
  appId: string
  operation: string
  capability?: string
  status?: 'success' | 'error'
  source?: 'sandbox' | 'agent' | 'manager'
  message?: string
}

export interface DesktopAppSaveInput {
  id?: string
  manifest: Partial<DesktopAppManifest> & {
    title: string
    kind: string
  }
  html: string
  data?: Record<string, unknown>
  prompt?: string
  createVersion?: boolean
}

export interface DesktopServiceHealthCheckItem {
  id: string
  name?: string
  url: string
  port?: number | null
}

export interface DesktopServiceHealthTcpResult {
  status: 'ok' | 'error' | 'skipped'
  host?: string
  port?: number
  latencyMs?: number
  error?: string
}

export interface DesktopServiceHealthHttpResult {
  status: 'ok' | 'warning' | 'error' | 'skipped'
  statusCode?: number
  latencyMs?: number
  error?: string
}

export interface DesktopServiceHealthCheckResult {
  id: string
  name?: string
  url: string
  status: 'healthy' | 'degraded' | 'down' | 'skipped'
  checkedAt: number
  message: string
  tcp: DesktopServiceHealthTcpResult
  http: DesktopServiceHealthHttpResult
}

export interface DesktopServiceHealthCheckInput {
  services: DesktopServiceHealthCheckItem[]
  timeoutMs?: number
}

export interface DesktopServiceHealthCheckBatchResult {
  checkedAt: number
  results: DesktopServiceHealthCheckResult[]
}

export interface DesktopMarketQuoteAsset {
  id: string
  name?: string
  symbol?: string
  market?: string
  type?: string
  currency?: string
}

export interface DesktopMarketQuotePoint {
  ts: number
  date: string
  close: number
}

export interface DesktopMarketQuoteResult {
  id: string
  name?: string
  symbol?: string
  requestSymbol: string
  ok: boolean
  price?: number
  currency?: string
  asOf?: number
  points: DesktopMarketQuotePoint[]
  source: 'yahoo-finance-chart'
  sourceUrl?: string
  message?: string
}

export interface DesktopMarketQuotesInput {
  assets: DesktopMarketQuoteAsset[]
  range?: '5d' | '1mo' | '3mo' | '6mo' | '1y' | '2y' | '5y' | string
  interval?: '1d' | '1wk' | '1mo' | string
}

export interface DesktopMarketQuoteBatchResult {
  checkedAt: number
  provider: string
  range: string
  interval: string
  results: DesktopMarketQuoteResult[]
}

export interface DesktopKubernetesConnectionInput {
  id?: string
  name?: string
  provider?: string
  apiUrl: string
  token?: string
  caCert?: string
  insecureSkipTlsVerify?: boolean
  credentialType?: 'serviceAccount' | 'rancher'
  rancherSourceId?: string
  rancherUrl?: string
  rancherClusterId?: string
  rancherClusterName?: string
}

export interface DesktopKubernetesSnapshotInput {
  connection: DesktopKubernetesConnectionInput
  saveCredential?: boolean
  timeoutMs?: number
}

export interface DesktopRancherImportInput {
  rancherUrl: string
  token?: string
  caCert?: string
  insecureSkipTlsVerify?: boolean
  provider?: string
  timeoutMs?: number
  cacheTtlMinutes?: number
}

export interface DesktopKubernetesConnectionSummary {
  id: string
  name: string
  provider: string
  apiUrl: string
  caCert?: string
  insecureSkipTlsVerify?: boolean
  credentialType?: 'serviceAccount' | 'rancher'
  rancherSourceId?: string
  rancherUrl?: string
  rancherClusterId?: string
  rancherClusterName?: string
  kubeconfigCached?: boolean
  kubeconfigFetchedAt?: number
  kubeconfigExpiresAt?: number
  credentialStored?: boolean
  lastSyncAt: number
  lastSyncStatus: 'success' | 'error'
  lastSyncMessage?: string
}

export interface DesktopKubernetesCredentialDeleteResult {
  ok: true
  deleted: boolean
}

export interface DesktopKubernetesClusterSnapshot {
  id: string
  connectionId: string
  name: string
  provider: string
  apiUrl: string
  region?: string
  nodes: number
  readyNodes: number
  vcpu: number
  memoryGb: number
  version?: string
  status: 'Healthy' | 'Degraded' | 'Down'
  cpuUsage?: number | null
  memoryUsage?: number | null
  storageUsage?: number | null
  lastSeen: string
  lastSyncAt: number
  warnings: string[]
}

export interface DesktopKubernetesNamespaceSnapshot {
  id: string
  clusterId: string
  name: string
  workloads: number
  podsReady: number
  podsTotal: number
  restarts: number
  cpuUsage?: number | null
  memoryUsage?: number | null
  status: '정상' | '경고' | '장애'
}

export interface DesktopKubernetesEventSnapshot {
  id: string
  clusterId: string
  severity: '정보' | '경고' | '장애'
  title: string
  namespace?: string
  age: string
  status: 'open' | 'resolved'
  createdAt: number
}

export interface DesktopKubernetesDeploymentSnapshot {
  id: string
  clusterId: string
  service: string
  namespace: string
  version?: string
  status: '안정' | '진행중' | '실패' | '대기'
  progress: number
  updatedAt: string
}

export interface DesktopKubernetesSnapshotResult {
  ok: true
  checkedAt: number
  connection: DesktopKubernetesConnectionSummary
  cluster: DesktopKubernetesClusterSnapshot
  namespaces: DesktopKubernetesNamespaceSnapshot[]
  events: DesktopKubernetesEventSnapshot[]
  deployments: DesktopKubernetesDeploymentSnapshot[]
  warnings: string[]
}

export interface DesktopRancherImportFailure {
  clusterId: string
  name: string
  message: string
}

export interface DesktopRancherImportResult {
  ok: true
  checkedAt: number
  source: {
    id: string
    url: string
    credentialStored: boolean
    kubeconfigCacheTtlMinutes: number
  }
  connections: DesktopKubernetesConnectionSummary[]
  snapshots: DesktopKubernetesSnapshotResult[]
  failures: DesktopRancherImportFailure[]
  warnings: string[]
}

export interface DesktopSmartHomeProviderStatus {
  provider: 'SmartThings' | 'Google Home'
  configured: boolean
  connected: boolean
  available: boolean
  locationId?: string | null
  projectId?: string | null
  message?: string
}

export interface DesktopSmartHomeStatus {
  smartThings: DesktopSmartHomeProviderStatus
  googleHome: DesktopSmartHomeProviderStatus
}

export interface DesktopSmartHomeDevice {
  id: string
  externalId?: string
  provider: 'SmartThings' | 'Google Home' | 'Matter' | 'Home Assistant' | '수동'
  name: string
  room: string
  type: string
  brand: string
  status: string
  power: boolean
  brightness?: number
  targetTemp?: number
  currentValue?: number
  unit?: string
  battery?: number | null
  mode?: string
  lastSeen: number
  componentId?: string
  capabilities?: string[]
  metadata?: Record<string, unknown>
}

export interface DesktopSmartHomeSyncInput {
  provider?: 'SmartThings' | 'Google Home' | string
}

export interface DesktopSmartHomeSyncResult {
  ok: boolean
  provider: 'SmartThings' | 'Google Home'
  devices: DesktopSmartHomeDevice[]
  status: DesktopSmartHomeStatus
  syncedAt: number
  warnings: string[]
}

export interface DesktopSmartHomeDeviceCommandInput {
  provider?: 'SmartThings' | 'Google Home' | string
  deviceId: string
  componentId?: string | null
  action: 'power' | 'brightness' | 'lock' | 'targetTemp' | string
  value?: unknown
  device?: Partial<DesktopSmartHomeDevice>
}

export interface DesktopSmartHomeDeviceCommandResult {
  ok: boolean
  provider: 'SmartThings' | 'Google Home'
  device?: DesktopSmartHomeDevice
  acceptedAt: number
  message?: string
}

export interface SepilotDesktopAppsApi {
  list(): Promise<DesktopAppBundle[]>
  diagnostics(): Promise<DesktopAppStorageDiagnostic[]>
  save(input: DesktopAppSaveInput): Promise<DesktopAppBundle>
  saveData(id: string, data: Record<string, unknown>): Promise<DesktopAppBundle>
  readChat(id: string): Promise<DesktopAppChatTurn[]>
  saveChat(id: string, history: DesktopAppChatTurn[]): Promise<DesktopAppChatTurn[]>
  restoreVersion(id: string, version: number): Promise<DesktopAppBundle>
  exportApp(id: string): Promise<string | null>
  importApp(): Promise<DesktopAppBundle | null>
  exportBackup(): Promise<string | null>
  restoreBackup(): Promise<DesktopAppBundle[] | null>
  audit(appId?: string): Promise<DesktopAppAuditEntry[]>
  recordAudit(input: DesktopAppAuditInput): Promise<void>
  delete(id: string): Promise<void>
  reset(apps: DesktopAppSaveInput[]): Promise<DesktopAppBundle[]>
  storageTree?(): Promise<DesktopAppStorageSnapshot>
  storageReadFile?(input: DesktopAppStorageReadInput): Promise<DesktopAppStorageReadResult>
  storageWriteFile?(input: DesktopAppStorageWriteInput): Promise<DesktopAppStorageWriteResult>
  checkServiceHealth?(
    input: DesktopServiceHealthCheckInput,
  ): Promise<DesktopServiceHealthCheckBatchResult>
  fetchMarketQuotes?(input: DesktopMarketQuotesInput): Promise<DesktopMarketQuoteBatchResult>
  fetchKubernetesSnapshot?(
    input: DesktopKubernetesSnapshotInput,
  ): Promise<DesktopKubernetesSnapshotResult>
  importRancherClusters?(input: DesktopRancherImportInput): Promise<DesktopRancherImportResult>
  deleteKubernetesCredential?(connectionId: string): Promise<DesktopKubernetesCredentialDeleteResult>
  smartHomeStatus?(): Promise<DesktopSmartHomeStatus>
  syncSmartHomeDevices?(input: DesktopSmartHomeSyncInput): Promise<DesktopSmartHomeSyncResult>
  commandSmartHomeDevice?(
    input: DesktopSmartHomeDeviceCommandInput,
  ): Promise<DesktopSmartHomeDeviceCommandResult>
}

export interface DesktopCalendarGoogleStatus {
  configured: boolean
  connected: boolean
  appId?: string
  appTitle?: string
  calendarId?: string
  selectedCalendarIds?: string[]
  defaultCalendarId?: string | null
  autoSyncIntervalMinutes?: number | null
  lastAutoSyncAt?: number
  lastAutoSyncStatus?: 'success' | 'error'
  lastAutoSyncMessage?: string
}

export interface DesktopCalendarGoogleCalendar {
  id: string
  summary: string
  primary?: boolean
  selected?: boolean
  accessRole?: string
  backgroundColor?: string
  foregroundColor?: string
  timeZone?: string
  writable: boolean
}

export interface DesktopCalendarGoogleEventLink {
  googleEventId: string
  googleCalendarId: string
  googleCalendarSummary?: string
  googleICalUID?: string
  googleRecurringEventId?: string
  googleOriginalStartTime?: DesktopCalendarGoogleEventDateTime
  remoteUpdatedAt?: string
}

export interface DesktopCalendarGoogleEventDateTime {
  date?: string
  dateTime?: string
  timeZone?: string
}

export interface DesktopCalendarEventInput {
  id: string
  title: string
  date: string
  time?: string
  endDate?: string
  endTime?: string
  type?: string
  notes?: string
  location?: string
  timezone?: string
  sharedOriginKind?: string
  sharedOriginTitle?: string
  sharedOriginId?: string
  sharedReadonly?: boolean
  googleEventId?: string
  googleCalendarId?: string
  googleCalendarSummary?: string
  googleICalUID?: string
  googleRecurringEventId?: string
  googleOriginalStartTime?: DesktopCalendarGoogleEventDateTime
  googleLinks?: DesktopCalendarGoogleEventLink[]
  updatedAt?: number
}

export interface DesktopCalendarGoogleSyncState {
  provider: 'google'
  calendarId?: string
  selectedCalendarIds?: string[]
  defaultCalendarId?: string | null
  hiddenEventKeys?: string[]
  eventMap: Record<
    string,
    {
      googleEventId: string
      calendarId?: string
      iCalUID?: string
      recurringEventId?: string
      originalStartTime?: DesktopCalendarGoogleEventDateTime
      contentHash?: string
      updatedAt?: number
      remoteUpdatedAt?: string
      links?: DesktopCalendarGoogleEventLink[]
    }
  >
  autoSyncIntervalMinutes?: number | null
  lastSyncedAt?: number
  lastSyncStatus?: 'success' | 'error'
  lastSyncMessage?: string
}

export interface DesktopCalendarGoogleSyncInput {
  appId: string
  appTitle: string
  events: DesktopCalendarEventInput[]
  state?: DesktopCalendarGoogleSyncState | null
  intervalMinutes?: number | null
}

export interface DesktopCalendarGoogleSyncResult {
  ok: boolean
  inserted: number
  updated: number
  deleted: number
  pulled: number
  skipped: number
  state: DesktopCalendarGoogleSyncState
  events: DesktopCalendarEventInput[]
  errors: string[]
}

export type DesktopCalendarGoogleDeleteScope = 'single' | 'series' | 'future'

export interface DesktopCalendarGoogleDeleteInput {
  appId: string
  appTitle?: string
  eventId: string
  scope: DesktopCalendarGoogleDeleteScope
  events: DesktopCalendarEventInput[]
  state?: DesktopCalendarGoogleSyncState | null
}

export interface DesktopCalendarGoogleDeleteResult {
  ok: boolean
  deleted: number
  updated: number
  state: DesktopCalendarGoogleSyncState
  events: DesktopCalendarEventInput[]
  errors: string[]
}

export interface DesktopCalendarAutoSyncInput {
  appId: string
  appTitle: string
  intervalMinutes?: number | null
}

export interface DesktopCalendarGoogleCalendarsConfigInput {
  appId: string
  appTitle?: string
  selectedCalendarIds: string[]
  defaultCalendarId?: string | null
}

export interface DesktopCalendarIcsExportInput {
  appTitle: string
  events: DesktopCalendarEventInput[]
}

export interface SepilotDesktopCalendarApi {
  status(
    appId?: string,
    state?: DesktopCalendarGoogleSyncState | null,
  ): Promise<DesktopCalendarGoogleStatus>
  configureGoogle(input: {
    clientId?: string | null
    clientSecret?: string | null
  }): Promise<DesktopCalendarGoogleStatus>
  connectGoogle(): Promise<DesktopCalendarGoogleStatus>
  disconnectGoogle(): Promise<DesktopCalendarGoogleStatus>
  listGoogleCalendars(appId?: string): Promise<DesktopCalendarGoogleCalendar[]>
  configureGoogleCalendars(
    input: DesktopCalendarGoogleCalendarsConfigInput,
  ): Promise<DesktopCalendarGoogleStatus>
  syncAppEvents(input: DesktopCalendarGoogleSyncInput): Promise<DesktopCalendarGoogleSyncResult>
  deleteGoogleEvent(
    input: DesktopCalendarGoogleDeleteInput,
  ): Promise<DesktopCalendarGoogleDeleteResult>
  configureAutoSync(input: DesktopCalendarAutoSyncInput): Promise<DesktopCalendarGoogleStatus>
  exportIcs(input: DesktopCalendarIcsExportInput): Promise<string | null>
}

export interface DesktopRendererErrorReportInput {
  source?: string
  message: string
  name?: string
  stack?: string
  componentStack?: string
  href?: string
  userAgent?: string
  sessionId?: string
  runId?: string
  requestId?: string
  attributes?: Record<string, unknown>
}

export interface DesktopRendererErrorReportResult {
  id: string
  timestamp: string
  logPath: string | null
  observabilityAccepted?: number
}

export interface DesktopDiagnosticsExportInput {
  reportId?: string
  requestId?: string
  sessionId?: string
  runId?: string
  includeHealthReport?: boolean
}

export interface DesktopDiagnosticsExportResult {
  id: string
  path: string
  filename: string
  generatedAt: string
  includedFiles: string[]
  sizeBytes: number
}

export type DesktopWikiFolderEntry =
  | {
      kind: 'directory'
      relativePath: string
    }
  | {
      kind: 'file'
      relativePath: string
      content: string
    }

export interface DesktopWikiFolderImportSelection {
  rootName: string
  entries: DesktopWikiFolderEntry[]
  ignoredPaths: string[]
  totalBytes: number
  directoryCount: number
}

export interface DesktopWikiFolderExportPayload {
  directoryName: string
  entries: DesktopWikiFolderEntry[]
}

export type DesktopWikiFolderExportConflictKind =
  | 'directory'
  | 'file'
  | 'other'
  | 'symbolic_link'

export interface DesktopWikiFolderExportConflict {
  relativePath: string
  kind: DesktopWikiFolderExportConflictKind
}

export interface DesktopWikiFolderExportPreview {
  token: string
  displayPath: string
  expiresAt: number
  entryCount: number
  fileCount: number
  directoryCount: number
  totalBytes: number
  createCount: number
  overwriteCount: number
  blockedConflicts: DesktopWikiFolderExportConflict[]
}

export interface DesktopWikiFolderExportApplyInput {
  token: string
  confirmed: boolean
}

export interface DesktopWikiFolderExportResult {
  displayPath: string
  entryCount: number
  createdCount: number
  overwrittenCount: number
}

export interface SepilotDesktopWikiFilesApi {
  pickImportFolder(): Promise<DesktopWikiFolderImportSelection | null>
  previewFolderExport(
    payload: DesktopWikiFolderExportPayload,
  ): Promise<DesktopWikiFolderExportPreview | null>
  applyFolderExport(
    input: DesktopWikiFolderExportApplyInput,
  ): Promise<DesktopWikiFolderExportResult>
}

export interface SepilotDesktopApi {
  health(): Promise<DaemonHealth>
  doctor?(): Promise<DaemonDoctorReport>
  healthReport(format?: 'markdown' | 'json'): Promise<string | DaemonHealthReportSnapshot>
  saveHealthReport(content: string): Promise<string | null>
  reportRendererError(
    input: DesktopRendererErrorReportInput,
  ): Promise<DesktopRendererErrorReportResult>
  exportDiagnostics(
    input?: DesktopDiagnosticsExportInput,
  ): Promise<DesktopDiagnosticsExportResult | null>
  selectDirectory(defaultPath?: string): Promise<string | null>
  /** Optional so a renderer hot-reloaded before its Electron preload can fail closed. */
  wikiFiles?: SepilotDesktopWikiFilesApi
  computerObservationImage(path: string): Promise<string | null>
  shell: SepilotDesktopShellApi
  apps?: SepilotDesktopAppsApi
  calendar?: SepilotDesktopCalendarApi
  chat(message: string, sessionId?: string, options?: ChatOptions): Promise<DaemonChatResult>
  streamChat(
    message: string,
    sessionId: string | undefined,
    options: ChatStreamOptions | undefined,
    onEvent: (event: DaemonChatStreamPayload) => void,
  ): Promise<void>
  streamApprovalResume(
    requestId: string,
    approved: boolean | ApprovalDecisionStatus,
    sessionId: string | undefined,
    onEvent: (event: DaemonChatStreamPayload) => void,
  ): Promise<void>
  streamSessionResume(
    sessionId: string,
    onEvent: (event: DaemonChatStreamPayload) => void,
    options?: { force?: boolean },
  ): Promise<void>
  cancelActiveRun(sessionId?: string): void
  watchSession(
    sessionId: string,
    onEvent: (event: DaemonSessionWatchPayload) => void,
  ): Promise<() => void>
  watchSessions(onEvent: (event: DaemonSessionsWatchPayload) => void): Promise<() => void>
  respondApproval(
    requestId: string,
    approved: boolean | ApprovalDecisionStatus,
    options?: {
      sessionId?: string
      scope?: ApprovalScope
      rule?: ApprovalRule
      note?: string
      signal?: AbortSignal
    },
  ): Promise<DaemonApprovalResponseResult>
  answerSessionQuestion(
    sessionId: string,
    questionId: string,
    answer: string,
  ): Promise<{ answered: boolean }>
  session(id: string): Promise<DaemonSessionDetail>
  sessionArtifacts(id: string): Promise<DaemonArtifact[]>
  sessions(query?: string, options?: DaemonListSessionsOptions): Promise<DaemonSessionList>
  deleteSession(id: string): Promise<void>
  compactSession(id: string): Promise<DaemonSessionCompactResult>
  usage(): Promise<DaemonUsageSummary>
  devices(): Promise<DaemonDevice[]>
  config(): Promise<DaemonConfig>
  providers(): Promise<DaemonProviderInfo[]>
  personas(): Promise<DaemonPersona[]>
  skills(): Promise<DaemonSkill[]>
  skillSearch(query: string): Promise<DaemonSkill[]>
  skillMarketplaceSearch(
    query: string,
    options?: { marketplace?: string; limit?: number },
  ): Promise<MarketplaceSkillSearchResult[]>
  skillInstallPreview(source: string): Promise<import('./daemon/types.js').InstallSkillPreviewResponse>
  skillInstall(
    source: string,
    options?: boolean | { force?: boolean; expectedDigest?: string },
  ): Promise<{ installed: DaemonSkill[] }>
  listRecentMemory(options?: DaemonMemoryRecentOptions): Promise<DaemonMemoryRecentEntry[]>
  memorySearch(query: string, options?: DaemonMemorySearchOptions): Promise<DaemonMemoryEntry[]>
  memoryDocumentSearch(
    query: string,
    options?: DaemonMemoryDocumentSearchOptions,
  ): Promise<DaemonMemoryDocumentChunk[]>
  memoryStatus(): Promise<DaemonMemorySemanticStatus>
  memoryLifecycle(options?: DaemonMemoryLifecycleOptions): Promise<DaemonMemoryLifecycleStatus>
  memoryAudit(options?: DaemonMemoryAuditOptions): Promise<DaemonMemoryAuditEntry[]>
  memorySecurityAudit(
    options?: DaemonMemorySecurityAuditOptions,
  ): Promise<DaemonMemorySecurityAuditResult>
  memoryScopes(): Promise<DaemonMemoryScopes>
  transferMemoryScope(
    source: string,
    input: DaemonMemoryScopeTransferInput,
  ): Promise<DaemonMemoryScopeTransferResult>
  runMemoryMaintenance(input?: DaemonMemoryMaintenanceInput): Promise<DaemonMemoryMaintenanceResult>
  fileMemory(): Promise<DaemonFileMemorySnapshot>
  updateFileMemorySection(
    sectionTitle: string,
    content: string,
  ): Promise<DaemonFileMemorySectionUpdateResult>
  deleteFileMemorySection(sectionTitle: string): Promise<{ deleted: boolean }>
  listMemoryDocuments(options?: DaemonMemoryDocumentListOptions): Promise<DaemonMemoryDocument[]>
  ingestMemoryDocument(input: DaemonMemoryDocumentIngestInput): Promise<DaemonMemoryDocument>
  memoryDocument(id: string): Promise<DaemonMemoryDocument>
  deleteMemoryDocument(id: string): Promise<void>
  addMemory(
    content: string,
    tags?: string[],
    options?: DaemonMemoryAddOptions,
  ): Promise<DaemonMemoryAddResult>
  updateMemory(id: string, input: DaemonMemoryUpdateInput): Promise<DaemonMemoryEntry>
  deleteMemory(id: string, options?: DaemonMemoryDeleteOptions): Promise<void>
  reindexMemory(): Promise<DaemonMemoryReindexResult>
  updateConfig(updates: DaemonConfigUpdateInput): Promise<DaemonConfigUpdateResult>
  agents(): Promise<DaemonAgentDescriptor[]>
  projects(): Promise<DaemonProject[]>
  attachSessionToProject(projectId: string, sessionId: string): Promise<DaemonProject>
  tickets(status?: string): Promise<PaginatedResult<Ticket>>
  ticketComments(ticketId: string): Promise<GatewayTicketComment[]>
  watchTickets(
    status: string | undefined,
    onEvent: (event: GatewayTicketsWatchPayload) => void,
  ): Promise<() => void>
  watchTicketComments(
    ticketId: string,
    onEvent: (event: GatewayTicketCommentsWatchPayload) => void,
  ): Promise<() => void>
  onUpdateAvailable?(callback: (version: string) => void): void
  onUpdateDownloaded?(callback: (version: string) => void): void
  onNewChat?(callback: () => void): void
}
