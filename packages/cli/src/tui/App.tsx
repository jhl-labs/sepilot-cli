import { writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import type React from 'react'
import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import type {
  DaemonAgentDescriptor,
  DaemonProviderInfo,
  DaemonProject,
  DaemonSkill,
  MarketplaceSkillSearchResult,
} from '@sepilotd/api-client'
import {
  delegationHealthDetail,
  delegationHealthLabel,
  formatTokenSpeedStats,
} from '@sepilotd/api-client'
import { artifactLabel, artifactPreview } from '@sepilotd/presentation'
import { Box, Text, useApp, useStdout } from 'ink'
import { DaemonClient } from '../client/http.js'
import { DaemonWsClient } from '../client/ws.js'
import {
  cancelSteer,
  formatCancelledSteerNote,
  formatQueuedSteerNote,
  submitSteer,
} from '../steer-shared.js'
import { formatProviderModelBadges } from '../utils/provider-display.js'
import { recordModelUse, readModelMru } from '../utils/model-mru.js'
import {
  createModelRefreshState,
  refreshModelsInBackground,
  refreshProviderModelLists,
  remapPickerIndex,
} from '../utils/model-refresh.js'
import { normalizeError, useChat } from './hooks/useChat.js'
import { useKeybindings } from './hooks/useKeybindings.js'
import { useMouseEvents, type MouseEvent } from './hooks/useMouseEvents.js'
import { useTerminalSize } from './hooks/useTerminalSize.js'
import {
  clearProjectBinding,
  loadCliState,
  recordProjectBinding,
  removeSessionAccess,
  recordSessionAccess,
  recordRecapPreference,
  recordThemePreference,
} from './cli-state.js'
import { buildTurnRecap, type TurnRecapSnapshot } from './utils/turn-recap.js'
import { Header } from './components/Header.js'
import { ChatView, type ChatViewHandle } from './components/ChatView.js'
import { calculateInputBoxHeight, InputBox } from './components/InputBox.js'
import { StatusBar } from './components/StatusBar.js'
import { InlineStatusPanels } from './components/InlineStatusPanels.js'
import { OverlayPanels } from './components/OverlayPanels.js'
import {
  SwarmAttachView,
  type SwarmAttachExitReason,
  type SwarmAttachTarget,
} from './components/SwarmAttachView.js'
import type { ProviderSetupStep } from './components/ProviderSetupModal.js'
import { buildModePickerList, findModeMatch } from './components/ModePicker.js'
import { buildSessionPickerList } from './components/SessionPicker.js'
import { loadProjectSessionRegistry, removeProjectSession } from './project-sessions.js'
import type { AppConfig } from './types.js'
import {
  colors,
  getActiveThemeId,
  getThemeOption,
  resolveThemeId,
  setActiveTheme,
  symbols,
  themeOptions,
  type ThemeId,
} from './theme.js'
import {
  AUTONOMY_OPTIONS,
  findAutonomyOption,
  getAutonomyOption,
  nextAutonomyLevel,
  type AutonomyLevel,
} from './utils/autonomy.js'
import {
  applyAttachmentCompletion,
  extractAttachmentReferences,
  findActiveAttachmentReference,
  resolveAttachmentCandidates,
  toAttachmentPath,
} from './utils/attachments.js'
import {
  applySkillCompletion,
  filterSkillAutocompleteCandidates,
  findActiveSkillReference,
} from './utils/skill-autocomplete.js'
import { invalidateFileIndex, getFileIndex } from './utils/file-index.js'
import { recordAttachment } from './utils/attachment-history.js'
import {
  findProjectByName,
  matchProjectForWorkspace,
  type WorkspaceProject,
} from './utils/projects.js'
import { findAdjacentRecentSessionId, findRewindTarget } from './utils/session-history.js'
import { resolvePlanModeToggle } from './utils/plan-mode-toggle.js'
import {
  buildBranchSuccessSummary,
  buildContextManagementSummary,
  buildRecentResumeSummary,
  buildResumePreflightSummary,
  buildRewindPlanSummary,
  buildRewindSuccessSummary,
  buildSessionLoadedSummary,
} from './utils/context-management.js'
import { buildSessionExportFilename, parseSessionExportArgs } from './utils/session-commands.js'
import {
  deriveInlinePanelVisibility,
  getComposerBlockedReason,
  getForegroundOverlayId,
  hasForegroundOverlay,
  isTranscriptNavigationBlocked,
  type OverlayStateSnapshot,
} from './utils/overlay-state.js'
import {
  deriveComposerActivityState,
  deriveComposerPresentationState,
  shouldSubmitImmediatelyWhileBusy,
} from './utils/composer-state.js'
import { buildDoctorSummary, buildDoctorUnavailableSummary } from './utils/doctor-summary.js'
import { deriveStatusBarPresentationState } from './utils/status-bar-state.js'
import {
  contextFillPercent,
  estimateConversationTokens,
} from './utils/context-token-estimate.js'
import { buildCommandPaletteItems } from './utils/command-autocomplete.js'
import { buildStartupPreflightSummary } from './utils/startup-preflight.js'
import { copyTextToClipboard, findLatestCopyableCodeBlock } from './utils/copy.js'
import { formatConversationForCopy, formatTerminalTranscriptDump } from './utils/transcript-text.js'
import { TUI_ENTER_SEQUENCE, TUI_EXIT_SEQUENCE } from './utils/terminal-modes.js'
import {
  appendUniqueMemoryLine,
  findMemorySection,
  formatManualMemoryBacklogLine,
  removeMatchingMemoryLines,
  TUI_OPEN_LOOP_QUEUE_SECTION,
} from './utils/memory.js'
import {
  buildTuiHooksUsage,
  formatProviderHealthSummary,
  formatTuiBacklogSummary,
  formatTuiDailyMemory,
  formatTuiFileMemorySummary,
  formatTuiMemoryAuditSummary,
  formatTuiMemoryLifecycleSummary,
  formatTuiMemoryMaintenanceSummary,
  formatTuiHookDeadLetter,
  formatTuiHookDelivery,
  formatTuiHookSummary,
  parseTuiHookEvents,
  parseTuiHookHeaders,
  parseTuiHookLimit,
  parseTuiHookValueFlag,
  splitSectionAndContent,
  TUI_FILE_MEMORY_USAGE,
} from './utils/app-helpers.js'
import {
  calculateRenderedPanelRows,
  calculateListPanelHeight,
  calculateListViewportCapacity,
  fitInlinePanelHeight,
} from './utils/layout.js'
import { selectNextListIndex, selectPreviousListIndex } from './utils/input-navigation.js'
import {
  buildLocalShellInvocation,
  buildLocalShellProgressLabel,
  buildLocalShellTranscript,
  resolveLocalShellGate,
  runLocalShellCommand,
} from './utils/local-shell.js'
import { buildStreamingProgress } from './utils/streaming.js'
import { EXIT_HINT_TIMEOUT_MS, resolveCtrlCIntent } from './utils/exit-prompt.js'
import { formatSkillInstallResult, SKILLS_COMMAND_USAGE } from './utils/skill-store.js'
import {
  buildProviderModelPickerList,
  findModelMatch,
  findProviderMatch,
  type ProviderModelPickerItem,
  type ProviderModelOption,
} from './utils/provider-models.js'
import {
  buildProviderSecretEnvUpdate,
  buildProviderConfigRecord,
  buildProviderConfigUpdate,
  buildProviderDefaultUpdate,
  buildProviderDeleteUpdate,
  buildProviderModelSuggestions,
  buildProviderSetupDraft,
  buildProviderSetupSuccessMessage,
  derivePreferredProviderFallback,
  findBuiltinProviderPreset,
  isValidProviderEnvVarName,
  managedProviderApiKeyEnvVar,
  reconcileProviderModelDiscovery,
  type ConfiguredProviderRecord,
  type ProviderSetupDraft,
} from './utils/provider-setup.js'
import {
  applyModelCapabilityOverride,
  describeModelCapabilities,
  parseCapabilityToggle,
  resolveCapabilityName,
  TOGGLEABLE_MODEL_CAPABILITIES,
} from './utils/model-capabilities.js'
import { runMcpSlashCommand } from './utils/mcp-slash.js'
import { buildLeftoverContainerBanner } from './utils/leftover-containers.js'
import { dockerAvailable, listManagedContainers } from '../utils/docker.js'
import {
  formatRagSource,
  formatRagSyncResult,
  formatRagVectorInfo,
  RAG_COMMAND_USAGE,
} from './utils/rag.js'
import { SlashCommandRegistry, type SlashCommandHandler } from './commands/slash-registry.js'
import {
  DAEMON_ADMIN_SLASH_COMMANDS,
  runDaemonAdminSlashCommand,
  splitSlashCommandInput,
} from './utils/daemon-admin-slash.js'
import { USAGE_DASHBOARD_MAX_DAYS, useUsageDashboard } from './hooks/useUsageDashboard.js'
import { MEMORY_SEARCH_LIMIT, useMemorySearch } from './hooks/useMemorySearch.js'
import { RAG_SEARCH_LIMIT, useRagPanel } from './hooks/useRagPanel.js'
import { createStateBoardSlashCommand } from './hooks/useStateBoardPanel.js'
import { useAgentRoster } from './hooks/useAgentRoster.js'
import { useAttachmentState } from './hooks/useAttachmentState.js'
import { useDaemonConnection } from './hooks/useDaemonConnection.js'
import { useProjectSelection, type ProjectSelectionSource } from './hooks/useProjectSelection.js'
import { useProviderRegistry } from './hooks/useProviderRegistry.js'
import { useSessionRoster } from './hooks/useSessionRoster.js'
import { useStreamProgressTimer } from './hooks/useStreamProgressTimer.js'
import { useRawCtrlC } from './hooks/useRawCtrlC.js'
import {
  connectWsOrNull,
  detectWorkspaceOrNull,
  fetchInitialSession,
  loadAgentModesOrEmpty,
  loadDaemonConfigDefaults,
  loadProvidersOrEmpty,
} from './state/daemon-bootstrap.js'
import { loadAttachmentSuggestions } from './state/attachment-suggestions.js'
import { loadFilePickerItems } from './state/file-picker-fetch.js'
import { resolveInitialProject, sortProjects, upsertProject } from './state/project-hydration.js'
import { applyDaemonConfigToState } from './state/apply-daemon-config.js'
import { decideAutoProjectFromSession } from './state/auto-project-selection.js'
import { decideOpenOverlayGate, decideStreamApprovalGate } from './state/open-overlay-gate.js'
import {
  decideAutoOpenProviderSetup,
  decideOpenProviderDeleteConfirmation,
  decideOpenProviderSetupRequest,
  getAutoProviderSetupBannerMessage,
} from './state/open-provider-setup.js'
import { syncProjectSessionRegistry } from './state/project-session-sync.js'
import { decideSessionProjectAttach } from './state/session-project-attach.js'
import { loadSessionPickerItems } from './state/session-picker-fetch.js'
import {
  sessionBelongsToWorkspace,
  sessionWorkspaceLoadError,
} from './utils/session-workspace.js'
import { useSwarmRun } from './hooks/useSwarmRun.js'
import { useTranscriptOverlays } from './hooks/useTranscriptOverlays.js'
import { useExclusivePanel } from './hooks/useExclusivePanel.js'
import { useExitHint } from './hooks/useExitHint.js'
import { useFilePicker } from './hooks/useFilePicker.js'
import { useSessionPicker } from './hooks/useSessionPicker.js'
import { useModelPicker } from './hooks/useModelPicker.js'
import { useModePicker } from './hooks/useModePicker.js'
import { useAutonomyPicker } from './hooks/useAutonomyPicker.js'
import { useProviderSetup } from './hooks/useProviderSetup.js'
import { useSkillManager } from './hooks/useSkillManager.js'
import { SKILL_STORE_SEARCH_LIMIT, useSkillStore } from './hooks/useSkillStore.js'
import { buildStoredToolCallState } from './utils/tooling.js'
import { nextAgentModeId } from './utils/agent-modes.js'
import {
  detectOllamaModels,
  PROVIDER_WIZARD_PRESETS,
  type ProviderWizardPreset,
} from '../utils/provider-presets.js'
import {
  formatProviderHeadersInput,
  parseProviderHeadersInput,
} from '../utils/provider-http-options.js'

interface AppProps {
  config: AppConfig
}

const FILE_PICKER_ROOT_DIR = process.cwd()
const ACP_COMMAND_USAGE = [
  'ACP Integration',
  'Usage: /acp [help|status|config|opencode|codex|zed]',
  '',
  'Supported stdio JSON-RPC methods:',
  '- initialize',
  '- session/new, session/prompt, session/cancel',
  '- legacy compatibility: newThread, sendMessage, cancelThread',
  '',
  'CLI entry: sepilot acp [--url <daemon-url>]',
].join('\n')

const A2A_COMMAND_USAGE = [
  'A2A Integration',
  'Usage: /a2a [help|status|card|send]',
  '',
  'Standard endpoints:',
  '- GET /.well-known/agent-card.json',
  '- POST /api/v1/a2a',
  '',
  'JSON-RPC methods:',
  '- SendMessage, GetTask, ListTasks, CancelTask',
  '- unsupported features return A2A JSON-RPC errors: streaming, push notifications, extended cards',
].join('\n')

function buildAcpConfigSnippet(daemonUrl?: string): string {
  const args = ['acp']
  if (daemonUrl) {
    args.push('--url', daemonUrl)
  }
  return [
    'ACP editor command:',
    `  sepilot ${args.join(' ')}`,
    '',
    'Generic stdio registration:',
    JSON.stringify(
      {
        command: 'sepilot',
        args,
        env: daemonUrl ? { SEPILOTD_URL: daemonUrl } : {},
      },
      null,
      2,
    ),
  ].join('\n')
}

function buildAcpOpencodeGuide(): string {
  return [
    'opencode ACP adapter path',
    '1. opencode exposes an ACP agent with `opencode acp`.',
    '2. sepilotd exposes its own ACP server with `sepilot acp`.',
    '3. sepilotd can run opencode as an external ACP agent through the daemon route and tool `external_acp.run`.',
    '',
    'CLI: sepilot acp-agent run --agent opencode --cwd <project> "<task>"',
    'Safety: client-side fs/terminal ACP capabilities are disabled for this first adapter; permission requests are denied unless sepilotd gains an explicit approval bridge.',
  ].join('\n')
}

function buildAcpCodexGuide(): string {
  return [
    'Codex ACP adapter path',
    '1. Codex CLI does not expose a native `codex acp` command in the tested CLI version.',
    '2. sepilotd runs Codex through the stdio adapter `@agentclientprotocol/codex-acp`.',
    '   Install it with: npm install -g @agentclientprotocol/codex-acp',
    '3. Override the adapter command with environment variables when using another adapter:',
    '   SEPILOTD_CODEX_ACP_COMMAND=npm',
    '   SEPILOTD_CODEX_ACP_ARGS="exec --yes --package @agentclientprotocol/codex-acp -- codex-acp"',
    '',
    'CLI: sepilot acp-agent run --agent codex --cwd <project> "<task>"',
    'Safety: client-side fs/terminal ACP capabilities are disabled for this first adapter; permission requests are denied unless sepilotd gains an explicit approval bridge.',
  ].join('\n')
}

function buildAcpZedHint(daemonUrl?: string): string {
  return [
    'Zed / ACP registration hint',
    'Register a stdio agent that runs:',
    `  sepilot acp${daemonUrl ? ` --url ${daemonUrl}` : ''}`,
    '',
    'The daemon must already be reachable. Start it with `sepilot start` and verify with `/acp status` or `sepilot status`.',
  ].join('\n')
}

function buildA2aGuide(daemonUrl?: string): string {
  const base = daemonUrl ?? 'http://<daemon-host>:17600'
  return [
    'A2A endpoint registration',
    `Agent Card: ${base}/.well-known/agent-card.json`,
    `JSON-RPC:   ${base}/api/v1/a2a`,
    '',
    'Protocol: Agent2Agent JSON-RPC binding, A2A-Version: 1.0',
    'Outbound tool: a2a.send',
  ].join('\n')
}

export const __appTesting = {
  ACP_COMMAND_USAGE,
  A2A_COMMAND_USAGE,
  buildAcpConfigSnippet,
  buildAcpOpencodeGuide,
  buildAcpCodexGuide,
  buildAcpZedHint,
  buildA2aGuide,
}

// ProjectSelectionSource is now exported by ./hooks/useProjectSelection.
// Re-imported below where the hook is consumed.
type ProviderSetupMode = 'new' | 'edit'

interface ProviderSetupState {
  mode: ProviderSetupMode
  allowPresetSelection: boolean
  step: ProviderSetupStep
  applyToCurrentSession: boolean
  preset: ProviderWizardPreset | null
  selectedPresetIndex: number
  sourceProviderId: string | null
  providerId: string
  baseUrl: string
  apiKeyEnvVar: string
  apiKeyValue: string
  headersText: string
  model: string
  modelSuggestions: string[]
  modelSuggestionIndex: number
  modelAliasSource?: string
  configProviders: ConfiguredProviderRecord[]
  loading: boolean
  validating: boolean
  saving: boolean
  error: string | null
}

interface ProviderDeleteConfirmState {
  providerId: string
  providerName: string
  configProviders: ConfiguredProviderRecord[]
  defaultProviderId: string | null
  defaultModel: string | null
  nextProviderId: string
  nextModel: string
  deleting: boolean
  error: string | null
}

interface LocalShellCommandState {
  command: string
  startedAt: number
}

function isExitInput(text: string): boolean {
  const normalized = text.trim().toLowerCase()
  return (
    normalized === 'exit' ||
    normalized === 'quit' ||
    normalized === '/exit' ||
    normalized === '/quit' ||
    normalized === '/q'
  )
}

export function App({ config }: AppProps) {
  const { exit } = useApp()
  const { stdout } = useStdout()
  const terminalSize = useTerminalSize(stdout)
  const termHeight = terminalSize.rows
  const termWidth = terminalSize.columns

  const [httpClient] = useState(() => new DaemonClient(config.url))
  const { swarmClient, lastSwarmRunId, setLastSwarmRunId } = useSwarmRun({
    baseUrl: httpClient.baseUrl,
    token: httpClient.token,
  })
  const [wsClient, setWsClient] = useState<DaemonWsClient | null>(null)
  const {
    version,
    setVersion,
    connectionStatus,
    setConnectionStatus,
    connectionError,
    setConnectionError,
  } = useDaemonConnection()
  const [themeId, setThemeId] = useState<ThemeId>(() => getActiveThemeId())
  const [inputValue, setInputValue] = useState('')
  const queuedFollowUpsRef = useRef<Array<{ sessionId: string; noteId: string }>>([])
  const followUpCancelInFlightRef = useRef(false)
  const [swarmAttachTarget, setSwarmAttachTarget] = useState<SwarmAttachTarget | null>(null)
  const exitHintApi = useExitHint(EXIT_HINT_TIMEOUT_MS)
  const { exitHint, clear: clearExitHint, show: showExitHint } = exitHintApi
  const { primaryAgentId, setPrimaryAgentId, agentModes, setAgentModes } = useAgentRoster()
  const [paletteIndex, setPaletteIndex] = useState(0)
  const {
    sessionPickerOpen,
    setSessionPickerOpen,
    sessionQuery,
    setSessionQuery,
    sessionPickerIndex,
    setSessionPickerIndex,
    sessionItems,
    setSessionItems,
    sessionsLoading,
    setSessionsLoading,
    sessionsError,
    setSessionsError,
    closeSessionPicker,
  } = useSessionPicker()
  const [sessionPickerRetryToken, setSessionPickerRetryToken] = useState(0)
  const {
    skillStoreOpen,
    setSkillStoreOpen,
    skillStoreQuery,
    setSkillStoreQuery,
    skillStoreSearchedQuery,
    setSkillStoreSearchedQuery,
    skillStoreResults,
    setSkillStoreResults,
    skillStoreIndex,
    setSkillStoreIndex,
    skillStoreLoading,
    setSkillStoreLoading,
    skillStoreInstallingSource,
    setSkillStoreInstallingSource,
    skillStoreError,
    setSkillStoreError,
    skillStoreMessage,
    setSkillStoreMessage,
    skillStoreRequestRef,
    skillStoreInstallRequestRef,
    closeSkillStorePicker,
  } = useSkillStore()
  const {
    skillManagerOpen,
    setSkillManagerOpen,
    skillManagerQuery,
    setSkillManagerQuery,
    skillManagerIndex,
    setSkillManagerIndex,
    skillManagerTogglingId,
    setSkillManagerTogglingId,
    skillManagerError,
    setSkillManagerError,
    skillManagerMessage,
    setSkillManagerMessage,
    closeSkillManagerPicker,
  } = useSkillManager()
  const { projectSessionIds, setProjectSessionIds, recentSessionIds, setRecentSessionIds } =
    useSessionRoster()
  const {
    providers,
    setProviders,
    daemonDefaultProvider,
    setDaemonDefaultProvider,
    daemonDefaultModel,
    setDaemonDefaultModel,
  } = useProviderRegistry()
  const {
    modelPickerOpen,
    setModelPickerOpen,
    modelQuery,
    setModelQuery,
    modelPickerIndex,
    setModelPickerIndex,
    providersLoading,
    setProvidersLoading,
    providersError,
    setProvidersError,
    closeModelPicker,
  } = useModelPicker()
  const [modelMru, setModelMru] = useState<string[]>([])
  const {
    providerSetup,
    setProviderSetup,
    providerDeleteConfirm,
    setProviderDeleteConfirm,
    providerSetupOpen,
    providerDeleteOpen,
    closeProviderSetup,
    closeProviderDeleteConfirm,
  } = useProviderSetup<ProviderSetupState, ProviderDeleteConfirmState>()
  const {
    modePickerOpen,
    setModePickerOpen,
    modeQuery,
    setModeQuery,
    modePickerIndex,
    setModePickerIndex,
    agentModesLoading,
    setAgentModesLoading,
    agentModesError,
    setAgentModesError,
    closeModePicker,
  } = useModePicker()
  const {
    autonomyPickerOpen,
    setAutonomyPickerOpen,
    autonomyPickerIndex,
    setAutonomyPickerIndex,
    closeAutonomyPicker,
  } = useAutonomyPicker()
  const {
    projects,
    setProjects,
    workspaceProject,
    setWorkspaceProject,
    projectSelectionSource,
    setProjectSelectionSource,
  } = useProjectSelection()
  const {
    attachmentSuggestions,
    setAttachmentSuggestions,
    attachmentSuggestionIndex,
    setAttachmentSuggestionIndex,
    attachmentReindexNotice,
    setAttachmentReindexNotice,
    fileIndexPaths,
    setFileIndexPaths,
    queuedAttachmentPaths,
    setQueuedAttachmentPaths,
  } = useAttachmentState()
  const [installedSkills, setInstalledSkills] = useState<DaemonSkill[]>([])
  const [installedSkillsLoading, setInstalledSkillsLoading] = useState(false)
  const [installedSkillsError, setInstalledSkillsError] = useState<string | null>(null)
  const [skillAutocompleteIndex, setSkillAutocompleteIndex] = useState(0)
  const {
    filePickerOpen,
    setFilePickerOpen,
    filePickerDir,
    setFilePickerDir,
    filePickerItems,
    setFilePickerItems,
    filePickerIndex,
    setFilePickerIndex,
    filePickerLoading,
    setFilePickerLoading,
    filePickerError,
    setFilePickerError,
    closeFilePicker,
  } = useFilePicker(FILE_PICKER_ROOT_DIR)
  const {
    memorySearchOpen,
    setMemorySearchOpen,
    memorySearchQuery,
    setMemorySearchQuery,
    memorySearchResults,
    setMemorySearchResults,
    memorySearchStatus,
    setMemorySearchStatus,
    memorySearchLoading,
    setMemorySearchLoading,
    memorySearchError,
    setMemorySearchError,
    closeMemorySearchPanel,
  } = useMemorySearch()
  const {
    ragPanelOpen,
    setRagPanelOpen,
    ragQuery,
    setRagQuery,
    ragSources,
    setRagSources,
    ragHits,
    setRagHits,
    ragVectorInfo,
    setRagVectorInfo,
    ragSyncResult,
    setRagSyncResult,
    ragSelectedIndex,
    setRagSelectedIndex,
    ragLoading,
    setRagLoading,
    ragError,
    setRagError,
    closeRagPanel,
  } = useRagPanel()
  const {
    helpOpen,
    setHelpOpen,
    showMcp,
    setShowMcp,
    transcriptClearedAt,
    setTranscriptClearedAt,
  } = useTranscriptOverlays()
  const {
    usageDashboardOpen,
    setUsageDashboardOpen,
    usageDashboardDays,
    setUsageDashboardDays,
    usageDashboardLoading,
    setUsageDashboardLoading,
    usageDashboardError,
    setUsageDashboardError,
    usageSummary,
    setUsageSummary,
    usageDaily,
    setUsageDaily,
    usageBuckets,
    closeUsageDashboard,
  } = useUsageDashboard()
  const exclusivePanel = useExclusivePanel()
  // Every overlay-hook close handler is registered with the exclusivity
  // coordinator. When loadUsageDashboard / loadMemorySearch (or any future
  // load that opens a panel) calls exclusivePanel.begin(), the coordinator
  // dismisses every other registered panel automatically — so opening a
  // dashboard now also collapses the file/session/model/mode/autonomy/
  // provider-setup pickers if they happened to be open. The legacy
  // OverlayState block messages stay in place for the manual open paths
  // (Ctrl+S, /files, etc.) that don't go through begin().
  useEffect(() => {
    exclusivePanel.register('usage', closeUsageDashboard)
    exclusivePanel.register('memory', closeMemorySearchPanel)
    exclusivePanel.register('rag', closeRagPanel)
    exclusivePanel.register('file', closeFilePicker)
    exclusivePanel.register('session', closeSessionPicker)
    exclusivePanel.register('skill-manager', closeSkillManagerPicker)
    exclusivePanel.register('skill-store', closeSkillStorePicker)
    exclusivePanel.register('model', closeModelPicker)
    exclusivePanel.register('mode', closeModePicker)
    exclusivePanel.register('autonomy', closeAutonomyPicker)
    exclusivePanel.register('provider-setup', closeProviderSetup)
    exclusivePanel.register('provider-delete', closeProviderDeleteConfirm)
  }, [
    exclusivePanel,
    closeAutonomyPicker,
    closeFilePicker,
    closeRagPanel,
    closeSkillManagerPicker,
    closeSkillStorePicker,
    closeMemorySearchPanel,
    closeModePicker,
    closeModelPicker,
    closeProviderDeleteConfirm,
    closeProviderSetup,
    closeSessionPicker,
    closeUsageDashboard,
  ])
  const [localShellCommand, setLocalShellCommand] = useState<LocalShellCommandState | null>(null)
  const lastLocalShellCommandRef = useRef<string | null>(null)
  const localShellAborterRef = useRef<AbortController | null>(null)
  // Ctrl+O terminal transcript mode. The TUI normally owns the alternate
  // screen, so Ctrl+O temporarily leaves it and writes the whole conversation
  // into the host terminal scrollback for drag-select copy.
  const [copyView, setCopyView] = useState(false)
  const [activityPanelHidden, setActivityPanelHidden] = useState(false)
  // True while the approval modal's inline comment editor is open; suspends
  // the global y/s/a/n/Esc shortcuts so typed feedback isn't misread as a
  // decision (or a run cancel).
  const [approvalCommentActive, setApprovalCommentActive] = useState(false)
  // End-of-turn recap (duration, tool calls, files, tokens). Default on;
  // toggled with /recap and persisted in cli-state.json.
  const [recapEnabled, setRecapEnabled] = useState(true)
  const recapSnapshotRef = useRef<TurnRecapSnapshot | null>(null)
  const pendingTerminalTranscriptRef = useRef<null | {
    transcript: string
    messageCount: number
  }>(null)
  const chatViewRef = useRef<ChatViewHandle | null>(null)
  const workspaceProjectRef = useRef<WorkspaceProject | null>(null)
  const lastAttachedProjectSessionRef = useRef<string | null>(null)
  const compactSuggestedForSessionRef = useRef<string | null>(null)
  const startupPreflightKeyRef = useRef<string | null>(null)
  const autoProviderSetupKeyRef = useRef<string | null>(null)
  const installedSkillsRequestRef = useRef(0)
  const {
    state,
    dispatch,
    sendMessage,
    loadSession,
    resolveApproval,
    answerQuestion,
    resumeSession,
    cancelStream,
    getTokenSpeedStats,
    resetTokenSpeedStats,
  } = useChat(wsClient, httpClient)
  const streamProgressNow = useStreamProgressTimer(state.isStreaming || localShellCommand !== null)
  // Capture a snapshot when a run starts and emit a one-shot recap system
  // message when it ends. The snapshot ref doubles as the edge detector.
  useEffect(() => {
    if (state.isStreaming) {
      if (!recapSnapshotRef.current) {
        recapSnapshotRef.current = {
          startedAt: Date.now(),
          toolCallCount: state.toolCalls.length,
        }
      }
      return
    }
    const snapshot = recapSnapshotRef.current
    recapSnapshotRef.current = null
    if (!snapshot || !recapEnabled) return
    const recap = buildTurnRecap(snapshot, state.toolCalls, state.usage, Date.now())
    if (recap) {
      dispatch({ type: 'SYSTEM_MESSAGE', content: recap })
    }
  }, [state.isStreaming])
  const overlayState = useMemo<OverlayStateSnapshot>(
    () => ({
      helpOpen,
      autonomyPickerOpen,
      providerSetupOpen,
      providerDeleteOpen,
      modelPickerOpen,
      modePickerOpen,
      filePickerOpen,
      sessionPickerOpen,
      skillManagerOpen,
      skillStoreOpen,
    }),
    [
      autonomyPickerOpen,
      filePickerOpen,
      helpOpen,
      modePickerOpen,
      modelPickerOpen,
      providerDeleteOpen,
      providerSetupOpen,
      sessionPickerOpen,
      skillManagerOpen,
      skillStoreOpen,
    ],
  )
  const composerOverlayBlocked = hasForegroundOverlay(overlayState)
  const composerBlockedReason = getComposerBlockedReason(overlayState)
  const isBangInput = inputValue.startsWith('!')
  const showPalette = inputValue.startsWith('/')
  const activeSkillReference = useMemo(
    () => (showPalette || isBangInput ? null : findActiveSkillReference(inputValue)),
    [inputValue, isBangInput, showPalette],
  )
  const skillAutocompleteCandidates = useMemo(
    () =>
      activeSkillReference
        ? filterSkillAutocompleteCandidates(installedSkills, activeSkillReference.query)
        : [],
    [activeSkillReference, installedSkills],
  )
  const showCommandPalette = showPalette && !composerOverlayBlocked && !state.pendingApproval
  const showAttachmentPalette =
    !showPalette &&
    !isBangInput &&
    !activeSkillReference &&
    attachmentSuggestions.length > 0 &&
    !composerOverlayBlocked &&
    !state.pendingApproval
  const showSkillPalette =
    Boolean(activeSkillReference) &&
    !showCommandPalette &&
    !composerOverlayBlocked &&
    !state.pendingApproval
  const shortcutInputBlocked = (
    composerOverlayBlocked
    || showCommandPalette
    || showAttachmentPalette
    || showSkillPalette
  )
  useMouseEvents((event: MouseEvent) => {
    if (event.kind !== 'wheel-up' && event.kind !== 'wheel-down') return
    if (copyView) {
      return
    }
    if (
      isTranscriptNavigationBlocked(overlayState, {
        showCommandPalette,
        showAttachmentPalette,
        showSkillPalette,
        pendingApproval: Boolean(state.pendingApproval),
      })
    )
      return
    if (event.kind === 'wheel-up') chatViewRef.current?.scrollUp(3)
    else chatViewRef.current?.scrollDown(3)
  })
  const attachmentReferences = useMemo(
    () =>
      showPalette || isBangInput ? [] : extractAttachmentReferences(inputValue, fileIndexPaths),
    [fileIndexPaths, inputValue, isBangInput, showPalette],
  )
  const activeAttachmentReference = useMemo(
    () =>
      showPalette || isBangInput ? null : findActiveAttachmentReference(inputValue, fileIndexPaths),
    [fileIndexPaths, inputValue, isBangInput, showPalette],
  )
  const pendingAttachmentPaths = useMemo(
    () =>
      Array.from(
        new Set([
          ...queuedAttachmentPaths,
          ...attachmentReferences.map((attachment) => attachment.path),
        ]),
      ),
    [attachmentReferences, queuedAttachmentPaths],
  )
  useEffect(() => {
    setSkillAutocompleteIndex(0)
  }, [activeSkillReference?.query, skillAutocompleteCandidates.length])
  const streamingProgress = useMemo(
    () =>
      buildStreamingProgress({
        isStreaming: state.isStreaming,
        streamStatus: state.streamStatus,
        streamStartedAt: state.providerWait?.startedAt ?? state.streamStartedAt,
        now: streamProgressNow,
        currentMessage: state.currentMessage,
        liveOutputTokens: state.usage.output,
        hasPendingApproval: Boolean(state.pendingApproval),
        hasPendingQuestion: state.pendingQuestions.length > 0,
      }),
    [
      state.currentMessage,
      state.isStreaming,
      state.pendingApproval,
      state.pendingQuestions.length,
      state.streamStartedAt,
      state.providerWait?.startedAt,
      state.streamStatus,
      state.usage.output,
      streamProgressNow,
    ],
  )
  const localShellProgressLabel = useMemo(
    () =>
      localShellCommand
        ? buildLocalShellProgressLabel(
            localShellCommand.command,
            localShellCommand.startedAt,
            streamProgressNow,
          )
        : null,
    [localShellCommand, streamProgressNow],
  )
  const { inputBusy, activeProgressLabel } = useMemo(
    () =>
      deriveComposerActivityState({
        isStreaming: state.isStreaming,
        streamLabel: streamingProgress?.label ?? state.streamStatus,
        localShellActive: localShellCommand !== null,
        localShellProgressLabel,
      }),
    [
      localShellCommand,
      localShellProgressLabel,
      state.isStreaming,
      state.streamStatus,
      streamingProgress,
    ],
  )
  const composerPresentation = useMemo(
    () =>
      deriveComposerPresentationState({
        composerBlockedReason,
        pendingApprovalToolName: state.pendingApproval?.toolName ?? null,
        localShellActive: localShellCommand !== null,
        hydratingSession: state.isHydratingSession,
        showPalette,
        isBangInput,
        attachmentSuggestionCount: attachmentSuggestions.length,
        skillAutocompleteActive: Boolean(activeSkillReference),
        composerOverlayBlocked,
        pendingAttachmentPaths,
      }),
    [
      activeSkillReference,
      attachmentSuggestions.length,
      composerBlockedReason,
      composerOverlayBlocked,
      isBangInput,
      localShellCommand,
      pendingAttachmentPaths,
      showPalette,
      state.isHydratingSession,
      state.pendingApproval,
    ],
  )
  const currentModelInfo = useMemo(() => {
    if (!state.provider || !state.model) return null
    const provider = providers.find((entry) => entry.id === state.provider)
    return provider?.models.find((entry) => entry.id === state.model) ?? null
  }, [providers, state.model, state.provider])
  const currentModelContextWindow =
    currentModelInfo?.contextWindow && currentModelInfo.contextWindow > 0
      ? currentModelInfo.contextWindow
      : null
  const currentModelMaxOutputTokens =
    currentModelInfo?.maxOutputTokens && currentModelInfo.maxOutputTokens > 0
      ? currentModelInfo.maxOutputTokens
      : null
  const currentContextTokens = state.contextUsage?.inputTokens ?? null
  const conversationContextEstimate = useMemo(
    () => state.messages.length > 0
      ? estimateConversationTokens(state.messages)
      : null,
    [state.messages],
  )
  const displayedContextTokens = currentContextTokens ?? conversationContextEstimate
  const effectiveContextWindow =
    state.contextUsage?.contextWindowTokens ?? currentModelContextWindow
  const contextEstimated = state.contextUsage?.source === 'estimated'
    || (state.contextUsage == null && conversationContextEstimate != null)
  const currentContextPercent = useMemo(
    () => currentContextTokens == null
      ? null
      : contextFillPercent(currentContextTokens, effectiveContextWindow),
    [currentContextTokens, effectiveContextWindow],
  )

  const copyTranscriptText = useMemo(
    () => formatConversationForCopy(state.messages, state.sessionId),
    [state.messages, state.sessionId],
  )
  const closeCopyView = useCallback(() => {
    pendingTerminalTranscriptRef.current = null
    try {
      stdout.write(TUI_ENTER_SEQUENCE)
    } catch {
      // Best-effort terminal restoration; the process may already be exiting.
    }
    setCopyView(false)
  }, [stdout])
  const openCopyView = useCallback(() => {
    pendingTerminalTranscriptRef.current = {
      transcript: copyTranscriptText,
      messageCount: state.messages.length,
    }
    setCopyView(true)
  }, [copyTranscriptText, state.messages.length])
  useEffect(() => {
    if (!copyView) return
    const pending = pendingTerminalTranscriptRef.current
    if (!pending) return
    pendingTerminalTranscriptRef.current = null
    try {
      stdout.write(TUI_EXIT_SEQUENCE)
      stdout.write(formatTerminalTranscriptDump(pending.transcript, pending.messageCount))
    } catch {
      // The clipboard fallback still gives the user a way to paste elsewhere.
    }
    void copyTextToClipboard(pending.transcript).catch(() => undefined)
  }, [copyView, stdout])
  const toggleCopyView = useCallback(() => {
    if (copyView) {
      closeCopyView()
    } else {
      openCopyView()
    }
  }, [copyView, closeCopyView, openCopyView])
  const closeSwarmAttach = useCallback(
    (reason: SwarmAttachExitReason, message?: string) => {
      setSwarmAttachTarget(null)
      const label =
        reason === 'cancelled'
          ? 'cancelled'
          : reason === 'ended'
            ? 'ended'
            : reason === 'error'
              ? 'error'
              : 'detached'
      dispatch({
        type: 'SYSTEM_MESSAGE',
        content: `swarm attach ${label}${message ? `: ${message}` : ''}`,
      })
    },
    [dispatch],
  )
  // Terminal transcript mode is wired through useKeybindings's single
  // useInput so Esc/Ctrl+O can restore the alternate screen cleanly.

  const commandPaletteItems = useMemo(
    () =>
      showPalette
        ? buildCommandPaletteItems({
            input: inputValue,
            agentModes,
            providers,
            projects,
            skills: installedSkills,
            currentMode: state.mode,
            currentProvider: state.provider,
            currentModel: state.model,
            defaultProvider: daemonDefaultProvider,
            defaultModel: daemonDefaultModel,
            currentProjectName: state.projectName,
            currentSessionId: state.sessionId,
            currentThemeId: themeId,
            contextPercent: currentContextPercent,
            hasPendingApproval: Boolean(state.pendingApproval),
            isStreaming: inputBusy,
            lastSwarmRunId,
          })
        : [],
    [
      agentModes,
      daemonDefaultModel,
      daemonDefaultProvider,
      inputValue,
      currentContextPercent,
      inputBusy,
      installedSkills,
      lastSwarmRunId,
      providers,
      projects,
      showPalette,
      state.model,
      state.mode,
      state.pendingApproval,
      state.projectName,
      state.provider,
      state.sessionId,
      themeId,
    ],
  )
  const selectedProject = useMemo(
    () =>
      state.projectId ? (projects.find((project) => project.id === state.projectId) ?? null) : null,
    [projects, state.projectId],
  )
  const statusBarPresentation = useMemo(
    () =>
      deriveStatusBarPresentationState({
        sessionId: state.sessionId,
        mode: state.mode,
        autonomy: state.autonomy,
        primaryAgentId,
        usage: state.usage,
        provider: state.provider,
        model: state.model,
        contextInputTokens: displayedContextTokens,
        contextWindow: effectiveContextWindow,
        contextEstimated,
        maxOutputTokens: currentModelMaxOutputTokens,
        isStreaming: inputBusy,
        streamLabel: activeProgressLabel,
        hasPendingApproval: Boolean(state.pendingApproval),
        currentPhase: state.currentPhase,
        stateBoardCounts: state.stateBoardCounts,
      }),
    [
      activeProgressLabel,
      contextEstimated,
      displayedContextTokens,
      effectiveContextWindow,
      currentModelMaxOutputTokens,
      inputBusy,
      primaryAgentId,
      state.autonomy,
      state.currentPhase,
      state.mode,
      state.model,
      state.pendingApproval,
      state.provider,
      state.sessionId,
      state.stateBoardCounts,
      state.usage,
    ],
  )
  const helpModalContext = useMemo(
    () => ({
      sessionId: state.sessionId,
      projectName: state.projectName,
      provider: state.provider,
      model: state.model,
      mode: String(state.mode),
      autonomyLabel: getAutonomyOption(state.autonomy).label,
      contextPercent: currentContextPercent,
      providerCount: providers.length,
      activeProgressLabel,
      pendingApprovalToolName: state.pendingApproval?.toolName ?? null,
      hasPendingApproval: Boolean(state.pendingApproval),
      isStreaming: inputBusy,
    }),
    [
      activeProgressLabel,
      currentContextPercent,
      inputBusy,
      providers.length,
      state.autonomy,
      state.mode,
      state.model,
      state.pendingApproval,
      state.projectName,
      state.provider,
      state.sessionId,
    ],
  )

  // Suggest /compact once per session when context usage crosses 80%.
  useEffect(() => {
    if (
      currentContextTokens == null ||
      !effectiveContextWindow ||
      effectiveContextWindow <= 0 ||
      !state.sessionId ||
      state.isStreaming
    ) {
      return
    }
    const pct = (currentContextTokens / effectiveContextWindow) * 100
    if (pct >= 80 && compactSuggestedForSessionRef.current !== state.sessionId) {
      compactSuggestedForSessionRef.current = state.sessionId
      dispatch({
        type: 'SYSTEM_MESSAGE',
        content:
          'Context usage is above 80%. Consider running /compact to summarize the conversation and free up context space.',
      })
    }
  }, [currentContextTokens, dispatch, effectiveContextWindow, state.isStreaming, state.sessionId])

  const modelPickerList = useMemo(
    () =>
      buildProviderModelPickerList(providers, state.provider, state.model, modelQuery, {
        defaultProviderId: daemonDefaultProvider,
        defaultModelId: daemonDefaultModel,
        mru: modelMru,
      }),
    [
      daemonDefaultModel,
      daemonDefaultProvider,
      modelMru,
      modelQuery,
      providers,
      state.model,
      state.provider,
    ],
  )
  const modePickerList = useMemo(
    () => buildModePickerList(agentModes, state.mode, modeQuery),
    [agentModes, modeQuery, state.mode],
  )
  const sessionPickerList = useMemo(
    () =>
      buildSessionPickerList(
        sessionItems,
        projectSessionIds,
        state.sessionId,
        recentSessionIds,
        sessionQuery,
      ),
    [projectSessionIds, recentSessionIds, sessionItems, sessionQuery, state.sessionId],
  )

  const applyThemeSelection = useCallback(
    (nextThemeId: ThemeId, source: 'command' | 'startup' = 'command') => {
      setActiveTheme(nextThemeId)
      setThemeId(nextThemeId)
      void recordThemePreference(nextThemeId).catch(() => {})

      if (source === 'command') {
        const theme = getThemeOption(nextThemeId)
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Theme → ${theme.label} (${theme.id})`,
        })
      }
    },
    [dispatch],
  )

  const applyProjectSelection = useCallback(
    (
      project: DaemonProject | null,
      source: ProjectSelectionSource | null,
      workspaceRoot = workspaceProjectRef.current?.rootDir,
    ) => {
      dispatch({
        type: 'SET_PROJECT',
        projectId: project?.id ?? null,
        projectName: project?.name ?? null,
      })
      setProjectSelectionSource(source)

      if (workspaceRoot) {
        if (project?.id) {
          void recordProjectBinding(workspaceRoot, project.id).catch(() => {})
        } else {
          void clearProjectBinding(workspaceRoot).catch(() => {})
        }
      }
    },
    [dispatch],
  )

  const refreshProjectSessionIds = useCallback(
    async (workspace: WorkspaceProject | null = workspaceProjectRef.current): Promise<string[]> => {
      if (!workspace) {
        setProjectSessionIds([])
        return []
      }

      try {
        const registry = await loadProjectSessionRegistry(workspace)
        const nextIds = registry.sessions.map((session) => session.sessionId)
        setProjectSessionIds(nextIds)
        return nextIds
      } catch {
        setProjectSessionIds([])
        return []
      }
    },
    [],
  )

  const loadAgentModes = useCallback(async () => {
    setAgentModesLoading(true)
    setAgentModesError(null)
    try {
      setAgentModes(await httpClient.agents())
    } catch (error) {
      setAgentModes([])
      setAgentModesError(error instanceof Error ? error.message : 'Failed to load agent modes.')
    } finally {
      setAgentModesLoading(false)
    }
  }, [httpClient])

  const modelRefreshStateRef = useRef(createModelRefreshState())

  const loadProviders = useCallback(async (): Promise<DaemonProviderInfo[]> => {
    setProvidersLoading(true)
    setProvidersError(null)
    try {
      const nextProviders = await httpClient.providers()
      setProviders(nextProviders)
      return nextProviders
    } catch (error) {
      setProviders([])
      setProvidersError(error instanceof Error ? error.message : 'Failed to load providers.')
      throw error
    } finally {
      setProvidersLoading(false)
    }
  }, [httpClient])

  const ensureProvidersLoaded = useCallback(async (): Promise<DaemonProviderInfo[]> => {
    if (providers.length > 0) {
      return providers
    }
    return loadProviders()
  }, [loadProviders, providers])

  const refreshInstalledSkills = useCallback(async (): Promise<DaemonSkill[]> => {
    const requestId = installedSkillsRequestRef.current + 1
    installedSkillsRequestRef.current = requestId
    setInstalledSkillsLoading(true)
    setInstalledSkillsError(null)

    try {
      const skills = await httpClient.skills({
        includeDisabled: true,
        cwd: FILE_PICKER_ROOT_DIR,
        workspaceRoot: FILE_PICKER_ROOT_DIR,
      })
      if (installedSkillsRequestRef.current !== requestId) {
        return skills
      }
      setInstalledSkills(skills)
      return skills
    } catch (error) {
      if (installedSkillsRequestRef.current === requestId) {
        setInstalledSkills([])
        setInstalledSkillsError(
          error instanceof Error ? error.message : 'Failed to load installed skills.',
        )
      }
      return []
    } finally {
      if (installedSkillsRequestRef.current === requestId) {
        setInstalledSkillsLoading(false)
      }
    }
  }, [httpClient])

  useEffect(() => {
    void refreshInstalledSkills()
  }, [refreshInstalledSkills])

  const loadProviderConfigSnapshot = useCallback(async () => {
    const daemonConfig = await httpClient.config()
    const configProviders = daemonConfig.providers as ConfiguredProviderRecord[]
    const defaultProviderId =
      daemonConfig.agent.defaultProvider ??
      configProviders.find((provider) => provider.default)?.id ??
      configProviders[0]?.id ??
      null
    const defaultModel =
      (defaultProviderId
        ? defaultProviderId === daemonConfig.agent.defaultProvider
          ? daemonConfig.agent.defaultModel
          : configProviders.find((provider) => provider.id === defaultProviderId)?.models?.[0]
        : null) ?? null

    return {
      configProviders,
      defaultProviderId,
      defaultModel,
    }
  }, [httpClient])

  const loadMemorySearch = useCallback(
    async (query: string) => {
      // Acquiring exclusivity dismisses the usage dashboard and bumps its
      // in-flight request id so any pending fetch ignores its own response.
      const ticket = exclusivePanel.begin('memory')
      setMemorySearchOpen(true)
      setMemorySearchQuery(query)
      setMemorySearchLoading(true)
      setMemorySearchError(null)
      setMemorySearchResults([])
      setMemorySearchStatus(null)

      const [resultsResult, statusResult] = await Promise.allSettled([
        httpClient.searchMemory(query, { type: 'hybrid', limit: MEMORY_SEARCH_LIMIT }),
        httpClient.memoryStatus(),
      ])

      if (!ticket.isCurrent()) return

      if (resultsResult.status === 'fulfilled') {
        setMemorySearchResults(resultsResult.value)
      } else {
        setMemorySearchError(
          resultsResult.reason instanceof Error
            ? resultsResult.reason.message
            : 'Failed to search memory.',
        )
      }

      if (statusResult.status === 'fulfilled') {
        setMemorySearchStatus(statusResult.value)
      }

      setMemorySearchLoading(false)
    },
    [
      exclusivePanel,
      httpClient,
      setMemorySearchError,
      setMemorySearchLoading,
      setMemorySearchOpen,
      setMemorySearchQuery,
      setMemorySearchResults,
      setMemorySearchStatus,
    ],
  )

  const loadRagPanel = useCallback(
    async (query = '') => {
      const trimmedQuery = query.trim()
      const ticket = exclusivePanel.begin('rag')
      setRagPanelOpen(true)
      setRagQuery(trimmedQuery)
      setRagLoading(true)
      setRagError(null)
      setRagHits([])

      const [sourcesResult, vectorResult, hitsResult] = await Promise.allSettled([
        httpClient.ragFolders(),
        httpClient.ragVectorDbInfo(),
        trimmedQuery ? httpClient.searchRag(trimmedQuery, RAG_SEARCH_LIMIT) : Promise.resolve([]),
      ])

      if (!ticket.isCurrent()) return

      if (sourcesResult.status === 'fulfilled') {
        setRagSources(sourcesResult.value)
        setRagSelectedIndex((current) =>
          sourcesResult.value.length === 0
            ? 0
            : Math.max(0, Math.min(current, sourcesResult.value.length - 1)),
        )
      } else {
        setRagError(
          sourcesResult.reason instanceof Error
            ? sourcesResult.reason.message
            : 'Failed to load RAG sources.',
        )
        setRagSources([])
      }

      if (vectorResult.status === 'fulfilled') {
        setRagVectorInfo(vectorResult.value)
      } else {
        setRagVectorInfo(null)
      }

      if (hitsResult.status === 'fulfilled') {
        setRagHits(hitsResult.value)
      } else {
        setRagError(
          hitsResult.reason instanceof Error ? hitsResult.reason.message : 'Failed to search RAG.',
        )
        setRagHits([])
      }

      setRagLoading(false)
    },
    [
      exclusivePanel,
      httpClient,
      setRagError,
      setRagHits,
      setRagLoading,
      setRagPanelOpen,
      setRagQuery,
      setRagSelectedIndex,
      setRagSources,
      setRagVectorInfo,
    ],
  )

  const syncRagPanel = useCallback(async () => {
    exclusivePanel.begin('rag')
    setRagPanelOpen(true)
    setRagLoading(true)
    setRagError(null)
    try {
      const result = await httpClient.syncRag()
      setRagSyncResult(result)
      await loadRagPanel(ragQuery)
      dispatch({
        type: 'SYSTEM_MESSAGE',
        content: formatRagSyncResult(result),
      })
    } catch (error) {
      setRagError(error instanceof Error ? error.message : 'Failed to sync RAG.')
    } finally {
      setRagLoading(false)
    }
  }, [
    dispatch,
    exclusivePanel,
    httpClient,
    loadRagPanel,
    ragQuery,
    setRagError,
    setRagLoading,
    setRagPanelOpen,
    setRagSyncResult,
  ])

  const addRagSourceFromSlash = useCallback(
    async (sourcePath: string, name?: string) => {
      const resolvedPath = resolve(sourcePath)
      const folder = await httpClient.upsertRagFolder({
        name: name?.trim() || resolvedPath.split(/[\\/]/).filter(Boolean).pop() || resolvedPath,
        path: resolvedPath,
        sourceType: 'git',
      })
      const result = await httpClient.syncRag()
      setRagSyncResult(result)
      await loadRagPanel('')
      dispatch({
        type: 'SYSTEM_MESSAGE',
        content: [`RAG source added: ${folder.name}`, formatRagSyncResult(result)].join('\n'),
      })
    },
    [dispatch, httpClient, loadRagPanel, setRagSyncResult],
  )

  const loadUsageDashboard = useCallback(
    async (days = usageDashboardDays) => {
      const ticket = exclusivePanel.begin('usage')
      setUsageDashboardOpen(true)
      setUsageDashboardDays(days)
      setUsageDashboardLoading(true)
      setUsageDashboardError(null)
      setUsageSummary(null)
      setUsageDaily([])

      try {
        const [summary, daily] = await Promise.all([
          httpClient.usage(),
          httpClient.usageDaily(days),
        ])
        if (!ticket.isCurrent()) return
        setUsageSummary(summary)
        setUsageDaily(daily)
      } catch (error) {
        if (!ticket.isCurrent()) return
        setUsageDashboardError(error instanceof Error ? error.message : 'Failed to load usage.')
      } finally {
        if (ticket.isCurrent()) {
          setUsageDashboardLoading(false)
        }
      }
    },
    [
      exclusivePanel,
      httpClient,
      setUsageDaily,
      setUsageDashboardDays,
      setUsageDashboardError,
      setUsageDashboardLoading,
      setUsageDashboardOpen,
      setUsageSummary,
      usageDashboardDays,
    ],
  )

  const searchSkillStore = useCallback(
    async (query: string) => {
      const trimmedQuery = query.trim()
      if (!trimmedQuery) {
        setSkillStoreOpen(true)
        setSkillStoreQuery('')
        setSkillStoreSearchedQuery('')
        setSkillStoreResults([])
        setSkillStoreIndex(0)
        setSkillStoreError(null)
        setSkillStoreMessage('Type a query and press Enter to search configured skill sources.')
        return
      }

      const ticket = exclusivePanel.begin('skill-store')
      const requestId = skillStoreRequestRef.current + 1
      skillStoreRequestRef.current = requestId
      setSkillStoreOpen(true)
      setSkillStoreQuery(trimmedQuery)
      setSkillStoreLoading(true)
      setSkillStoreError(null)
      setSkillStoreMessage(null)
      setSkillStoreResults([])
      setSkillStoreIndex(0)

      try {
        const results = await httpClient.searchMarketplaceSkills(trimmedQuery, {
          limit: SKILL_STORE_SEARCH_LIMIT,
        })
        if (!ticket.isCurrent() || skillStoreRequestRef.current !== requestId) {
          return
        }
        setSkillStoreResults(results)
        setSkillStoreSearchedQuery(trimmedQuery)
        setSkillStoreMessage(
          results.length > 0
            ? `${results.length} result${results.length === 1 ? '' : 's'} for "${trimmedQuery}".`
            : null,
        )
      } catch (error) {
        if (!ticket.isCurrent() || skillStoreRequestRef.current !== requestId) {
          return
        }
        setSkillStoreError(
          error instanceof Error ? error.message : 'Failed to search skill sources.',
        )
        setSkillStoreResults([])
        setSkillStoreSearchedQuery(trimmedQuery)
      } finally {
        if (ticket.isCurrent() && skillStoreRequestRef.current === requestId) {
          setSkillStoreLoading(false)
        }
      }
    },
    [
      exclusivePanel,
      httpClient,
      setSkillStoreError,
      setSkillStoreIndex,
      setSkillStoreLoading,
      setSkillStoreMessage,
      setSkillStoreOpen,
      setSkillStoreQuery,
      setSkillStoreResults,
      setSkillStoreSearchedQuery,
      skillStoreRequestRef,
    ],
  )

  const installSkillStoreResult = useCallback(
    async (result: MarketplaceSkillSearchResult) => {
      if (result.installed || skillStoreInstallingSource !== null) {
        return
      }

      const requestId = skillStoreInstallRequestRef.current + 1
      skillStoreInstallRequestRef.current = requestId
      setSkillStoreInstallingSource(result.source)
      setSkillStoreError(null)
      setSkillStoreMessage(`Installing ${result.metadata.name}...`)

      try {
        const installed = await httpClient.installSkill({ source: result.source })
        if (skillStoreInstallRequestRef.current !== requestId) {
          return
        }
        setSkillStoreResults((current) =>
          current.map((item) =>
            item.source === result.source ? { ...item, installed: true } : item,
          ),
        )
        setSkillStoreMessage(formatSkillInstallResult(installed.installed))
        void refreshInstalledSkills()
      } catch (error) {
        if (skillStoreInstallRequestRef.current !== requestId) {
          return
        }
        setSkillStoreError(error instanceof Error ? error.message : 'Failed to install skill.')
      } finally {
        if (skillStoreInstallRequestRef.current === requestId) {
          setSkillStoreInstallingSource(null)
        }
      }
    },
    [
      httpClient,
      refreshInstalledSkills,
      setSkillStoreError,
      setSkillStoreInstallingSource,
      setSkillStoreMessage,
      setSkillStoreResults,
      skillStoreInstallRequestRef,
      skillStoreInstallingSource,
    ],
  )

  const focusModelPickerSelection = useCallback(
    (target: 'session' | 'default', query = '') => {
      const preferredProvider = target === 'default' ? daemonDefaultProvider : state.provider
      const preferredModel = target === 'default' ? daemonDefaultModel : state.model
      const ordered = buildProviderModelPickerList(providers, state.provider, state.model, query, {
        defaultProviderId: daemonDefaultProvider,
        defaultModelId: daemonDefaultModel,
        mru: modelMru,
      }).ordered
      const preferredIndex = ordered.findIndex(
        (item) =>
          item.kind === 'model' &&
          item.providerId === preferredProvider &&
          item.modelId === preferredModel,
      )

      setModelPickerIndex(preferredIndex >= 0 ? preferredIndex : 0)
    },
    [daemonDefaultModel, daemonDefaultProvider, modelMru, providers, state.model, state.provider],
  )

  const openProviderSetup = useCallback(
    async (
      options: {
        presetQuery?: string | null
        providerId?: string | null
        allowFromModelPicker?: boolean
      } = {},
    ) => {
      const decision = decideOpenProviderSetupRequest({
        request: options,
        isStreaming: state.isStreaming,
        hasPendingApproval: Boolean(state.pendingApproval),
        overlayState,
        currentProvider: state.provider,
      })
      if (decision.kind === 'block') {
        dispatch({ type: 'SYSTEM_MESSAGE', content: decision.message })
        return
      }

      if (modelPickerOpen && options.allowFromModelPicker) {
        closeModelPicker()
      }

      const { requestedPreset, fallbackPreset } = decision

      setInputValue('')
      setProviderSetup({
        mode: options.providerId ? 'edit' : 'new',
        allowPresetSelection: !options.providerId && !requestedPreset,
        step: options.providerId || requestedPreset ? 'providerId' : 'preset',
        applyToCurrentSession: false,
        preset: fallbackPreset,
        selectedPresetIndex: Math.max(
          0,
          fallbackPreset
            ? PROVIDER_WIZARD_PRESETS.findIndex((preset) => preset.type === fallbackPreset.type)
            : 0,
        ),
        sourceProviderId: options.providerId ?? null,
        providerId: fallbackPreset?.type ?? '',
        baseUrl: '',
        apiKeyEnvVar: fallbackPreset?.apiKeyEnvVar ?? '',
        apiKeyValue: '',
        headersText: '',
        model: state.model,
        modelSuggestions: fallbackPreset
          ? buildProviderModelSuggestions(
              buildProviderSetupDraft({
                preset: fallbackPreset,
                currentProvider: state.provider,
                currentModel: state.model,
              }),
            )
          : [],
        modelSuggestionIndex: 0,
        configProviders: [],
        loading: true,
        validating: false,
        saving: false,
        error: null,
      })

      try {
        const daemonConfig = await httpClient.config()
        const configProviders = daemonConfig.providers as ConfiguredProviderRecord[]
        const existingProvider = options.providerId
          ? (configProviders.find((provider) => provider.id === options.providerId) ?? null)
          : null

        if (options.providerId && !existingProvider) {
          throw new Error(`Provider ${options.providerId} is not configured.`)
        }
        if (existingProvider && !findBuiltinProviderPreset(existingProvider.type)) {
          throw new Error(
            `Provider ${existingProvider.id} uses unsupported type "${existingProvider.type}".`,
          )
        }

        const preset =
          requestedPreset ?? findBuiltinProviderPreset(existingProvider?.type) ?? fallbackPreset

        if (!preset) {
          throw new Error('No built-in provider presets are available.')
        }

        const draft = buildProviderSetupDraft({
          provider: existingProvider,
          preset,
          currentProvider: state.provider,
          currentModel: state.model,
        })

        setProviderSetup({
          mode: options.providerId ? 'edit' : 'new',
          allowPresetSelection: !options.providerId && !requestedPreset,
          step: options.providerId || requestedPreset ? 'providerId' : 'preset',
          applyToCurrentSession: false,
          preset,
          selectedPresetIndex: Math.max(
            0,
            PROVIDER_WIZARD_PRESETS.findIndex((item) => item.type === preset.type),
          ),
          sourceProviderId: draft.sourceProviderId,
          providerId: draft.providerId,
          baseUrl: draft.baseUrl,
          apiKeyEnvVar: draft.apiKeyEnvVar,
          apiKeyValue: '',
          headersText: formatProviderHeadersInput(draft.headers),
          model: draft.model,
          modelSuggestions: preset.type === 'ollama' ? [] : buildProviderModelSuggestions(draft),
          modelSuggestionIndex: 0,
          configProviders,
          loading: false,
          validating: false,
          saving: false,
          error: null,
        })
      } catch (error) {
        setProviderSetup(null)
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: error instanceof Error ? error.message : 'Failed to open provider setup.',
        })
      }
    },
    [
      closeModelPicker,
      dispatch,
      httpClient,
      modelPickerOpen,
      overlayState,
      state.isStreaming,
      state.model,
      state.pendingApproval,
      state.provider,
    ],
  )

  const openProviderSetupRef = useRef(openProviderSetup)
  useEffect(() => {
    openProviderSetupRef.current = openProviderSetup
  }, [openProviderSetup])

  const selectProviderSetupPreset = useCallback(
    (index: number) => {
      const preset = PROVIDER_WIZARD_PRESETS[index]
      if (!preset) {
        return
      }

      const draft = buildProviderSetupDraft({
        preset,
        currentProvider: state.provider,
        currentModel: state.model,
      })

      setProviderSetup((current) =>
        current
          ? {
              ...current,
              step: 'providerId',
              preset,
              selectedPresetIndex: index,
              sourceProviderId: draft.sourceProviderId,
              providerId: draft.providerId,
              baseUrl: draft.baseUrl,
              apiKeyEnvVar: draft.apiKeyEnvVar,
              apiKeyValue: '',
              headersText: formatProviderHeadersInput(draft.headers),
              model: draft.model,
              modelSuggestions:
                preset.type === 'ollama' ? [] : buildProviderModelSuggestions(draft),
              modelSuggestionIndex: 0,
              error: null,
            }
          : current,
      )
    },
    [state.model, state.provider],
  )

  const toggleProviderSetupApplyToCurrentSession = useCallback(() => {
    setProviderSetup((current) => {
      if (
        !current ||
        current.loading ||
        current.validating ||
        current.saving ||
        current.step !== 'confirm'
      ) {
        return current
      }

      return {
        ...current,
        applyToCurrentSession: !current.applyToCurrentSession,
        error: null,
      }
    })
  }, [])

  const goBackProviderSetup = useCallback(() => {
    setProviderSetup((current) => {
      if (!current || current.loading || current.validating || current.saving) {
        return current
      }

      switch (current.step) {
        case 'confirm':
          return {
            ...current,
            step: 'model',
            error: null,
          }
        case 'model':
          return {
            ...current,
            step:
              current.preset?.type === 'ollama'
                ? 'baseUrl'
                : current.preset?.type === 'custom'
                  ? 'headers'
                  : 'apiKeyValue',
            error: null,
          }
        case 'headers':
          return {
            ...current,
            step: current.apiKeyValue.length > 0 ? 'apiKeyValue' : 'apiKeyEnv',
            error: null,
          }
        case 'apiKeyValue':
          return {
            ...current,
            step: current.preset?.type === 'custom' ? 'baseUrl' : 'providerId',
            error: null,
          }
        case 'baseUrl':
          return {
            ...current,
            step: 'providerId',
            error: null,
          }
        case 'apiKeyEnv':
          return {
            ...current,
            step: 'apiKeyValue',
            error: null,
          }
        case 'providerId':
          return current.allowPresetSelection
            ? {
                ...current,
                step: 'preset',
                error: null,
              }
            : null
        case 'preset':
          return null
      }
    })
  }, [])

  const saveProviderSetup = useCallback(
    async (current: ProviderSetupState) => {
      if (!current.preset) {
        return
      }

      const draft: ProviderSetupDraft = {
        sourceProviderId: current.sourceProviderId,
        preset: current.preset,
        providerId: current.providerId.trim(),
        baseUrl: current.baseUrl.trim(),
        apiKeyEnvVar: current.apiKeyEnvVar.trim(),
        headers: parseProviderHeadersInput(current.headersText ?? ''),
        model: current.model.trim(),
        modelAliasSource: current.modelAliasSource,
        models:
          current.preset.type === 'custom' || current.preset.type === 'ollama'
            ? current.modelSuggestions
            : undefined,
      }
      const secretUpdate = buildProviderSecretEnvUpdate({
        preset: current.preset,
        apiKeyEnvVar: draft.apiKeyEnvVar,
        apiKeyValue: current.apiKeyValue,
      })

      setProviderSetup((state) =>
        state ? { ...state, validating: true, saving: false, error: null } : state,
      )

      try {
        const providerConfig = buildProviderConfigRecord({
          config: {
            providers: current.configProviders,
          },
          draft,
        })
        const validation = await httpClient.validateProvider({
          provider: providerConfig,
          model: draft.model,
          env: secretUpdate?.updates
            ? Object.fromEntries(
                Object.entries(secretUpdate.updates).filter(
                  (entry): entry is [string, string] => typeof entry[1] === 'string',
                ),
              )
            : undefined,
        })
        setProviderSetup((state) =>
          state ? { ...state, validating: false, saving: true, error: null } : state,
        )
        if (secretUpdate) {
          await httpClient.updateConfigEnv(secretUpdate)
        }
        const configUpdate = buildProviderConfigUpdate({
          config: {
            providers: current.configProviders,
          },
          draft,
        })
        await httpClient.updateConfig(configUpdate)
        await httpClient.refreshProviderModels(draft.providerId).catch(() => undefined)
        await loadProviders().catch(() => undefined)
        setDaemonDefaultProvider(draft.providerId)
        setDaemonDefaultModel(draft.model)
        const defaultModeChangedTo =
          typeof configUpdate['agent.mode'] === 'string' ? configUpdate['agent.mode'] : null
        if (defaultModeChangedTo) {
          dispatch({ type: 'SET_MODE', mode: defaultModeChangedTo })
        }
        const sessionAlreadyMatches =
          state.provider === draft.providerId && state.model === draft.model
        if (current.applyToCurrentSession) {
          dispatch({ type: 'SET_PROVIDER', provider: draft.providerId })
          dispatch({ type: 'SET_MODEL', model: draft.model })
        }
        setProviderSetup(null)
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: buildProviderSetupSuccessMessage({
            mode: current.mode,
            draft,
            validationLatencyMs: validation.latencyMs,
            secretUpdated: Boolean(secretUpdate),
            defaultModeChangedTo,
            applyToCurrentSession: current.applyToCurrentSession,
            sessionAlreadyMatches,
            currentSessionProvider: state.provider,
            currentSessionModel: state.model,
          }),
        })
      } catch (error) {
        setProviderSetup((state) =>
          state
            ? {
                ...state,
                validating: false,
                saving: false,
                error:
                  error instanceof Error ? error.message : 'Failed to save provider configuration.',
              }
            : state,
        )
      }
    },
    [dispatch, httpClient, loadProviders, state.model, state.provider],
  )

  const advanceProviderSetup = useCallback(async () => {
    const current = providerSetup
    if (!current || current.loading || current.validating || current.saving || !current.preset) {
      return
    }

    const nextProviderId = current.providerId.trim()
    const rawNextBaseUrl = current.baseUrl.trim()
    const normalizedCustomBaseUrl = rawNextBaseUrl.replace(/\/+$/, '')
    const nextBaseUrl =
      current.preset?.type === 'custom' && normalizedCustomBaseUrl
        ? normalizedCustomBaseUrl.endsWith('/v1')
          ? normalizedCustomBaseUrl
          : `${normalizedCustomBaseUrl}/v1`
        : rawNextBaseUrl
    const nextApiKeyEnvVar = current.apiKeyEnvVar.trim()
    const nextApiKeyValue = current.apiKeyValue
    const nextHeadersText = (current.headersText ?? '').trim()
    const nextModel = current.model.trim()

    switch (current.step) {
      case 'preset':
        selectProviderSetupPreset(current.selectedPresetIndex)
        return
      case 'providerId':
        if (!nextProviderId) {
          setProviderSetup((state) =>
            state ? { ...state, error: 'Provider id is required.' } : state,
          )
          return
        }
        setProviderSetup((state) =>
          state
            ? {
                ...state,
                providerId: nextProviderId,
                step:
                  current.preset?.type === 'ollama' || current.preset?.type === 'custom'
                    ? 'baseUrl'
                    : 'apiKeyValue',
                error: null,
              }
            : state,
        )
        return
      case 'baseUrl': {
        if (!nextBaseUrl) {
          setProviderSetup((state) =>
            state ? { ...state, error: 'Base URL is required.' } : state,
          )
          return
        }
        setProviderSetup((state) =>
          state ? { ...state, baseUrl: nextBaseUrl, loading: true, error: null } : state,
        )
        if (current.preset.type === 'custom') {
          setProviderSetup((state) =>
            state
              ? {
                  ...state,
                  baseUrl: nextBaseUrl,
                  loading: false,
                  step: 'apiKeyValue',
                  error: null,
                }
              : state,
          )
          return
        }
        const discoveredModels = await detectOllamaModels(nextBaseUrl)
        const selection = reconcileProviderModelDiscovery({
          currentModel: nextModel,
          discoveredModels,
          fallbackModels: current.modelSuggestions.length > 0
            ? current.modelSuggestions
            : current.preset.suggestedModels,
          defaultModel: current.preset.type === 'ollama' ? 'llama3.3' : undefined,
        })
        const draft: ProviderSetupDraft = {
          sourceProviderId: current.sourceProviderId,
          preset: current.preset,
          providerId: nextProviderId || current.providerId,
          baseUrl: nextBaseUrl,
          apiKeyEnvVar: nextApiKeyEnvVar || current.apiKeyEnvVar,
          model: selection.model,
          models: selection.modelSuggestions,
        }
        setProviderSetup((state) =>
          state
            ? {
                ...state,
                providerId: draft.providerId,
                baseUrl: draft.baseUrl,
                model: draft.model,
                modelSuggestions: selection.modelSuggestions,
                modelSuggestionIndex: selection.modelSuggestionIndex,
                step: 'model',
                loading: false,
                error: null,
              }
            : state,
        )
        return
      }
      case 'apiKeyEnv': {
        if (!nextApiKeyEnvVar && current.preset.type !== 'custom') {
          setProviderSetup((state) =>
            state ? { ...state, error: 'API key env var is required.' } : state,
          )
          return
        }
        if (nextApiKeyEnvVar && !isValidProviderEnvVarName(nextApiKeyEnvVar)) {
          setProviderSetup((state) =>
            state
              ? {
                  ...state,
                  error: 'API key env var must match [A-Za-z_][A-Za-z0-9_]*.',
                }
              : state,
          )
          return
        }
        const draft: ProviderSetupDraft = {
          sourceProviderId: current.sourceProviderId,
          preset: current.preset,
          providerId: nextProviderId || current.providerId,
          baseUrl: nextBaseUrl || current.baseUrl,
          apiKeyEnvVar: nextApiKeyEnvVar,
          model: nextModel || current.model,
        }
        setProviderSetup((state) =>
          state
            ? {
                ...state,
                providerId: draft.providerId,
                apiKeyEnvVar: draft.apiKeyEnvVar,
                model: draft.model,
                modelSuggestions: buildProviderModelSuggestions(draft),
                modelSuggestionIndex: 0,
                step: draft.preset.type === 'custom' ? 'headers' : 'model',
                error: null,
              }
            : state,
        )
        return
      }
      case 'apiKeyValue': {
        if (!nextApiKeyValue.length) {
          setProviderSetup((state) =>
            state ? { ...state, apiKeyValue: '', step: 'apiKeyEnv', error: null } : state,
          )
          return
        }
        const managedEnvVar = managedProviderApiKeyEnvVar({
          providerId: nextProviderId || current.providerId,
          preset: current.preset,
          currentEnvVar: nextApiKeyEnvVar,
        })
        setProviderSetup((state) =>
          state
            ? {
                ...state,
                apiKeyValue: nextApiKeyValue,
                apiKeyEnvVar: managedEnvVar,
                step: current.preset?.type === 'custom' ? 'headers' : 'model',
                error: null,
              }
            : state,
        )
        return
      }
      case 'headers': {
        let headers: Record<string, string>
        try {
          headers = parseProviderHeadersInput(nextHeadersText)
        } catch (error) {
          setProviderSetup((state) =>
            state
              ? {
                  ...state,
                  error: error instanceof Error ? error.message : 'Invalid custom headers.',
                }
              : state,
          )
          return
        }
        setProviderSetup((state) =>
          state ? { ...state, headersText: nextHeadersText, loading: true, error: null } : state,
        )
        const discoveredModels = await httpClient
          .discoverProviderModels({
            type: current.preset.type,
            baseUrl: nextBaseUrl || current.baseUrl,
            apiKey:
              nextApiKeyValue.length > 0
                ? nextApiKeyValue
                : (nextApiKeyEnvVar ? `\${${nextApiKeyEnvVar}}` : undefined),
            headers,
          })
          .then((result) => result.models)
          .catch(() => [])
        const selection = reconcileProviderModelDiscovery({
          currentModel: nextModel,
          discoveredModels,
          fallbackModels: current.modelSuggestions,
        })
        setProviderSetup((state) =>
          state
            ? {
                ...state,
                headersText: nextHeadersText,
                model: selection.model,
                modelSuggestions: selection.modelSuggestions,
                modelSuggestionIndex: selection.modelSuggestionIndex,
                modelAliasSource: selection.modelAliasSource,
                step: 'model',
                loading: false,
                error: null,
              }
            : state,
        )
        return
      }
      case 'model':
        if (!nextModel) {
          setProviderSetup((state) =>
            state ? { ...state, error: 'Model name is required.' } : state,
          )
          return
        }
        setProviderSetup((state) =>
          state
            ? {
                ...state,
                model: nextModel,
                step: 'confirm',
                error: null,
              }
            : state,
        )
        return
      case 'confirm':
        await saveProviderSetup({
          ...current,
          providerId: nextProviderId,
          baseUrl: nextBaseUrl,
          apiKeyEnvVar: nextApiKeyEnvVar,
          apiKeyValue: nextApiKeyValue,
          model: nextModel,
        })
    }
  }, [providerSetup, saveProviderSetup, selectProviderSetupPreset])

  const applyProviderModelSelection = useCallback(
    (
      selection: Pick<ProviderModelOption, 'providerId' | 'modelId'>,
      source: 'picker' | 'command' = 'command',
    ) => {
      dispatch({ type: 'SET_PROVIDER', provider: selection.providerId })
      dispatch({ type: 'SET_MODEL', model: selection.modelId })
      if (source === 'picker') {
        closeModelPicker()
      }
      dispatch({
        type: 'SYSTEM_MESSAGE',
        content: `Model set to ${selection.providerId}/${selection.modelId}`,
      })
    },
    [closeModelPicker, dispatch],
  )

  const applyDaemonDefaultToSession = useCallback(
    async (source: 'picker' | 'command' = 'command') => {
      if (!daemonDefaultProvider || !daemonDefaultModel) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content:
            'No daemon default model is configured yet. Use /model default or Ctrl+T then Ctrl+D first.',
        })
        return
      }

      let availableProviders = providers
      if (availableProviders.length === 0) {
        try {
          availableProviders = await ensureProvidersLoaded()
        } catch (error) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: error instanceof Error ? error.message : 'Failed to load providers.',
          })
          return
        }
      }

      const defaultProvider = availableProviders.find(
        (provider) => provider.id === daemonDefaultProvider,
      )
      if (!defaultProvider) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Daemon default ${daemonDefaultProvider}/${daemonDefaultModel} is not configured right now. Use /provider edit ${daemonDefaultProvider} to repair it.`,
        })
        return
      }

      if (defaultProvider.health.status !== 'ready') {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Daemon default ${daemonDefaultProvider}/${daemonDefaultModel} is not usable yet: ${formatProviderHealthSummary(defaultProvider)}.\nUse /provider edit ${daemonDefaultProvider} to repair it.`,
        })
        return
      }

      const model = defaultProvider.models.find((item) => item.id === daemonDefaultModel)
      if (!model) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Daemon default model ${daemonDefaultProvider}/${daemonDefaultModel} is missing from the active provider registry.\nUse /provider edit ${daemonDefaultProvider} or /model default to repair it.`,
        })
        return
      }

      applyProviderModelSelection(
        {
          providerId: defaultProvider.id,
          modelId: model.id,
        },
        source,
      )
    },
    [
      applyProviderModelSelection,
      daemonDefaultModel,
      daemonDefaultProvider,
      dispatch,
      ensureProvidersLoaded,
      providers,
    ],
  )

  const setProviderAsDefault = useCallback(
    async (options: {
      providerId: string
      modelId?: string | null
      source?: 'picker' | 'command'
    }) => {
      const decision = decideStreamApprovalGate({
        isStreaming: state.isStreaming,
        hasPendingApproval: Boolean(state.pendingApproval),
        action: 'changing provider defaults',
      })
      if (decision.kind === 'block') {
        dispatch({ type: 'SYSTEM_MESSAGE', content: decision.message })
        return
      }

      try {
        const snapshot = await loadProviderConfigSnapshot()
        const updates = buildProviderDefaultUpdate({
          config: {
            providers: snapshot.configProviders,
            agent: {
              defaultProvider: snapshot.defaultProviderId ?? undefined,
              defaultModel: snapshot.defaultModel ?? undefined,
            },
          },
          providerId: options.providerId,
          preferredModel: options.modelId ?? null,
        })
        const nextProvider = updates['agent.defaultProvider']
        const nextModel = updates['agent.defaultModel']
        if (typeof nextProvider !== 'string' || typeof nextModel !== 'string') {
          throw new Error('Failed to resolve the next daemon default provider.')
        }

        await httpClient.updateConfig(updates)
        setDaemonDefaultProvider(nextProvider)
        setDaemonDefaultModel(nextModel)
        await loadProviders().catch(() => undefined)
        if (options.source === 'picker') {
          closeModelPicker()
        }
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content:
            nextProvider === state.provider && nextModel === state.model
              ? `Daemon default set to ${nextProvider}/${nextModel}. Current session already matches.`
              : `Daemon default set to ${nextProvider}/${nextModel}. Current session stays on ${state.provider}/${state.model}. Run /model default apply to switch this session now.`,
        })
      } catch (error) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content:
            error instanceof Error
              ? `${error.message}\nUse /provider setup or Ctrl+T to inspect configured providers.`
              : 'Failed to update the daemon default provider.\nUse /provider setup or Ctrl+T to inspect configured providers.',
        })
      }
    },
    [
      closeModelPicker,
      dispatch,
      httpClient,
      loadProviderConfigSnapshot,
      loadProviders,
      state.isStreaming,
      state.model,
      state.pendingApproval,
      state.provider,
    ],
  )

  const openProviderDeleteConfirmation = useCallback(
    async (options: { providerId: string; allowFromModelPicker?: boolean }) => {
      const decision = decideOpenProviderDeleteConfirmation({
        request: options,
        isStreaming: state.isStreaming,
        hasPendingApproval: Boolean(state.pendingApproval),
        providerSetupOpen,
        providerDeleteOpen,
        overlayState,
      })
      if (decision.kind === 'noop') return
      if (decision.kind === 'block') {
        dispatch({ type: 'SYSTEM_MESSAGE', content: decision.message })
        return
      }
      if (modelPickerOpen && options.allowFromModelPicker) {
        closeModelPicker()
      }

      try {
        const snapshot = await loadProviderConfigSnapshot()
        const providerInfo =
          providers.find((provider) => provider.id === options.providerId) ?? null
        const providerName = providerInfo?.name ?? options.providerId
        const fallback = derivePreferredProviderFallback({
          deletedProviderId: options.providerId,
          daemonDefaultProviderId: snapshot.defaultProviderId,
          sessionProvider: state.provider,
          sessionModel: state.model,
        })
        const deleteUpdates = buildProviderDeleteUpdate({
          config: {
            providers: snapshot.configProviders,
            agent: {
              defaultProvider: snapshot.defaultProviderId ?? undefined,
              defaultModel: snapshot.defaultModel ?? undefined,
            },
          },
          providerId: options.providerId,
          preferredFallbackProviderId: fallback.providerId,
          preferredFallbackModel: fallback.model,
        })

        const nextProviderId = deleteUpdates['agent.defaultProvider']
        const nextModel = deleteUpdates['agent.defaultModel']
        if (typeof nextProviderId !== 'string' || typeof nextModel !== 'string') {
          throw new Error('Failed to resolve the fallback provider after deletion.')
        }

        setProviderDeleteConfirm({
          providerId: options.providerId,
          providerName,
          configProviders: snapshot.configProviders,
          defaultProviderId: snapshot.defaultProviderId,
          defaultModel: snapshot.defaultModel,
          nextProviderId,
          nextModel,
          deleting: false,
          error: null,
        })
      } catch (error) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: error instanceof Error ? error.message : 'Failed to prepare provider deletion.',
        })
      }
    },
    [
      closeModelPicker,
      dispatch,
      loadProviderConfigSnapshot,
      modelPickerOpen,
      overlayState,
      providerDeleteOpen,
      providers,
      state.isStreaming,
      state.model,
      state.pendingApproval,
      state.provider,
    ],
  )

  const confirmProviderDelete = useCallback(async () => {
    const current = providerDeleteConfirm
    if (!current || current.deleting) {
      return
    }

    setProviderDeleteConfirm((state) => (state ? { ...state, deleting: true, error: null } : state))

    try {
      const fallback = derivePreferredProviderFallback({
        deletedProviderId: current.providerId,
        daemonDefaultProviderId: current.defaultProviderId,
        sessionProvider: state.provider,
        sessionModel: state.model,
      })
      const updates = buildProviderDeleteUpdate({
        config: {
          providers: current.configProviders,
          agent: {
            defaultProvider: current.defaultProviderId ?? undefined,
            defaultModel: current.defaultModel ?? undefined,
          },
        },
        providerId: current.providerId,
        preferredFallbackProviderId: fallback.providerId,
        preferredFallbackModel: fallback.model,
      })
      const nextProviderId = updates['agent.defaultProvider']
      const nextModel = updates['agent.defaultModel']
      if (typeof nextProviderId !== 'string' || typeof nextModel !== 'string') {
        throw new Error('Failed to resolve the fallback provider after deletion.')
      }

      await httpClient.updateConfig(updates)
      await loadProviders().catch(() => undefined)
      setDaemonDefaultProvider(nextProviderId)
      setDaemonDefaultModel(nextModel)
      if (state.provider === current.providerId) {
        dispatch({ type: 'SET_PROVIDER', provider: nextProviderId })
        dispatch({ type: 'SET_MODEL', model: nextModel })
      }
      setProviderDeleteConfirm(null)
      dispatch({
        type: 'SYSTEM_MESSAGE',
        content: `Deleted provider ${current.providerId}. Daemon default is now ${nextProviderId}/${nextModel}.`,
      })
    } catch (error) {
      setProviderDeleteConfirm((state) =>
        state
          ? {
              ...state,
              deleting: false,
              error: error instanceof Error ? error.message : 'Failed to delete provider.',
            }
          : state,
      )
    }
  }, [dispatch, httpClient, loadProviders, providerDeleteConfirm, state.model, state.provider])

  const switchDaemonDefaultAndSession = useCallback(
    async (selection: Pick<ProviderModelOption, 'providerId' | 'modelId'>) => {
      const decision = decideStreamApprovalGate({
        isStreaming: state.isStreaming,
        hasPendingApproval: Boolean(state.pendingApproval),
        action: 'changing provider defaults',
      })
      if (decision.kind === 'block') {
        dispatch({ type: 'SYSTEM_MESSAGE', content: decision.message })
        return
      }

      try {
        const result = await httpClient.switchDefaultModel({
          target: `${selection.providerId}/${selection.modelId}`,
        })
        if (!result.ok) {
          dispatch({ type: 'SYSTEM_MESSAGE', content: result.message })
          return
        }
        setDaemonDefaultProvider(result.providerId)
        setDaemonDefaultModel(result.model)
        dispatch({ type: 'SET_PROVIDER', provider: result.providerId })
        dispatch({ type: 'SET_MODEL', model: result.model })
        closeModelPicker()
        void recordModelUse(`${result.providerId}/${result.model}`)
        dispatch({ type: 'SYSTEM_MESSAGE', content: result.message })
      } catch (error) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content:
            error instanceof Error
              ? error.message
              : 'Failed to save the daemon default model.',
        })
      }
    },
    [
      closeModelPicker,
      dispatch,
      httpClient,
      setDaemonDefaultModel,
      setDaemonDefaultProvider,
      state.isStreaming,
      state.pendingApproval,
    ],
  )

  const handleModelPickerSelection = useCallback(
    (selection: ProviderModelPickerItem, target: 'session' | 'default' = 'session') => {
      if (selection.kind === 'model') {
        if (target === 'default') {
          void switchDaemonDefaultAndSession(selection)
          return
        }
        applyProviderModelSelection(selection, 'picker')
        void recordModelUse(`${selection.providerId}/${selection.modelId}`)
        return
      }

      switch (selection.action) {
        case 'sync-session-default':
          void applyDaemonDefaultToSession('picker')
          return
        case 'setup':
          void openProviderSetup({ allowFromModelPicker: true })
          return
        case 'setup-preset':
          void openProviderSetup({
            presetQuery: selection.presetType,
            allowFromModelPicker: true,
          })
          return
        case 'edit-provider':
          if (!selection.providerId) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: 'Provider edit action is missing a provider id.',
            })
            return
          }
          void openProviderSetup({
            providerId: selection.providerId,
            allowFromModelPicker: true,
          })
          return
        case 'set-default-provider':
          if (!selection.providerId) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: 'Provider default action is missing a provider id.',
            })
            return
          }
          void setProviderAsDefault({
            providerId: selection.providerId,
            modelId: selection.modelId ?? null,
            source: 'picker',
          })
          return
        case 'delete-provider':
          if (!selection.providerId) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: 'Provider delete action is missing a provider id.',
            })
            return
          }
          void openProviderDeleteConfirmation({
            providerId: selection.providerId,
            allowFromModelPicker: true,
          })
      }
    },
    [
      applyDaemonDefaultToSession,
      applyProviderModelSelection,
      dispatch,
      openProviderDeleteConfirmation,
      openProviderSetup,
      setProviderAsDefault,
      switchDaemonDefaultAndSession,
    ],
  )

  const applyAutonomySetting = useCallback(
    async (autonomy: AutonomyLevel, source: 'command' | 'picker' | 'shortcut' = 'command') => {
      try {
        await httpClient.updateConfig({ 'agent.autonomy': autonomy })
        dispatch({
          type: 'SET_AUTONOMY',
          autonomy,
        })
        if (source === 'picker') {
          closeAutonomyPicker()
        }
        const option = getAutonomyOption(autonomy)
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Autonomy set to ${option.label} (${option.id}).`,
        })
      } catch (error) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: error instanceof Error ? error.message : 'Failed to update autonomy.',
        })
      }
    },
    [closeAutonomyPicker, dispatch, httpClient],
  )

  useEffect(() => {
    setPaletteIndex(0)
  }, [inputValue])

  useEffect(() => {
    setAttachmentSuggestionIndex(0)
  }, [inputValue])

  useEffect(() => {
    let cancelled = false

    void loadCliState().then((cliState) => {
      if (cancelled) return
      if (cliState.recapEnabled === false) {
        setRecapEnabled(false)
      }
      if (!cliState.themeId) return
      applyThemeSelection(cliState.themeId, 'startup')
    })

    return () => {
      cancelled = true
    }
  }, [applyThemeSelection])

  useEffect(() => {
    workspaceProjectRef.current = workspaceProject
  }, [workspaceProject])

  useEffect(() => {
    void refreshProjectSessionIds(workspaceProject)
  }, [refreshProjectSessionIds, workspaceProject])

  // Connect on mount
  useEffect(() => {
    let cancelled = false

    async function connect() {
      if (!cancelled) {
        setConnectionStatus('connecting')
        setConnectionError(null)
      }

      try {
        const health = await httpClient.health()
        if (cancelled) return
        setVersion(health.version)
        setConnectionStatus('connected')
        // One-time notice about sepilot-managed Docker containers left over
        // from previous sessions. Fire-and-forget; the helper swallows errors
        // and returns [] when Docker is unavailable, so this never blocks.
        void buildLeftoverContainerBanner().then((lines) => {
          if (cancelled || lines.length === 0) return
          dispatch({ type: 'SYSTEM_MESSAGE', content: lines.join('\n') })
        })
        const agents = await loadAgentModesOrEmpty(httpClient)
        if (cancelled) return
        setAgentModes(agents)

        const providersResult = await loadProvidersOrEmpty(httpClient)
        if (cancelled) return
        setProviders(providersResult.providers)
        const configuredProviders = providersResult.providers
        const providersLoaded = providersResult.loaded

        const daemonConfig = await loadDaemonConfigDefaults(httpClient)
        if (cancelled) return
        const configuredDefaultProvider = daemonConfig?.defaultProvider ?? ''
        const configuredDefaultModel = daemonConfig?.defaultModel ?? ''
        applyDaemonConfigToState({
          daemonConfig,
          cliOverrides: { model: config.model, provider: config.provider },
          dispatch,
        })
        if (daemonConfig) {
          setDaemonDefaultProvider(configuredDefaultProvider)
          setDaemonDefaultModel(configuredDefaultModel)
        }

        const earlyAutoProviderSetup = decideAutoOpenProviderSetup({
          daemonUrl: config.url,
          providersLoaded,
          providers: configuredProviders,
          defaultProviderId: config.provider ?? configuredDefaultProvider,
          defaultModelId: config.model ?? configuredDefaultModel,
          initialSessionId: config.sessionId ?? null,
          openedKey: autoProviderSetupKeyRef.current,
        })
        if (!cancelled && earlyAutoProviderSetup.kind === 'open') {
          autoProviderSetupKeyRef.current = earlyAutoProviderSetup.key
          const banner = getAutoProviderSetupBannerMessage(earlyAutoProviderSetup.reason)
          if (banner) {
            dispatch({ type: 'SYSTEM_MESSAGE', content: banner })
          }
          if (earlyAutoProviderSetup.providerId) {
            void openProviderSetupRef.current({ providerId: earlyAutoProviderSetup.providerId })
          } else if (earlyAutoProviderSetup.presetQuery) {
            void openProviderSetupRef.current({ presetQuery: earlyAutoProviderSetup.presetQuery })
          } else {
            void openProviderSetupRef.current()
          }
        }

        const cliState = await loadCliState()
        if (cancelled) return
        setRecentSessionIds(cliState.recentSessionIds ?? [])

        const detectedWorkspace = await detectWorkspaceOrNull(FILE_PICKER_ROOT_DIR)
        if (cancelled) return
        if (detectedWorkspace) {
          setWorkspaceProject(detectedWorkspace)
          setFilePickerDir(detectedWorkspace.rootDir)
          await refreshProjectSessionIds(detectedWorkspace)
          if (cancelled) return
        } else {
          setWorkspaceProject(null)
          setProjectSessionIds([])
        }

        let initialSessionId = config.sessionId
        if (!initialSessionId && config.resume) {
          const workspaceSessions = await httpClient.sessions(undefined, {
            perPage: 100,
            workspaceRoot: FILE_PICKER_ROOT_DIR,
          })
          // Sort explicitly rather than relying on the daemon's list order —
          // next/App.tsx does the same, and resume must attach the most
          // recent workspace session even if that contract ever changes.
          initialSessionId = workspaceSessions.items
            .filter((session) => sessionBelongsToWorkspace(session, FILE_PICKER_ROOT_DIR))
            .slice()
            .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))[0]?.id
        }

        const projectResult = await resolveInitialProject({
          httpClient,
          detectedWorkspace,
          initialSessionId: initialSessionId ?? undefined,
          projectBindings: cliState.projectBindings,
          cancelled: () => cancelled,
        })
        if (cancelled) return
        if (projectResult.ok && !projectResult.cancelled) {
          setProjects(projectResult.availableProjects)
          if (projectResult.selection) {
            applyProjectSelection(
              projectResult.selection.project,
              projectResult.selection.source,
              projectResult.selection.workspaceRootDir,
            )
          }
        }

        if (initialSessionId) {
          dispatch({ type: 'SET_HYDRATING_SESSION', value: true })
          const result = await fetchInitialSession({
            httpClient,
            sessionId: initialSessionId,
          })
          if (cancelled) return
          if (result.ok) {
            const workspaceError = sessionWorkspaceLoadError(
              result.session,
              FILE_PICKER_ROOT_DIR,
            )
            if (workspaceError) {
              dispatch({ type: 'ERROR', message: workspaceError })
            } else {
              loadSession(result.session, result.artifacts)
            }
          } else {
            const error = result.error
            if (detectedWorkspace && error instanceof Error && error.message.includes('404')) {
              void removeProjectSession(detectedWorkspace, initialSessionId)
                .then((registry) => {
                  if (!cancelled) {
                    setProjectSessionIds(registry.sessions.map((session) => session.sessionId))
                  }
                })
                .catch(() => {})
            }
            dispatch({
              type: 'ERROR',
              message:
                error instanceof Error
                  ? `Failed to load session ${initialSessionId}: ${error.message}`
                  : `Failed to load session ${initialSessionId}`,
            })
            dispatch({ type: 'SET_HYDRATING_SESSION', value: false })
          }
        }

        const ws = await connectWsOrNull({
          client: new DaemonWsClient(config.url),
          cancelled: () => cancelled,
        })
        if (cancelled) return
        setWsClient(ws)

        const startupPreflight = buildStartupPreflightSummary({
          health,
          providers: configuredProviders,
          daemonDefaultProvider: configuredDefaultProvider,
          daemonDefaultModel: configuredDefaultModel,
          providersLoaded,
        })
        const startupPreflightKey = startupPreflight ? `${config.url}:${startupPreflight}` : null
        if (
          !cancelled &&
          !initialSessionId &&
          startupPreflight &&
          startupPreflightKeyRef.current !== startupPreflightKey
        ) {
          startupPreflightKeyRef.current = startupPreflightKey
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: startupPreflight,
          })
        }
      } catch (error) {
        if (!cancelled) {
          setConnectionStatus('error')
          setConnectionError(error instanceof Error ? error.message : 'Unknown connection error.')
        }
      }
    }

    connect()
    return () => {
      cancelled = true
    }
  }, [
    applyProjectSelection,
    config.model,
    config.provider,
    config.resume,
    config.sessionId,
    config.url,
    dispatch,
    httpClient,
    loadSession,
    refreshProjectSessionIds,
  ])

  useEffect(() => {
    if (!state.sessionId) return
    setTranscriptClearedAt(null)
    void recordSessionAccess(state.sessionId)
      .then((cliState) => {
        setRecentSessionIds(cliState.recentSessionIds ?? [])
      })
      .catch(() => {})
  }, [state.sessionId])

  useEffect(() => {
    if (!wsClient) {
      return
    }

    return () => {
      wsClient.close()
    }
  }, [wsClient])

  useEffect(() => {
    if (!state.sessionId || !workspaceProject) {
      return
    }

    let cancelled = false
    const currentSessionId = state.sessionId
    const currentWorkspace = workspaceProject

    void (async () => {
      const result = await syncProjectSessionRegistry({
        httpClient,
        workspace: currentWorkspace,
        sessionId: currentSessionId,
        fallback: {
          provider: state.provider,
          model: state.model,
          projectId: state.projectId,
          projectName: state.projectName,
        },
        cancelled: () => cancelled,
      })
      if (cancelled) return
      if (result.ok && !result.cancelled && result.projectSessionIds.length > 0) {
        setProjectSessionIds(result.projectSessionIds)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [
    httpClient,
    state.model,
    state.projectId,
    state.projectName,
    state.provider,
    state.sessionId,
    workspaceProject,
  ])

  useEffect(() => {
    const decision = decideAutoProjectFromSession({
      sessionId: state.sessionId,
      projects,
      selectedProjectId: selectedProject?.id ?? null,
      projectSelectionSource,
    })
    if (decision.kind === 'apply') {
      applyProjectSelection(decision.project, 'session', workspaceProject?.rootDir)
    }
  }, [
    applyProjectSelection,
    projectSelectionSource,
    projects,
    selectedProject?.id,
    state.sessionId,
    workspaceProject?.rootDir,
  ])

  useEffect(() => {
    const decision = decideSessionProjectAttach({
      sessionId: state.sessionId,
      selectedProject: selectedProject ?? null,
      projects,
      projectSelectionSource,
      lastAttachedKey: lastAttachedProjectSessionRef.current,
    })
    switch (decision.kind) {
      case 'reset':
      case 'already-attached':
        lastAttachedProjectSessionRef.current = decision.nextRef
        return
      case 'skip':
        return
      case 'attach':
        lastAttachedProjectSessionRef.current = decision.attachmentKey
        void httpClient
          .attachSessionToProject(decision.projectId, decision.sessionId)
          .then((project) => {
            setProjects((current) => upsertProject(current, project))
            if (state.projectId === project.id) {
              dispatch({
                type: 'SET_PROJECT',
                projectId: project.id,
                projectName: project.name,
              })
            }
          })
          .catch(() => {
            lastAttachedProjectSessionRef.current = null
          })
        return
    }
  }, [
    dispatch,
    httpClient,
    projectSelectionSource,
    projects,
    selectedProject,
    state.projectId,
    state.sessionId,
  ])

  useEffect(() => {
    if (!sessionPickerOpen) return

    let cancelled = false

    void (async () => {
      setSessionsLoading(true)
      setSessionsError(null)
      const result = await loadSessionPickerItems({
        httpClient,
        workspaceRoot: FILE_PICKER_ROOT_DIR,
        workspaceProject,
        sessionQuery,
        cancelled: () => cancelled,
        refreshProjectSessionIds,
      })
      if (cancelled) return
      if (result.ok && !result.cancelled) {
        setSessionItems(result.items)
        setSessionPickerIndex(0)
      } else if (!result.ok) {
        setSessionsError(result.error)
        setSessionItems([])
      }
      setSessionsLoading(false)
    })()

    return () => {
      cancelled = true
    }
  }, [
    httpClient,
    refreshProjectSessionIds,
    sessionPickerOpen,
    sessionQuery,
    sessionPickerRetryToken,
    workspaceProject,
    setSessionItems,
    setSessionPickerIndex,
    setSessionsError,
    setSessionsLoading,
  ])

  useEffect(() => {
    if (!modePickerOpen) return
    void loadAgentModes()
  }, [loadAgentModes, modePickerOpen])

  useEffect(() => {
    if (!modelPickerOpen) return
    if (providers.length === 0) {
      // First-ever open (or after a hard failure): fetch synchronously so
      // the picker has something to show and can surface loading/error state.
      void loadProviders().catch(() => {})
      return
    }
    // Subsequent opens: real rediscovery in the background, gated by a
    // 5-minute TTL cache. The daemon re-runs discovery server-side with its
    // stored (unredacted) credentials per provider, then we reload the
    // provider list. Failures are silent and leave the existing list in
    // place; a fresh result updates it in place.
    const pickerOptions = {
      defaultProviderId: daemonDefaultProvider,
      defaultModelId: daemonDefaultModel,
      mru: modelMru,
    }
    void refreshModelsInBackground(
      modelRefreshStateRef.current,
      () =>
        refreshProviderModelLists(
          httpClient,
          providers.map((provider) => provider.id),
        ),
      (nextProviders) => {
        // Remap the selection so a changed item list does not silently move
        // the highlight onto a different provider/model.
        const previousOrdered = buildProviderModelPickerList(
          providers,
          state.provider,
          state.model,
          modelQuery,
          pickerOptions,
        ).ordered
        const nextOrdered = buildProviderModelPickerList(
          nextProviders,
          state.provider,
          state.model,
          modelQuery,
          pickerOptions,
        ).ordered
        setModelPickerIndex((index) => remapPickerIndex(previousOrdered, nextOrdered, index))
        setProviders(nextProviders)
      },
    )
  }, [
    daemonDefaultModel,
    daemonDefaultProvider,
    httpClient,
    loadProviders,
    modelMru,
    modelQuery,
    modelPickerOpen,
    providers,
    setModelPickerIndex,
    setProviders,
    state.model,
    state.provider,
  ])

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const index = await getFileIndex(FILE_PICKER_ROOT_DIR)
        if (cancelled) return
        setFileIndexPaths(new Set(index.entries.map((entry) => entry.path)))
      } catch {
        // Leave the set empty; normalization simply becomes a no-op.
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (
      !activeAttachmentReference ||
      activeSkillReference ||
      sessionPickerOpen ||
      filePickerOpen ||
      modelPickerOpen ||
      modePickerOpen ||
      autonomyPickerOpen ||
      skillManagerOpen ||
      skillStoreOpen
    ) {
      setAttachmentSuggestions([])
      setAttachmentSuggestionIndex(0)
      return
    }

    let cancelled = false
    const reference = activeAttachmentReference

    void (async () => {
      const result = await loadAttachmentSuggestions({
        referencePath: reference.path,
        rootDir: FILE_PICKER_ROOT_DIR,
        cancelled: () => cancelled,
      })
      if (cancelled) return
      if (result.ok && !result.cancelled) {
        setAttachmentSuggestions(result.items)
      } else if (!result.ok) {
        setAttachmentSuggestions([])
      }
      setAttachmentSuggestionIndex(0)
    })()
    return () => {
      cancelled = true
    }
  }, [
    activeAttachmentReference,
    activeSkillReference,
    autonomyPickerOpen,
    filePickerOpen,
    modelPickerOpen,
    modePickerOpen,
    sessionPickerOpen,
    skillManagerOpen,
    skillStoreOpen,
    setAttachmentSuggestionIndex,
    setAttachmentSuggestions,
  ])

  useEffect(() => {
    if (!filePickerOpen) return

    let cancelled = false

    void (async () => {
      setFilePickerLoading(true)
      setFilePickerError(null)
      const result = await loadFilePickerItems({
        dir: filePickerDir,
        cancelled: () => cancelled,
      })
      if (cancelled) return
      if (result.ok && !result.cancelled) {
        setFilePickerItems(result.items)
        setFilePickerIndex(0)
      } else if (!result.ok) {
        setFilePickerItems([])
        setFilePickerError(result.error)
      }
      setFilePickerLoading(false)
    })()
    return () => {
      cancelled = true
    }
  }, [
    filePickerDir,
    filePickerOpen,
    setFilePickerError,
    setFilePickerIndex,
    setFilePickerItems,
    setFilePickerLoading,
  ])

  const openSessionPicker = useCallback(() => {
    const decision = decideOpenOverlayGate({
      isStreaming: state.isStreaming,
      hasPendingApproval: Boolean(state.pendingApproval),
      action: 'switching sessions',
      overlayState,
      target: 'session-picker',
    })
    if (decision.kind === 'block') {
      dispatch({ type: 'SYSTEM_MESSAGE', content: decision.message })
      return
    }
    setInputValue('')
    setSessionQuery('')
    setSessionPickerIndex(0)
    setSessionPickerOpen(true)
  }, [dispatch, overlayState, state.isStreaming, state.pendingApproval])

  const retrySessionPicker = useCallback(() => {
    setSessionPickerIndex(0)
    setSessionPickerRetryToken((current) => current + 1)
  }, [setSessionPickerIndex])

  const openAutonomyPicker = useCallback(() => {
    const decision = decideOpenOverlayGate({
      isStreaming: state.isStreaming,
      hasPendingApproval: Boolean(state.pendingApproval),
      action: 'switching autonomy',
      overlayState,
      target: 'autonomy-picker',
    })
    if (decision.kind === 'block') {
      dispatch({ type: 'SYSTEM_MESSAGE', content: decision.message })
      return
    }
    setInputValue('')
    setAutonomyPickerIndex(0)
    setAutonomyPickerOpen(true)
  }, [dispatch, overlayState, state.isStreaming, state.pendingApproval])

  const openModelPicker = useCallback(
    (target: 'session' | 'default' = 'session') => {
      const decision = decideOpenOverlayGate({
        isStreaming: state.isStreaming,
        hasPendingApproval: Boolean(state.pendingApproval),
        action: 'opening the model picker',
        overlayState,
        target: 'model-picker',
      })
      if (decision.kind === 'block') {
        dispatch({ type: 'SYSTEM_MESSAGE', content: decision.message })
        return
      }
      setInputValue('')
      setModelQuery('')
      focusModelPickerSelection(target)
      setModelPickerOpen(true)
      void readModelMru()
        .then(setModelMru)
        .catch(() => undefined)
    },
    [dispatch, focusModelPickerSelection, overlayState, state.isStreaming, state.pendingApproval],
  )

  const openModePicker = useCallback(() => {
    const decision = decideOpenOverlayGate({
      isStreaming: state.isStreaming,
      hasPendingApproval: Boolean(state.pendingApproval),
      action: 'switching modes',
      overlayState,
      target: 'mode-picker',
    })
    if (decision.kind === 'block') {
      dispatch({ type: 'SYSTEM_MESSAGE', content: decision.message })
      return
    }
    setInputValue('')
    setModeQuery('')
    setModePickerIndex(0)
    setModePickerOpen(true)
  }, [dispatch, overlayState, state.isStreaming, state.pendingApproval])

  const openFilePicker = useCallback(() => {
    const decision = decideOpenOverlayGate({
      isStreaming: state.isStreaming,
      hasPendingApproval: Boolean(state.pendingApproval),
      action: 'attaching files',
      overlayState,
      target: 'file-picker',
    })
    if (decision.kind === 'block') {
      dispatch({ type: 'SYSTEM_MESSAGE', content: decision.message })
      return
    }
    setFilePickerOpen(true)
    setFilePickerError(null)
  }, [dispatch, overlayState, state.isStreaming, state.pendingApproval])

  const openSkillStore = useCallback(
    (initialQuery = '') => {
      const decision = decideOpenOverlayGate({
        isStreaming: state.isStreaming,
        hasPendingApproval: Boolean(state.pendingApproval),
        action: 'opening the skill catalog',
        overlayState,
        target: 'skill-store',
      })
      if (decision.kind === 'block') {
        dispatch({ type: 'SYSTEM_MESSAGE', content: decision.message })
        return
      }

      const query = initialQuery.trim()
      setInputValue('')
      setSkillStoreOpen(true)
      setSkillStoreQuery(query)
      setSkillStoreSearchedQuery('')
      setSkillStoreResults([])
      setSkillStoreIndex(0)
      setSkillStoreLoading(false)
      setSkillStoreInstallingSource(null)
      setSkillStoreError(null)
      setSkillStoreMessage(
        query ? null : 'Type a query and press Enter to search configured skill sources.',
      )
      if (query) {
        void searchSkillStore(query)
      } else {
        exclusivePanel.begin('skill-store')
      }
    },
    [
      dispatch,
      exclusivePanel,
      overlayState,
      searchSkillStore,
      setSkillStoreError,
      setSkillStoreIndex,
      setSkillStoreInstallingSource,
      setSkillStoreLoading,
      setSkillStoreMessage,
      setSkillStoreOpen,
      setSkillStoreQuery,
      setSkillStoreResults,
      setSkillStoreSearchedQuery,
      state.isStreaming,
      state.pendingApproval,
    ],
  )

  const openSkillManager = useCallback(
    (initialQuery = '') => {
      const decision = decideOpenOverlayGate({
        isStreaming: state.isStreaming,
        hasPendingApproval: Boolean(state.pendingApproval),
        action: 'managing installed skills',
        overlayState,
        target: 'skill-manager',
      })
      if (decision.kind === 'block') {
        dispatch({ type: 'SYSTEM_MESSAGE', content: decision.message })
        return
      }

      const query = initialQuery.trim()
      const ticket = exclusivePanel.begin('skill-manager')
      setInputValue('')
      setSkillManagerOpen(true)
      setSkillManagerQuery(query)
      setSkillManagerIndex(0)
      setSkillManagerTogglingId(null)
      setSkillManagerError(null)
      setSkillManagerMessage('Loading installed skills...')

      void refreshInstalledSkills().then((skills) => {
        if (!ticket.isCurrent()) return
        setSkillManagerMessage(
          skills.length > 0
            ? 'Enter toggles the selected skill. Tab prepares /run for enabled skills.'
            : null,
        )
      })
    },
    [
      dispatch,
      exclusivePanel,
      overlayState,
      refreshInstalledSkills,
      setSkillManagerError,
      setSkillManagerIndex,
      setSkillManagerMessage,
      setSkillManagerOpen,
      setSkillManagerQuery,
      setSkillManagerTogglingId,
      state.isStreaming,
      state.pendingApproval,
    ],
  )

  const toggleSkillFromManager = useCallback(
    async (skill: DaemonSkill) => {
      if (skillManagerTogglingId) {
        return
      }

      const nextEnabled = skill.enabled === false
      setSkillManagerTogglingId(skill.id)
      setSkillManagerError(null)
      setSkillManagerMessage(`${nextEnabled ? 'Enabling' : 'Disabling'} ${skill.name}...`)

      try {
        const result = await httpClient.setSkillEnabled(skill.id, nextEnabled)
        setInstalledSkills((current) =>
          current.map((item) =>
            item.id === result.id ? { ...item, enabled: result.enabled } : item,
          ),
        )
        setSkillManagerMessage(
          result.enabled
            ? `Enabled ${result.id}${result.id === 'container-sandbox' ? ' (requires Docker on the host)' : ''}`
            : `Disabled ${result.id}`,
        )
        void refreshInstalledSkills()
      } catch (error) {
        setSkillManagerError(error instanceof Error ? error.message : 'Failed to toggle skill.')
      } finally {
        setSkillManagerTogglingId(null)
      }
    },
    [
      httpClient,
      refreshInstalledSkills,
      setSkillManagerError,
      setSkillManagerMessage,
      setSkillManagerTogglingId,
      skillManagerTogglingId,
    ],
  )

  const runSkillFromManager = useCallback(
    (skill: DaemonSkill) => {
      if (skill.enabled === false) {
        setSkillManagerMessage('Enable the selected skill before running it.')
        return
      }
      closeSkillManagerPicker()
      setInputValue(`/run ${skill.id} `)
    },
    [closeSkillManagerPicker, setSkillManagerMessage],
  )

  const toggleQueuedAttachmentPath = useCallback((absolutePath: string) => {
    const attachmentPath = toAttachmentPath(absolutePath, FILE_PICKER_ROOT_DIR)
    setQueuedAttachmentPaths((current) =>
      current.includes(attachmentPath)
        ? current.filter((path) => path !== attachmentPath)
        : [...current, attachmentPath],
    )
  }, [])

  const startNewSession = useCallback(() => {
    closeFilePicker()
    closeAutonomyPicker()
    closeModelPicker()
    closeProviderSetup()
    closeModePicker()
    closeSessionPicker()
    closeSkillManagerPicker()
    closeSkillStorePicker()
    resetTokenSpeedStats()
    setQueuedAttachmentPaths([])
    setTranscriptClearedAt(null)
    dispatch({ type: 'NEW_SESSION' })
  }, [
    closeAutonomyPicker,
    closeFilePicker,
    closeModePicker,
    closeModelPicker,
    closeProviderSetup,
    closeSessionPicker,
    closeSkillManagerPicker,
    closeSkillStorePicker,
    dispatch,
    resetTokenSpeedStats,
  ])

  const loadSessionById = useCallback(
    async (sessionId: string) => {
      dispatch({ type: 'SET_HYDRATING_SESSION', value: true })
      try {
        const [session, artifacts] = await Promise.all([
          httpClient.session(sessionId),
          httpClient.sessionArtifacts(sessionId).catch(() => []),
        ])
        const workspaceError = sessionWorkspaceLoadError(
          session,
          FILE_PICKER_ROOT_DIR,
        )
        if (workspaceError) {
          dispatch({ type: 'ERROR', message: workspaceError })
          dispatch({ type: 'SET_HYDRATING_SESSION', value: false })
          return
        }
        setQueuedAttachmentPaths([])
        setTranscriptClearedAt(null)
        setProjectSelectionSource(null)
        loadSession(session, artifacts)
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: buildSessionLoadedSummary(session),
        })
        closeAutonomyPicker()
        closeModelPicker()
        closeProviderSetup()
        closeModePicker()
        closeSessionPicker()
      } catch (error) {
        if (
          workspaceProjectRef.current &&
          error instanceof Error &&
          error.message.includes('404')
        ) {
          void removeProjectSession(workspaceProjectRef.current, sessionId)
            .then((registry) => {
              setProjectSessionIds(registry.sessions.map((session) => session.sessionId))
            })
            .catch(() => {})
        }
        dispatch({
          type: 'ERROR',
          message:
            error instanceof Error
              ? `Failed to load session ${sessionId}: ${error.message}`
              : `Failed to load session ${sessionId}`,
        })
        dispatch({ type: 'SET_HYDRATING_SESSION', value: false })
      }
    },
    [
      closeAutonomyPicker,
      closeModePicker,
      closeModelPicker,
      closeProviderSetup,
      closeSessionPicker,
      dispatch,
      httpClient,
      loadSession,
    ],
  )

  const navigateRecentSession = useCallback(
    async (direction: 'older' | 'newer') => {
      const decision = decideOpenOverlayGate({
        isStreaming: state.isStreaming,
        hasPendingApproval: Boolean(state.pendingApproval),
        action: 'switching sessions',
        overlayState,
        target: 'session-picker',
      })
      if (decision.kind === 'block') {
        dispatch({ type: 'SYSTEM_MESSAGE', content: decision.message })
        return
      }

      const targetSessionId = findAdjacentRecentSessionId(
        recentSessionIds,
        state.sessionId,
        direction,
      )
      if (!targetSessionId) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content:
            direction === 'older'
              ? 'No older recent session is available.'
              : 'No newer recent session is available.',
        })
        return
      }

      await loadSessionById(targetSessionId)
    },
    [
      dispatch,
      loadSessionById,
      overlayState,
      recentSessionIds,
      state.isStreaming,
      state.pendingApproval,
      state.sessionId,
    ],
  )

  const rewindCurrentSession = useCallback(
    async (turns = 1) => {
      if (!state.sessionId) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: 'No active session to rewind.',
        })
        return
      }

      const decision = decideStreamApprovalGate({
        isStreaming: state.isStreaming,
        hasPendingApproval: Boolean(state.pendingApproval),
        action: 'rewinding',
      })
      if (decision.kind === 'block') {
        dispatch({ type: 'SYSTEM_MESSAGE', content: decision.message })
        return
      }

      try {
        const session = await httpClient.session(state.sessionId)
        const target = findRewindTarget(session.events, turns)
        if (target === null) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'No user messages found in the current session yet.',
          })
          return
        }

        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: buildRewindPlanSummary(session, target),
        })
        const sourceSessionId = state.sessionId
        const branch = await httpClient.branchSession(state.sessionId, {
          fromEventIndex: target.fromEventIndex,
        })
        await loadSessionById(branch.branchId)
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: buildRewindSuccessSummary(
            sourceSessionId,
            branch.branchId,
            target.turns,
            branch.copiedEvents,
            target,
          ),
        })
      } catch (error) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: error instanceof Error ? error.message : 'Failed to rewind the current session.',
        })
      }
    },
    [
      dispatch,
      httpClient,
      loadSessionById,
      state.isStreaming,
      state.pendingApproval,
      state.sessionId,
    ],
  )

  const branchSessionAtTail = useCallback(
    async (sourceSessionId: string) => {
      try {
        const branch = await httpClient.branchSession(sourceSessionId)
        await loadSessionById(branch.branchId)
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: buildBranchSuccessSummary(sourceSessionId, branch.branchId, branch.copiedEvents),
        })
      } catch (error) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content:
            error instanceof Error ? error.message : 'Failed to branch the selected session.',
        })
      }
    },
    [dispatch, httpClient, loadSessionById],
  )

  const branchCurrentSession = useCallback(async () => {
    if (!state.sessionId) {
      dispatch({
        type: 'SYSTEM_MESSAGE',
        content: 'No active session to branch.',
      })
      return
    }

    const decision = decideStreamApprovalGate({
      isStreaming: state.isStreaming,
      hasPendingApproval: Boolean(state.pendingApproval),
      action: 'branching sessions',
    })
    if (decision.kind === 'block') {
      dispatch({ type: 'SYSTEM_MESSAGE', content: decision.message })
      return
    }

    await branchSessionAtTail(state.sessionId)
  }, [branchSessionAtTail, dispatch, state.isStreaming, state.pendingApproval, state.sessionId])

  const exportSessionById = useCallback(
    async (sessionId: string, args: string[]) => {
      const { format, outputPath } = parseSessionExportArgs(args)
      const targetPath = resolve(
        process.cwd(),
        outputPath ?? buildSessionExportFilename(sessionId, format),
      )

      try {
        const serialized =
          format === 'json'
            ? JSON.stringify(await httpClient.sessionExport(sessionId, 'json'), null, 2)
            : await httpClient.sessionExport(sessionId, 'markdown')
        await writeFile(targetPath, serialized, 'utf-8')
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Exported session ${sessionId.slice(0, 8)} to ${targetPath} (${format}).`,
        })
      } catch (error) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content:
            error instanceof Error ? error.message : 'Failed to export the selected session.',
        })
      }
    },
    [dispatch, httpClient],
  )

  const exportCurrentSession = useCallback(
    async (args: string[]) => {
      if (!state.sessionId) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: 'No active session to export.',
        })
        return
      }

      await exportSessionById(state.sessionId, args)
    },
    [dispatch, exportSessionById, state.sessionId],
  )

  const compactSessionById = useCallback(
    async (sessionId: string) => {
      if (sessionId === state.sessionId && state.isStreaming) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: 'Wait for the current run to finish before compacting the active session.',
        })
        return
      }

      if (sessionId === state.sessionId && state.pendingApproval) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: 'Resolve the pending approval before compacting the active session.',
        })
        return
      }

      try {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Compacting session ${sessionId.slice(0, 8)}...`,
        })
        const compactResult = await httpClient.compactSession(sessionId)
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: [
            `Session ${sessionId.slice(0, 8)} compacted: ${compactResult.originalTokens.toLocaleString()} -> ${compactResult.compactedTokens.toLocaleString()} tokens (saved ${compactResult.savedTokens.toLocaleString()})`,
            compactResult.removedMessageCount || compactResult.preservedMessageCount
              ? [
                  compactResult.removedMessageCount
                    ? `Compacted ${compactResult.removedMessageCount.toLocaleString()} earlier messages`
                    : null,
                  compactResult.preservedMessageCount
                    ? `preserved ${compactResult.preservedMessageCount.toLocaleString()} recent messages`
                    : null,
                  compactResult.strategy === 'summary_only' ? '(summary only fallback)' : null,
                ]
                  .filter(Boolean)
                  .join(', ')
              : null,
            compactResult.summary || null,
          ]
            .filter(Boolean)
            .join('\n'),
        })
      } catch (error) {
        dispatch({
          type: 'ERROR',
          message: error instanceof Error ? `Compact failed: ${error.message}` : 'Compact failed.',
        })
      }
    },
    [dispatch, httpClient, state.isStreaming, state.pendingApproval, state.sessionId],
  )

  const showDoctorSummary = useCallback(async () => {
    dispatch({
      type: 'SYSTEM_MESSAGE',
      content: 'Running doctor summary...',
    })

    try {
      const [health, freshProviders, usage] = await Promise.all([
        httpClient.health(),
        httpClient.providers().catch(() => providers),
        httpClient.usage().catch(() => null),
      ])
      dispatch({
        type: 'SYSTEM_MESSAGE',
        content: buildDoctorSummary({
          health,
          providers: freshProviders,
          usage,
          currentSessionId: state.sessionId,
        }),
      })
    } catch (error) {
      dispatch({
        type: 'SYSTEM_MESSAGE',
        content: buildDoctorUnavailableSummary({
          error,
          baseUrl: httpClient.baseUrl,
        }),
      })
    }
  }, [dispatch, httpClient, providers, state.sessionId])

  const removeSessionReferences = useCallback(async (sessionId: string) => {
    setSessionItems((current) => current.filter((session) => session.id !== sessionId))
    setProjectSessionIds((current) => current.filter((id) => id !== sessionId))

    try {
      const cliState = await removeSessionAccess(sessionId)
      setRecentSessionIds(cliState.recentSessionIds ?? [])
    } catch {
      setRecentSessionIds((current) => current.filter((id) => id !== sessionId))
    }

    const workspace = workspaceProjectRef.current
    if (!workspace) {
      return
    }

    try {
      const registry = await removeProjectSession(workspace, sessionId)
      setProjectSessionIds(registry.sessions.map((session) => session.sessionId))
    } catch {
      setProjectSessionIds((current) => current.filter((id) => id !== sessionId))
    }
  }, [])

  const deleteSessionById = useCallback(
    async (sessionId: string) => {
      try {
        await httpClient.deleteSession(sessionId)
        await removeSessionReferences(sessionId)

        if (state.sessionId === sessionId) {
          startNewSession()
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Deleted active session ${sessionId.slice(0, 8)}. Started a new session.`,
          })
          return
        }

        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Deleted session ${sessionId.slice(0, 8)}.`,
        })
      } catch (error) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content:
            error instanceof Error ? error.message : 'Failed to delete the selected session.',
        })
      }
    },
    [dispatch, httpClient, removeSessionReferences, startNewSession, state.sessionId],
  )

  const ensureAgentModes = useCallback(async (): Promise<{
    agents: DaemonAgentDescriptor[]
    error: string | null
  }> => {
    if (agentModes.length > 0) {
      return { agents: agentModes, error: null }
    }
    try {
      const fetched = await httpClient.agents()
      setAgentModes(fetched)
      return { agents: fetched, error: null }
    } catch (error) {
      return {
        agents: [],
        error: error instanceof Error ? error.message : 'Failed to load agent modes.',
      }
    }
  }, [agentModes, httpClient])

  const selectAgentMode = useCallback(
    (agent: DaemonAgentDescriptor, source: 'picker' | 'command' = 'command') => {
      dispatch({
        type: 'SET_MODE',
        mode: agent.id,
      })
      if (source === 'picker') {
        closeModePicker()
      }
      dispatch({
        type: 'SYSTEM_MESSAGE',
        content: `Mode set to ${agent.id}${agent.name && agent.name !== agent.id ? ` (${agent.name})` : ''}`,
      })
    },
    [closeModePicker, dispatch],
  )

  const cycleAgentMode = useCallback(async () => {
    if (
      autonomyPickerOpen ||
      filePickerOpen ||
      modelPickerOpen ||
      providerSetupOpen ||
      providerDeleteOpen ||
      modePickerOpen ||
      sessionPickerOpen ||
      skillManagerOpen ||
      skillStoreOpen ||
      state.isStreaming ||
      state.pendingApproval
    ) {
      return
    }

    const ensured = await ensureAgentModes()
    if (ensured.error) {
      dispatch({ type: 'SYSTEM_MESSAGE', content: ensured.error })
      return
    }

    if (ensured.agents.length === 0) {
      dispatch({
        type: 'SYSTEM_MESSAGE',
        content: 'No agent modes available.',
      })
      return
    }

    const nextId = nextAgentModeId(state.mode, ensured.agents)
    if (!nextId) {
      return
    }
    const nextAgent = ensured.agents.find((agent) => agent.id === nextId)
    if (!nextAgent) {
      return
    }

    selectAgentMode(nextAgent, 'command')
  }, [
    autonomyPickerOpen,
    dispatch,
    ensureAgentModes,
    filePickerOpen,
    modelPickerOpen,
    modePickerOpen,
    providerDeleteOpen,
    providerSetupOpen,
    selectAgentMode,
    sessionPickerOpen,
    skillManagerOpen,
    skillStoreOpen,
    state.isStreaming,
    state.mode,
    state.pendingApproval,
  ])

  const rememberQueuedFollowUp = useCallback((sessionId: string, noteId: string) => {
    queuedFollowUpsRef.current = [
      ...queuedFollowUpsRef.current.filter((queued) => queued.noteId !== noteId),
      { sessionId, noteId },
    ].slice(-20)
  }, [])

  const cancelQueuedFollowUps = useCallback(async (
    target: { selector: 'latest' | 'all' } | { noteId: string },
  ) => {
    const sessionId = state.sessionId
    if (!sessionId || followUpCancelInFlightRef.current) return
    followUpCancelInFlightRef.current = true
    try {
      const result = await cancelSteer(httpClient, sessionId, target)
      if (result.ok) {
        const cancelledIds = new Set(result.cancelledNoteIds)
        queuedFollowUpsRef.current = queuedFollowUpsRef.current.filter(
          (queued) => !cancelledIds.has(queued.noteId),
        )
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: formatCancelledSteerNote(
            result.cancelledNoteIds,
            result.pendingSteeringNoteCount,
          ),
        })
        return
      }
      if ('noteId' in target && result.reason !== 'error') {
        queuedFollowUpsRef.current = queuedFollowUpsRef.current.filter(
          (queued) => queued.noteId !== target.noteId,
        )
      } else if (['no_pending', 'no_active_run'].includes(result.reason)) {
        queuedFollowUpsRef.current = queuedFollowUpsRef.current.filter(
          (queued) => queued.sessionId !== sessionId,
        )
      }
      dispatch({ type: 'SYSTEM_MESSAGE', content: result.message })
    } finally {
      followUpCancelInFlightRef.current = false
    }
  }, [dispatch, httpClient, state.sessionId])

  useEffect(() => {
    if (state.isStreaming || !state.sessionId) return
    queuedFollowUpsRef.current = queuedFollowUpsRef.current.filter(
      (queued) => queued.sessionId !== state.sessionId,
    )
  }, [state.isStreaming, state.sessionId])

  // Stateless commands live in this registry. The dispatcher checks it first
  // and only falls through to the legacy switch for commands that still need
  // closure access to large blocks of local state. As more commands are
  // moved here the switch shrinks toward zero. Adding a new stateless
  // command no longer requires editing the switch (OCP).
  const slashCommandRegistry = useMemo(() => {
    const registry = new SlashCommandRegistry()
    registry.register({
      name: '/help',
      description: 'Show keyboard shortcuts and help',
      handler: () => {
        setInputValue('')
        setHelpOpen(true)
      },
    })
    registry.register({
      name: '/new',
      description: 'Start a new session',
      handler: () => {
        startNewSession()
      },
    })
    registry.register({
      name: '/clear',
      description: 'Clear the on-screen transcript (session context unchanged)',
      handler: () => {
        setTranscriptClearedAt(Date.now())
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content:
            'Transcript cleared locally. Session context is still active; use /new for a fresh session.',
        })
      },
    })
    registry.register({
      name: '/steer',
      description: "Send a mid-run steering note to the current session's active turn",
      handler: async ({ args }) => {
        const message = args.join(' ').trim()
        if (!message) {
          dispatch({ type: 'SYSTEM_MESSAGE', content: 'Usage: /steer <message>' })
          return
        }
        if (!state.sessionId) {
          dispatch({ type: 'SYSTEM_MESSAGE', content: 'No active session yet — send a message first.' })
          return
        }
        const result = await submitSteer(httpClient, state.sessionId, message)
        if (result.ok) {
          rememberQueuedFollowUp(state.sessionId, result.noteId)
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `[steer] ${formatQueuedSteerNote(
              result.noteId,
              result.pendingSteeringNoteCount,
            )} · Esc to undo`,
          })
        } else if (result.noActiveRun) {
          dispatch({ type: 'SYSTEM_MESSAGE', content: result.guidance })
        } else {
          dispatch({ type: 'SYSTEM_MESSAGE', content: `Failed to steer: ${result.message}` })
        }
      },
    })
    registry.register({
      name: '/followup',
      description: 'Cancel queued follow-ups for the active run',
      handler: async ({ args }) => {
        const [action = '', target = 'latest', ...extra] = args
        if (!['cancel', 'undo'].includes(action.toLowerCase()) || extra.length > 0) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'Usage: /followup cancel [latest|all|<note-id>]',
          })
          return
        }
        await cancelQueuedFollowUps(
          target === 'latest' || target === 'all'
            ? { selector: target }
            : { noteId: target },
        )
      },
    })
    registry.register({
      name: '/mcp',
      description: 'Toggle MCP status, or manage MCP servers',
      handler: async ({ args }) => {
        if (args.length === 0) {
          setShowMcp((prev) => !prev)
          return
        }
        try {
          const content = await runMcpSlashCommand(httpClient, args)
          dispatch({ type: 'SYSTEM_MESSAGE', content })
        } catch (error) {
          dispatch({ type: 'ERROR', message: normalizeError(error) })
        }
      },
    })
    registry.register({
      name: '/acp',
      description: 'Inspect ACP setup for editor and external-agent integration',
      handler: async ({ args }) => {
        const action = args[0]?.toLowerCase() ?? 'help'
        const daemonUrl = process.env.SEPILOTD_URL
        if (['help', '--help', '-h'].includes(action)) {
          dispatch({ type: 'SYSTEM_MESSAGE', content: ACP_COMMAND_USAGE })
          return
        }
        if (action === 'config') {
          dispatch({ type: 'SYSTEM_MESSAGE', content: buildAcpConfigSnippet(daemonUrl) })
          return
        }
        if (action === 'opencode') {
          dispatch({ type: 'SYSTEM_MESSAGE', content: buildAcpOpencodeGuide() })
          return
        }
        if (action === 'codex') {
          dispatch({ type: 'SYSTEM_MESSAGE', content: buildAcpCodexGuide() })
          return
        }
        if (action === 'zed') {
          dispatch({ type: 'SYSTEM_MESSAGE', content: buildAcpZedHint(daemonUrl) })
          return
        }
        if (action === 'status' || action === 'current' || action === 'info') {
          try {
            const health = await httpClient.health()
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: [
                'ACP status',
                `Daemon: ${health.status} (${health.version})`,
                `Command: sepilot acp${daemonUrl ? ` --url ${daemonUrl}` : ''}`,
                'Protocol: JSON-RPC 2.0 over stdio Content-Length framing',
                'Methods: initialize, session/new, session/prompt, session/cancel',
              ].join('\n'),
            })
          } catch (error) {
            dispatch({ type: 'ERROR', message: normalizeError(error) })
          }
          return
        }
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Unknown /acp action: ${action}\n\n${ACP_COMMAND_USAGE}`,
        })
      },
    })
    registry.register({
      name: '/a2a',
      description: 'Inspect A2A setup for agent-to-agent integration',
      handler: async ({ args }) => {
        const action = args[0]?.toLowerCase() ?? 'help'
        const daemonUrl = process.env.SEPILOTD_URL
        if (['help', '--help', '-h'].includes(action)) {
          dispatch({ type: 'SYSTEM_MESSAGE', content: A2A_COMMAND_USAGE })
          return
        }
        if (action === 'card' || action === 'config') {
          dispatch({ type: 'SYSTEM_MESSAGE', content: buildA2aGuide(daemonUrl) })
          return
        }
        if (action === 'send') {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: [
              'Use the daemon tool `a2a.send` from an agent turn.',
              'Required input: agentCardUrl, message.',
              'Optional input: headers, timeoutMs.',
            ].join('\n'),
          })
          return
        }
        if (action === 'status' || action === 'current' || action === 'info') {
          try {
            const health = await httpClient.health()
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: [
                'A2A status',
                `Daemon: ${health.status} (${health.version})`,
                'Protocol: Agent2Agent JSON-RPC binding',
                'Version: 1.0',
                'Methods: SendMessage, GetTask, ListTasks, CancelTask',
                `Agent Card: ${(daemonUrl ?? '').replace(/\/$/, '') || '<daemon-origin>'}/.well-known/agent-card.json`,
              ].join('\n'),
            })
          } catch (error) {
            dispatch({ type: 'ERROR', message: normalizeError(error) })
          }
          return
        }
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Unknown /a2a action: ${action}\n\n${A2A_COMMAND_USAGE}`,
        })
      },
    })
    registry.register({
      name: '/hooks',
      description: 'Manage outbound delivery hooks',
      handler: async ({ args }) => {
        const subcommand = (args[0] ?? 'list').toLowerCase()
        try {
          if (['list', 'ls', 'current', 'info'].includes(subcommand)) {
            const webhooks = await httpClient.outboundWebhooks()
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content:
                webhooks.length > 0
                  ? webhooks.map(formatTuiHookSummary).join('\n')
                  : `No outbound hooks configured.\n${buildTuiHooksUsage()}`,
            })
            return
          }

          if (subcommand === 'add') {
            const rest = args.slice(1)
            const url = rest.find((entry) => !entry.startsWith('--'))
            if (!url) {
              dispatch({ type: 'SYSTEM_MESSAGE', content: buildTuiHooksUsage() })
              return
            }
            const events: string[] = []
            const headerEntries: string[] = []
            let secret: string | undefined
            let retryAttempts: number | undefined
            let retryBackoffMs: number | undefined
            let disabled = false
            for (let index = 0; index < rest.length; ) {
              const arg = rest[index]!
              if (!arg.startsWith('--')) {
                index += 1
                continue
              }
              if (arg === '--disabled') {
                disabled = true
                index += 1
                continue
              }
              if (arg === '--event' || arg.startsWith('--event=')) {
                const parsed = parseTuiHookValueFlag(rest, index, '--event')
                if (parsed.value) events.push(parsed.value)
                index = parsed.nextIndex
                continue
              }
              if (arg === '--header' || arg.startsWith('--header=')) {
                const parsed = parseTuiHookValueFlag(rest, index, '--header')
                if (parsed.value) headerEntries.push(parsed.value)
                index = parsed.nextIndex
                continue
              }
              if (arg === '--secret' || arg.startsWith('--secret=')) {
                const parsed = parseTuiHookValueFlag(rest, index, '--secret')
                secret = parsed.value ?? undefined
                index = parsed.nextIndex
                continue
              }
              if (arg === '--retry-attempts' || arg.startsWith('--retry-attempts=')) {
                const parsed = parseTuiHookValueFlag(rest, index, '--retry-attempts')
                retryAttempts = parsed.value ? Number.parseInt(parsed.value, 10) : undefined
                index = parsed.nextIndex
                continue
              }
              if (arg === '--retry-backoff-ms' || arg.startsWith('--retry-backoff-ms=')) {
                const parsed = parseTuiHookValueFlag(rest, index, '--retry-backoff-ms')
                retryBackoffMs = parsed.value ? Number.parseInt(parsed.value, 10) : undefined
                index = parsed.nextIndex
                continue
              }
              throw new Error(`Unknown /hooks add flag: ${arg}`)
            }
            await httpClient.upsertOutboundWebhook({
              enabled: !disabled,
              url,
              events: parseTuiHookEvents(events),
              ...(secret ? { secret } : {}),
              headers: parseTuiHookHeaders(headerEntries),
              retry:
                Number.isFinite(retryAttempts) || Number.isFinite(retryBackoffMs)
                  ? {
                      ...(Number.isFinite(retryAttempts) ? { maxAttempts: retryAttempts } : {}),
                      ...(Number.isFinite(retryBackoffMs) ? { backoffMs: retryBackoffMs } : {}),
                    }
                  : undefined,
            })
            const hook = (await httpClient.outboundWebhooks()).find((entry) => entry.url === url)
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: hook
                ? `Outbound hook saved.\n${formatTuiHookSummary(hook)}`
                : `Outbound hook saved: ${url}`,
            })
            return
          }

          if (['remove', 'delete', 'rm'].includes(subcommand)) {
            const id = args[1]
            if (!id) {
              dispatch({ type: 'SYSTEM_MESSAGE', content: 'Usage: /hooks remove <id>' })
              return
            }
            await httpClient.deleteOutboundWebhook(id)
            dispatch({ type: 'SYSTEM_MESSAGE', content: `Outbound hook removed: ${id}` })
            return
          }

          if (subcommand === 'enable' || subcommand === 'disable') {
            const id = args[1]
            if (!id) {
              dispatch({ type: 'SYSTEM_MESSAGE', content: `Usage: /hooks ${subcommand} <id>` })
              return
            }
            const enabled = subcommand === 'enable'
            await httpClient.setOutboundWebhookEnabled(id, enabled)
            const hook = (await httpClient.outboundWebhooks()).find((entry) => entry.id === id)
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: hook
                ? `Outbound hook ${enabled ? 'enabled' : 'disabled'}.\n${formatTuiHookSummary(hook)}`
                : `Outbound hook ${enabled ? 'enabled' : 'disabled'}: ${id}`,
            })
            return
          }

          if (subcommand === 'deliveries') {
            const rest = args.slice(1)
            let id: string | undefined
            let status: 'success' | 'error' | undefined
            let limit: number | undefined
            for (let index = 0; index < rest.length; ) {
              const arg = rest[index]!
              if (arg === '--status' || arg.startsWith('--status=')) {
                const parsed = parseTuiHookValueFlag(rest, index, '--status')
                status =
                  parsed.value === 'success' || parsed.value === 'error' ? parsed.value : undefined
                index = parsed.nextIndex
                continue
              }
              if (arg === '--limit' || arg.startsWith('--limit=')) {
                const parsed = parseTuiHookValueFlag(rest, index, '--limit')
                limit = parseTuiHookLimit(parsed.value ?? undefined)
                index = parsed.nextIndex
                continue
              }
              if (!arg.startsWith('--') && !id) {
                id = arg
              }
              index += 1
            }
            const deliveries = await httpClient.outboundWebhookDeliveries({
              id,
              status,
              limit: limit ?? 20,
            })
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content:
                deliveries.length > 0
                  ? deliveries.map(formatTuiHookDelivery).join('\n')
                  : 'No outbound hook deliveries recorded.',
            })
            return
          }

          if (subcommand === 'dead-letters') {
            const rest = args.slice(1)
            let id: string | undefined
            let stateFilter: 'open' | 'acknowledged' | 'all' | undefined = 'open'
            let limit: number | undefined
            for (let index = 0; index < rest.length; ) {
              const arg = rest[index]!
              if (arg === '--state' || arg.startsWith('--state=')) {
                const parsed = parseTuiHookValueFlag(rest, index, '--state')
                stateFilter =
                  parsed.value === 'open' ||
                  parsed.value === 'acknowledged' ||
                  parsed.value === 'all'
                    ? parsed.value
                    : 'open'
                index = parsed.nextIndex
                continue
              }
              if (arg === '--limit' || arg.startsWith('--limit=')) {
                const parsed = parseTuiHookValueFlag(rest, index, '--limit')
                limit = parseTuiHookLimit(parsed.value ?? undefined)
                index = parsed.nextIndex
                continue
              }
              if (!arg.startsWith('--') && !id) {
                id = arg
              }
              index += 1
            }
            const deadLetters = await httpClient.outboundWebhookDeadLetters({
              id,
              state: stateFilter,
              limit: limit ?? 20,
            })
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content:
                deadLetters.length > 0
                  ? deadLetters.map(formatTuiHookDeadLetter).join('\n')
                  : 'No outbound hook dead letters.',
            })
            return
          }

          if (subcommand === 'replay') {
            const deliveryId = args.find((entry, index) => index > 0 && !entry.startsWith('--'))
            if (!deliveryId) {
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: 'Usage: /hooks replay <deliveryId> [--force]',
              })
              return
            }
            const replay = await httpClient.replayOutboundWebhookDelivery(
              deliveryId,
              args.includes('--force') ? { force: true } : undefined,
            )
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: `Outbound hook replay created.\n${formatTuiHookDelivery(replay)}`,
            })
            return
          }

          if (subcommand === 'replay-failed') {
            const rest = args.slice(1)
            let id: string | undefined
            let limit: number | undefined
            let force = false
            for (let index = 0; index < rest.length; ) {
              const arg = rest[index]!
              if (arg === '--force') {
                force = true
                index += 1
                continue
              }
              if (arg === '--limit' || arg.startsWith('--limit=')) {
                const parsed = parseTuiHookValueFlag(rest, index, '--limit')
                limit = parseTuiHookLimit(parsed.value ?? undefined)
                index = parsed.nextIndex
                continue
              }
              if (!arg.startsWith('--') && !id) {
                id = arg
              }
              index += 1
            }
            const replays = await httpClient.replayOutboundWebhookDeadLetters({
              id,
              state: 'open',
              limit: limit ?? 20,
              force,
            })
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content:
                replays.length > 0
                  ? `Replayed ${replays.length} outbound hook dead letter${replays.length === 1 ? '' : 's'}.\n${replays.map(formatTuiHookDelivery).join('\n')}`
                  : 'No outbound hook dead letters were replayed.',
            })
            return
          }

          if (subcommand === 'ack') {
            const rootDeliveryId = args[1]
            if (!rootDeliveryId) {
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: 'Usage: /hooks ack <rootDeliveryId> [note...]',
              })
              return
            }
            const note = args.slice(2).join(' ').trim()
            const deadLetter = await httpClient.acknowledgeOutboundWebhookDeadLetter(
              rootDeliveryId,
              note ? { note } : undefined,
            )
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: `Outbound hook dead letter acknowledged.\n${formatTuiHookDeadLetter(deadLetter)}`,
            })
            return
          }

          dispatch({ type: 'SYSTEM_MESSAGE', content: buildTuiHooksUsage() })
        } catch (error) {
          dispatch({ type: 'ERROR', message: normalizeError(error) })
        }
      },
    })
    registry.register({
      name: '/doctor',
      description: 'Run the doctor diagnostic and print a summary',
      handler: async () => {
        await showDoctorSummary()
      },
    })
    registry.register({
      name: '/skills',
      description:
        'Open installed skill manager, search configured skill sources, or install a skill',
      handler: async ({ args }) => {
        const action = args[0]?.toLowerCase() ?? 'installed'
        try {
          if (action === 'help' || action === '--help' || action === '-h') {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: SKILLS_COMMAND_USAGE,
            })
            return
          }

          if (action === 'search' || action === 'store') {
            const query = args.slice(1).join(' ').trim()
            openSkillStore(query)
            return
          }

          if (action === 'install') {
            const source = args.slice(1).join(' ').trim()
            if (!source) {
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: `Usage: /skills install <source>\n\n${SKILLS_COMMAND_USAGE}`,
              })
              return
            }
            const result = await httpClient.installSkill({ source })
            void refreshInstalledSkills()
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: formatSkillInstallResult(result.installed),
            })
            return
          }

          if (action === 'enable' || action === 'disable') {
            const id = args.slice(1).join(' ').trim()
            if (!id) {
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: `Usage: /skills ${action} <id>\n\n${SKILLS_COMMAND_USAGE}`,
              })
              return
            }
            const result = await httpClient.setSkillEnabled(id, action === 'enable')
            void refreshInstalledSkills()
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: result.enabled
                ? `Skill enabled: ${result.id}${id === 'container-sandbox' ? ' (requires Docker on the host)' : ''}`
                : `Skill disabled: ${result.id}`,
            })
            return
          }

          if (action === 'installed' || action === 'list' || action === 'manage') {
            const query = args.slice(1).join(' ').trim()
            openSkillManager(query)
            return
          }

          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Unknown /skills action: ${action}\n\n${SKILLS_COMMAND_USAGE}`,
          })
        } catch (error) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: error instanceof Error ? error.message : 'Failed to load skills.',
          })
        }
      },
    })
    registry.register({
      name: '/files',
      description: 'Open the file picker for attachments',
      handler: () => {
        openFilePicker()
      },
    })
    registry.register({
      name: '/artifacts',
      description: 'List artifacts saved in the current session',
      handler: () => {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content:
            state.artifacts.length > 0
              ? state.artifacts
                  .slice()
                  .reverse()
                  .map(
                    (artifact) =>
                      `${artifactLabel(artifact)} [${artifact.type}${artifact.language ? `/${artifact.language}` : ''}]\n  ${artifactPreview(artifact, 120)}`,
                  )
                  .join('\n')
              : 'No saved artifacts in the current session.',
        })
      },
    })
    registry.register({
      name: '/thinking',
      description: 'Set the thinking budget (off|low|medium|high|max)',
      handler: ({ args }) => {
        const level = args[0]
        if (!level || !['off', 'low', 'medium', 'high', 'max'].includes(level)) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'Usage: /thinking <off|low|medium|high|max>',
          })
          return
        }
        dispatch({
          type: 'SET_THINKING_LEVEL',
          thinkingLevel: level as 'off' | 'low' | 'medium' | 'high' | 'max',
        })
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Thinking level set to ${level}`,
        })
      },
    })
    registry.register({
      name: '/max-tokens',
      description: 'Set or clear the per-session max-output-tokens override',
      handler: ({ args }) => {
        const arg = args[0]?.toLowerCase()
        if (!arg || arg === 'current' || arg === 'info') {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content:
              state.maxTokens === null
                ? 'Max output tokens: provider default (no session override)'
                : `Max output tokens: ${state.maxTokens.toLocaleString()} (session override)`,
          })
          return
        }
        if (arg === 'off' || arg === 'none' || arg === 'default' || arg === 'clear') {
          dispatch({ type: 'SET_MAX_TOKENS', maxTokens: null })
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'Cleared max-tokens override; using provider default.',
          })
          return
        }
        const parsed = Number.parseInt(args[0] ?? '', 10)
        if (!Number.isFinite(parsed) || parsed <= 0) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'Usage: /max-tokens <positive integer|off|current>',
          })
          return
        }
        dispatch({ type: 'SET_MAX_TOKENS', maxTokens: parsed })
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Max output tokens set to ${parsed.toLocaleString()} (session override).`,
        })
      },
    })
    registry.register({
      name: '/run',
      description: 'Run a named skill with explicit skill context',
      handler: async ({ args }) => {
        const skillName = args[0]?.trim()
        if (!skillName) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'Usage: /run <skill> [prompt]',
          })
          return
        }
        const prompt = args.slice(1).join(' ').trim() || `Execute the "${skillName}" skill`
        const sent = await sendMessage(
          prompt,
          pendingAttachmentPaths.map((path) => ({ path })),
          {
            displayContent: prompt,
            skillRefs: [{ name: skillName }],
          },
        )
        if (sent) {
          setQueuedAttachmentPaths([])
        }
      },
    })
    registry.register({
      name: '/remember',
      description: 'Save a free-form memory entry from the current input',
      handler: async ({ args }) => {
        const content = args.join(' ').trim()
        if (!content) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'Usage: /remember <content>',
          })
          return
        }
        try {
          const result = await httpClient.addMemory(content)
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Saved memory ${result.id.slice(0, 8)} from manual input.`,
          })
        } catch (error) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: error instanceof Error ? error.message : 'Failed to save memory.',
          })
        }
      },
    })
    registry.register({
      name: '/theme',
      description: 'Toggle theme or set explicitly with /theme <id>',
      handler: ({ args }) => {
        const query = args.join(' ').trim()
        if (!query) {
          const nextThemeId = resolveThemeId('toggle', themeId)
          if (!nextThemeId) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: 'Failed to resolve the next theme preset.',
            })
            return
          }
          applyThemeSelection(nextThemeId)
          return
        }
        if (['current', 'info'].includes(query.toLowerCase())) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: [
              `Current theme: ${getThemeOption(themeId).label} (${themeId})`,
              `Available: ${themeOptions.map((theme) => `${theme.id} (${theme.label})`).join(', ')}`,
              'Use /theme to cycle, /theme prev to go back, or /theme <id> to switch directly.',
            ].join('\n'),
          })
          return
        }
        const nextThemeId = resolveThemeId(query, themeId)
        if (!nextThemeId) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Usage: /theme [${themeOptions.map((theme) => theme.id).join('|')}|prev|current]`,
          })
          return
        }
        applyThemeSelection(nextThemeId)
      },
    })
    registry.register({
      name: '/compact',
      description: 'Compact the current session to free context tokens',
      handler: async () => {
        if (!state.sessionId) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'No active session to compact.',
          })
          return
        }
        await compactSessionById(state.sessionId)
      },
    })
    const resumeOrContinueHandler: SlashCommandHandler = async ({ args }) => {
      if (!state.sessionId) {
        try {
          const result = await httpClient.sessions(undefined, {
            perPage: 100,
            workspaceRoot: FILE_PICKER_ROOT_DIR,
          })
          const sorted = result.items
            .filter((session) => (
              sessionBelongsToWorkspace(session, FILE_PICKER_ROOT_DIR)
            ))
            .slice()
            .sort(
              (a: { updatedAt: string }, b: { updatedAt: string }) =>
                Date.parse(b.updatedAt) - Date.parse(a.updatedAt),
            )
          if (sorted.length === 0) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: 'No previous sessions found.',
            })
          } else {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: buildRecentResumeSummary(sorted[0]!, sorted.length),
            })
            await loadSessionById(sorted[0]!.id)
          }
        } catch (error) {
          dispatch({
            type: 'ERROR',
            message:
              error instanceof Error
                ? `Failed to list sessions: ${error.message}`
                : 'Failed to list sessions.',
          })
        }
        return
      }
      const forceResume = args.includes('--force')
      try {
        const session = await httpClient.session(state.sessionId)
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: buildResumePreflightSummary(session, forceResume),
        })
        // No run checkpoint on the daemon → the resume endpoint is
        // guaranteed to 404. Don't fire a doomed request; the preflight
        // summary above already tells the user to just keep typing.
        if (!session.resumableRun && !forceResume) {
          return
        }
      } catch {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content:
            'Resume Checkpoint\nSession  current\nSafety   failed to inspect checkpoint; daemon will attempt resume directly',
        })
      }
      await resumeSession(forceResume)
    }
    registry.register({
      name: '/resume',
      description: 'Resume the last session, or pre-flight the current one',
      handler: resumeOrContinueHandler,
    })
    registry.register({
      name: '/continue',
      description: 'Alias for /resume',
      handler: resumeOrContinueHandler,
    })
    registry.register({
      name: '/rewind',
      description: 'Rewind the current session by N user turns (default 1)',
      handler: async ({ args }) => {
        const turnsArg = args[0]?.trim()
        const turns = turnsArg ? Number.parseInt(turnsArg, 10) : 1
        if (turnsArg && turnsArg.length > 0 && (!Number.isInteger(turns) || turns <= 0)) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'Usage: /rewind [positive turn count]',
          })
          return
        }
        await rewindCurrentSession(turns)
      },
    })
    registry.register({
      name: '/mode',
      description: 'Open the agent mode picker, or switch with /mode <id>',
      handler: async ({ args }) => {
        const query = args.join(' ').trim()
        if (!query) {
          openModePicker()
          return
        }
        const ensured = await ensureAgentModes()
        if (ensured.error) {
          dispatch({ type: 'SYSTEM_MESSAGE', content: ensured.error })
          return
        }
        const availableAgents = ensured.agents
        if (query === 'current' || query === 'info') {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: [
              `Current mode: ${state.mode}`,
              `Available: ${availableAgents.map((agent) => `${agent.id}${agent.id !== agent.name ? ` (${agent.name})` : ''}`).join(', ') || 'none'}`,
              'Use /mode to open the picker or /mode <id> to switch directly.',
            ].join('\n'),
          })
          return
        }
        const match = findModeMatch(availableAgents, state.mode, query)
        if (match.ambiguousMatches.length > 0) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Multiple modes match "${query}": ${match.ambiguousMatches.map((agent) => agent.id).join(', ')}`,
          })
          return
        }
        if (!match.match) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Unknown mode "${query}". Available modes: ${availableAgents.map((agent) => agent.id).join(', ') || 'none'}`,
          })
          return
        }
        selectAgentMode(match.match)
      },
    })
    registry.register({
      name: '/model',
      description: 'Open the model picker or switch model with /model <name>',
      handler: async ({ args }) => {
        const query = args.join(' ').trim()
        if (!query) {
          openModelPicker()
          return
        }
        if (query === 'default') {
          openModelPicker('default')
          return
        }
        if (query === 'current' || query === 'info') {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: [
              `Current model: ${state.provider}/${state.model}`,
              daemonDefaultProvider
                ? `Daemon default: ${daemonDefaultProvider}/${daemonDefaultModel || '?'}`
                : 'Daemon default: unknown',
              providers.length > 0
                ? `Configured providers: ${providers.map((provider) => `${provider.id} (${provider.models.length})`).join(', ')}`
                : 'Configured providers: none loaded',
              'Use /model to switch the current session, /model default to save daemon defaults, or /model default apply to switch this session to the saved default.',
            ].join('\n'),
          })
          return
        }
        const isDefaultSelection = query.startsWith('default ')
        const modelQuery = isDefaultSelection ? query.slice('default '.length).trim() : query
        if (isDefaultSelection && ['apply', 'sync', 'session', 'use'].includes(modelQuery)) {
          await applyDaemonDefaultToSession('command')
          return
        }
        if (!modelQuery) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content:
              'Usage: /model [current|default] | /model <provider/model|name> | /model default <provider/model|name|apply>',
          })
          return
        }
        let availableProviders = providers
        if (availableProviders.length === 0) {
          try {
            availableProviders = await ensureProvidersLoaded()
          } catch (error) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: error instanceof Error ? error.message : 'Failed to load providers.',
            })
            return
          }
        }
        const match = findModelMatch(
          availableProviders,
          isDefaultSelection ? daemonDefaultProvider || state.provider : state.provider,
          isDefaultSelection ? daemonDefaultModel || state.model : state.model,
          modelQuery,
          {
            defaultProviderId: daemonDefaultProvider,
            defaultModelId: daemonDefaultModel,
          },
        )
        if (match.ambiguousMatches.length > 0) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Multiple models match "${modelQuery}": ${match.ambiguousMatches
              .slice(0, 8)
              .map((item) => `${item.providerId}/${item.modelId}`)
              .join(', ')}`,
          })
          return
        }
        if (!match.match) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content:
              'Usage: /model [current|default] | /model <provider/model|name> | /model default <provider/model|name|apply>',
          })
          return
        }
        if (isDefaultSelection) {
          await setProviderAsDefault({
            providerId: match.match.providerId,
            modelId: match.match.modelId,
            source: 'command',
          })
          return
        }
        applyProviderModelSelection(match.match)
      },
    })
    registry.register({
      name: '/recap',
      description: 'Toggle the end-of-turn recap summary (on|off|status), default on',
      handler: async ({ args }) => {
        const sub = (args[0] ?? 'status').toLowerCase()
        if (sub === 'on' || sub === 'off') {
          const enabled = sub === 'on'
          setRecapEnabled(enabled)
          try {
            await recordRecapPreference(enabled)
          } catch {
            // Preference persistence is best-effort; the in-session toggle
            // above already applied.
          }
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Recap ${enabled ? 'enabled' : 'disabled'}. ${
              enabled
                ? 'A summary (duration, tool calls, files, tokens) is shown after each turn.'
                : 'No end-of-turn summary will be shown.'
            }`,
          })
          return
        }
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Recap is ${recapEnabled ? 'on' : 'off'}. Usage: /recap <on|off>`,
        })
      },
    })
    registry.register({
      name: '/capability',
      description:
        'Show or toggle per-model capability overrides (promptReactPreferred, adaptivePromptReact, deepCoderAnalysis)',
      handler: async ({ args }) => {
        const usage = [
          'Usage: /capability — show flags for the current model',
          `       /capability <${TOGGLEABLE_MODEL_CAPABILITIES.join('|')}> <on|off> [provider/model]`,
          'adaptivePromptReact: start native tool calling, auto-flip to the prompt-react text protocol when native turns break (weak local models).',
          'promptReactPreferred: start with the portable prompt-react tool protocol; use when this endpoint has unreliable native function parsing.',
          'deepCoderAnalysis: always run coder-graph deep analysis scaffolding before implementing (helps weak models).',
        ].join('\n')
        try {
          if (args.length === 0 || args[0] === 'current' || args[0] === 'info') {
            const daemonConfig = await httpClient.config()
            const views = describeModelCapabilities(
              daemonConfig.providers,
              state.provider,
              state.model,
            )
            if (!views) {
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: `Provider "${state.provider}" not found in daemon config.`,
              })
              return
            }
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: [
                `Capability overrides for ${state.provider}/${state.model}:`,
                ...views.map(
                  (view) =>
                    `- ${view.capability}: ${view.enabled ? 'on' : 'off'} (${view.source})`,
                ),
                'Toggle with /capability <name> <on|off> [provider/model]. Changes persist to config.yaml and hot-reload without a daemon restart.',
              ].join('\n'),
            })
            return
          }
          const capability = resolveCapabilityName(args[0] ?? '')
          const toggle = parseCapabilityToggle(args[1] ?? '')
          if (!capability || toggle === null) {
            dispatch({ type: 'SYSTEM_MESSAGE', content: usage })
            return
          }
          let targetProviderId = state.provider
          let targetModelId = state.model
          const modelQuery = args.slice(2).join(' ').trim()
          if (modelQuery) {
            let availableProviders = providers
            if (availableProviders.length === 0) {
              availableProviders = await ensureProvidersLoaded()
            }
            const match = findModelMatch(
              availableProviders,
              state.provider,
              state.model,
              modelQuery,
              {
                defaultProviderId: daemonDefaultProvider,
                defaultModelId: daemonDefaultModel,
              },
            )
            if (match.ambiguousMatches.length > 0) {
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: `Multiple models match "${modelQuery}": ${match.ambiguousMatches
                  .slice(0, 8)
                  .map((item) => `${item.providerId}/${item.modelId}`)
                  .join(', ')}`,
              })
              return
            }
            if (!match.match) {
              dispatch({ type: 'SYSTEM_MESSAGE', content: usage })
              return
            }
            targetProviderId = match.match.providerId
            targetModelId = match.match.modelId
          }
          const daemonConfig = await httpClient.config()
          const result = applyModelCapabilityOverride(
            daemonConfig.providers,
            targetProviderId,
            targetModelId,
            capability,
            toggle,
          )
          if (!result) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: `Provider "${targetProviderId}" not found in daemon config.`,
            })
            return
          }
          if (!result.changed) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: `${capability} is already ${toggle ? 'on' : 'off'} for ${targetProviderId}/${targetModelId}.`,
            })
            return
          }
          await httpClient.updateConfig({ providers: result.providers })
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Set ${capability}=${toggle ? 'on' : 'off'} for ${targetProviderId}/${targetModelId}. Persisted to config.yaml; provider registry hot-reloaded (no daemon restart needed).`,
          })
        } catch (error) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: error instanceof Error ? error.message : 'Failed to update capability.',
          })
        }
      },
    })
    registry.register({
      name: '/swarm',
      description:
        'Orchestrate external CLI agents in tmux (run/list/history/status/agents/logs/kill/attach)',
      handler: async ({ args }) => {
        const sub = (args[0] ?? 'help').toLowerCase()
        const rest = args.slice(1)
        try {
          if (sub === 'help' || sub === '?' || sub === '') {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: [
                '/swarm — orchestrate external CLI agents (claude/codex/gemini/opencode) in tmux',
                '',
                '  /swarm run <goal>            start a new run (warm pool: claude)',
                '  /swarm run! <goal>           start a no-supervisor run (user-driven)',
                '  /swarm list                  list active runs',
                '  /swarm history [n]           show recent runs (active + completed) from jsonl',
                '  /swarm status [runId]        show one run',
                '  /swarm agents [runId]        list agents + handles',
                '  /swarm logs   [runId] [n]    show last n events (default 20)',
                '  /swarm kill   [runId]        cancel a run',
                '  /swarm attach [runId] [hdl]  open interactive tmux mirror',
                '',
                'After /swarm run/run!, follow-up subcommands default to the last',
                'started run — you can omit the runId. Output marks defaulted ids.',
              ].join('\n'),
            })
            return
          }
          if (sub === 'list') {
            const runs = await swarmClient.list()
            if (!runs.length) {
              dispatch({ type: 'SYSTEM_MESSAGE', content: '(no swarm runs)' })
            } else {
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: runs.map((r) => `${r.id}  ${r.status}  ${r.goal}`).join('\n'),
              })
            }
            return
          }
          if (sub === 'history') {
            const limit = Number(rest[0]) || 20
            const runs = await swarmClient.history(limit)
            if (!runs.length) {
              dispatch({ type: 'SYSTEM_MESSAGE', content: '(no completed runs in history)' })
            } else {
              const lines = runs.map((r) => {
                const when = r.endedAt
                  ? new Date(r.endedAt).toISOString().replace('T', ' ').slice(0, 19)
                  : new Date(r.createdAt).toISOString().replace('T', ' ').slice(0, 19)
                const dur = r.endedAt ? `${Math.round((r.endedAt - r.createdAt) / 1000)}s` : 'live'
                const goal = r.goal.length > 50 ? r.goal.slice(0, 47) + '…' : r.goal
                return `${when}  ${r.id}  ${r.status.padEnd(11)}  ${dur.padEnd(5)}  ${goal}`
              })
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: `Last ${runs.length} runs (newest first):\n${lines.join('\n')}`,
              })
            }
            return
          }
          if (sub === 'run' || sub === 'run!') {
            const goal = rest.join(' ').trim()
            if (!goal) {
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: 'Usage: /swarm run <goal>   (or /swarm run! for no-supervisor)',
              })
              return
            }
            const { runId } = await swarmClient.createRun({
              goal,
              cwd: process.cwd(),
              warmPool: ['claude'],
              ...(sub === 'run!' ? ({ noSupervisor: true } as { noSupervisor: true }) : {}),
            })
            setLastSwarmRunId(runId)
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: `swarm started ${runId}\nattach with: /swarm attach ${runId}\nfollow-up subcommands can omit the runId — they default to this one`,
            })
            return
          }
          // For sub-commands beyond list/run, the runId is the FIRST positional
          // arg after the subcommand. Fall back to the last-known run.
          let id = rest[0] ?? null
          let usingDefault = false
          if (!id && lastSwarmRunId) {
            id = lastSwarmRunId
            usingDefault = true
          }
          if (!id) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: `Usage: /swarm ${sub} <runId>   (no recent run remembered)`,
            })
            return
          }
          const usingDefaultHint = usingDefault ? ` (last run ${id})` : ''
          // Remember this id as the latest, even when reading.
          setLastSwarmRunId(id)
          if (sub === 'status') {
            const run = await swarmClient.get(id)
            const blocked = run.agents
              .filter(
                (a) =>
                  a.startupEvidence?.lifecycleState === 'trust_required' ||
                  a.startupEvidence?.lifecycleState === 'tool_permission_required',
              )
              .map((a) => `${a.handle}:${a.startupEvidence?.lifecycleState}`)
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: [
                `id: ${run.id}${usingDefaultHint}`,
                `status: ${run.status}`,
                `goal: ${run.goal}`,
                `worktree: ${run.worktree.path}${run.worktree.createdByDaemon ? ' (daemon-created)' : ''}`,
                `active: ${run.activeHandle ?? '-'}`,
                `agents: ${run.agents.length}`,
                ...(blocked.length ? [`blocked: ${blocked.join(', ')}`] : []),
              ].join('\n'),
            })
            return
          }
          if (sub === 'agents') {
            const run = await swarmClient.get(id)
            const lines = run.agents.map((a) => {
              const marker = run.activeHandle === a.handle ? '●' : ' '
              const startup = a.startupEvidence?.lifecycleState
                ? `${a.runtime ?? a.startupEvidence.runtime}/${a.startupEvidence.lifecycleState}`
                : ''
              return `${marker} ${a.handle}  ${a.agent}  ${a.status}  ${startup}  ${a.role ?? ''}`.trimEnd()
            })
            const header = usingDefaultHint ? `${run.id}${usingDefaultHint}\n` : ''
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: lines.length ? `${header}${lines.join('\n')}` : `${header}(no agents)`,
            })
            return
          }
          if (sub === 'logs') {
            const limit = Number(rest[1]) || 20
            const events = await swarmClient.events(id)
            const tail = events.slice(-limit).map((e) => {
              const t = new Date(e.ts).toISOString().slice(11, 19)
              if (e.type === 'tool.call')
                return `${t}  [tool] ${e.tool} ${JSON.stringify(e.input)}`.slice(0, 200)
              if (e.type === 'tool.result')
                return `${t}  [tool] ${e.tool} → ${e.outputPreview}`.slice(0, 200)
              if (e.type === 'agent.spawned')
                return `${t}  [agent] ${e.agent.agent} (${e.agent.handle}) spawned`
              if (e.type === 'agent.startup') {
                const preview = e.evidence.lastOutputPreview
                  ? ` — ${e.evidence.lastOutputPreview.slice(0, 120)}`
                  : ''
                const label =
                  e.evidence.lifecycleState === 'trust_required' ||
                  e.evidence.lifecycleState === 'tool_permission_required'
                    ? 'blocker'
                    : 'startup'
                return `${t}  [${label}] ${e.handle} ${e.evidence.runtime}/${e.evidence.lifecycleState}${preview}`.slice(
                  0,
                  200,
                )
              }
              if (e.type === 'agent.active') return `${t}  [active] → ${e.handle}`
              if (e.type === 'run.started') return `${t}  [run] started — "${e.goal}"`
              if (e.type === 'run.ended') return `${t}  [run] ${e.status}`
              return `${t}  [${e.type}]`
            })
            const header = usingDefaultHint ? `${id}${usingDefaultHint}\n` : ''
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: tail.length ? `${header}${tail.join('\n')}` : `${header}(no events)`,
            })
            return
          }
          if (sub === 'kill') {
            await swarmClient.cancel(id)
            if (lastSwarmRunId === id) setLastSwarmRunId(null)
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: `cancelled ${id}${usingDefaultHint}`,
            })
            return
          }
          if (sub === 'attach') {
            const run = await swarmClient.get(id)
            const handleArg = rest[1]
            const target = handleArg
              ? run.agents.find((a) => a.handle === handleArg || a.role === handleArg)
              : (run.agents.find((a) => a.handle === run.activeHandle) ?? run.agents[0])
            if (!target) {
              dispatch({ type: 'SYSTEM_MESSAGE', content: '(no agents to attach to)' })
              return
            }
            setSwarmAttachTarget({
              runId: run.id,
              handle: target.handle,
              agent: target.agent,
              role: target.role,
              tmuxSessionName: target.tmuxSessionName,
            })
            return
          }
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Unknown /swarm subcommand: ${sub}\nTry /swarm help`,
          })
        } catch (err) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `swarm error: ${err instanceof Error ? err.message : String(err)}`,
          })
        }
      },
    })
    registry.register({
      name: '/approvals',
      description: 'List pending + remembered approvals, or clear saved rules',
      handler: async ({ args }) => {
        const action = args[0]?.toLowerCase()
        if (action === 'clear') {
          const scopeArg = args[1]?.toLowerCase()
          const scope = scopeArg === 'session' || scopeArg === 'always' ? scopeArg : undefined
          try {
            await httpClient.clearRememberedApprovals({ scope })
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: scope
                ? `Cleared remembered ${scope}-scope approvals.`
                : 'Cleared all remembered approvals (session and persistent).',
            })
          } catch (error) {
            dispatch({ type: 'ERROR', message: normalizeError(error) })
          }
          return
        }
        const lines: string[] = []
        if (state.pendingApproval) {
          lines.push('Pending:')
          lines.push(
            `  ${state.pendingApproval.toolName} (${state.pendingApproval.state}) request ${state.pendingApproval.requestId.slice(0, 8)}`,
          )
          lines.push('')
        }
        try {
          const { decisions } = await httpClient.listRememberedApprovals()
          if (decisions.length === 0) {
            lines.push('No remembered approvals. (Use s/a on an approval prompt to save one.)')
          } else {
            lines.push(`Remembered approvals (${decisions.length}):`)
            for (const entry of decisions) {
              const verdict = entry.approved ? 'allow' : 'deny'
              const scopeLabel =
                entry.scope === 'always'
                  ? 'persistent'
                  : `session:${entry.sessionId?.slice(0, 8) ?? '?'}`
              lines.push(`  ${verdict.padEnd(5)} ${entry.tool}: ${entry.pattern}  [${scopeLabel}]`)
            }
            lines.push('')
            lines.push('Use /approvals clear [session|always] to remove them.')
          }
        } catch (error) {
          lines.push(`Failed to load remembered approvals: ${normalizeError(error)}`)
        }
        if (!state.pendingApproval && lines.length === 0) {
          lines.push('No pending approvals.')
        }
        dispatch({ type: 'SYSTEM_MESSAGE', content: lines.join('\n') })
      },
    })
    registry.register({
      name: '/questions',
      description: 'List pending agent questions',
      handler: () => {
        const lines: string[] = []
        if (state.pendingQuestions.length === 0) {
          lines.push('No pending questions.')
        } else {
          lines.push(`Pending questions (${state.pendingQuestions.length}):`)
          for (const question of state.pendingQuestions) {
            const choices = question.choices?.length
              ? ` choices: ${question.choices.join(' · ')}`
              : ''
            lines.push(`  ${question.id}: ${question.prompt}${choices}`)
            lines.push(`    answer with /answer ${question.id} <reply>`)
          }
        }
        dispatch({ type: 'SYSTEM_MESSAGE', content: lines.join('\n') })
      },
    })
    registry.register({
      name: '/answer',
      description: 'Answer a pending agent question',
      handler: async ({ args }) => {
        const questionId = args[0]
        const answer = args.slice(1).join(' ').trim()
        if (!questionId || !answer) {
          dispatch({ type: 'SYSTEM_MESSAGE', content: 'Usage: /answer <question-id> <reply>' })
          return
        }
        await answerQuestion(questionId, answer)
      },
    })
    registry.register({
      name: '/project',
      description: 'Inspect / auto-detect / clear / select a project',
      handler: async ({ args }) => {
        const query = args.join(' ').trim()
        if (!query || query === 'current' || query === 'info') {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: [
              `Project: ${state.projectName ?? 'none'}`,
              `Workspace: ${workspaceProject?.rootDir ?? 'not detected'}`,
              `Selection: ${projectSelectionSource ?? 'none'}`,
              projects.length > 0
                ? `Available: ${projects
                    .slice(0, 6)
                    .map((project) => project.name)
                    .join(', ')}`
                : 'Available: none',
              'Use /project auto, /project none, or /project <name>.',
            ].join('\n'),
          })
          return
        }
        if (query === 'auto') {
          if (!workspaceProject) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: 'No workspace project could be detected from the current directory.',
            })
            return
          }
          try {
            const latestProjects = sortProjects(await httpClient.projects())
            setProjects(latestProjects)
            const matchedProject = matchProjectForWorkspace(
              latestProjects,
              workspaceProject,
              (await loadCliState()).projectBindings,
              state.sessionId,
            )
            if (matchedProject) {
              applyProjectSelection(
                matchedProject,
                matchedProject.sessionIds.includes(state.sessionId ?? '') ? 'session' : 'workspace',
              )
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: `Project set to ${matchedProject.name}.`,
              })
              return
            }
            const createdProject = await httpClient.createProject({
              name: workspaceProject.name,
            })
            setProjects((current) => upsertProject(current, createdProject))
            applyProjectSelection(createdProject, 'workspace')
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: `Created and selected project ${createdProject.name}.`,
            })
          } catch (error) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: error instanceof Error ? error.message : 'Failed to resolve project.',
            })
          }
          return
        }
        if (['none', 'off', 'clear'].includes(query)) {
          applyProjectSelection(null, 'manual')
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'Project selection cleared.',
          })
          return
        }
        const projectLookup = findProjectByName(projects, query)
        if (projectLookup.ambiguousMatches.length > 0) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Multiple projects match "${query}": ${projectLookup.ambiguousMatches.map((project) => project.name).join(', ')}`,
          })
          return
        }
        try {
          const project = projectLookup.match ?? (await httpClient.createProject({ name: query }))
          setProjects((current) => upsertProject(current, project))
          applyProjectSelection(project, 'manual')
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: projectLookup.match
              ? `Project set to ${project.name}.`
              : `Created and selected project ${project.name}.`,
          })
        } catch (error) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: error instanceof Error ? error.message : 'Failed to select project.',
          })
        }
      },
    })
    registry.register({
      name: '/session',
      description: 'Open the session picker, inspect, branch/compact/export/delete, or load by id',
      handler: async ({ args }) => {
        if (!args[0]) {
          openSessionPicker()
          return
        }
        if (args[0] === 'current' || args[0] === 'info') {
          if (!state.sessionId) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content:
                'Context Map\nSession  none\nNext     /resume to open recent history, or type a prompt to start fresh',
            })
            return
          }
          try {
            const session = await httpClient.session(state.sessionId)
            const delegationLines = session.delegation
              ? [
                  `Delegation  ${delegationHealthLabel(session.delegation)} ${symbols.separator} worker ${session.delegation.targetDevice}`,
                  `Lease      ${delegationHealthDetail(session.delegation)}`,
                ]
              : []
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: [
                buildContextManagementSummary({
                  session,
                  usage: state.usage,
                  contextInputTokens: currentContextTokens,
                  contextWindow: effectiveContextWindow,
                  contextEstimated,
                  projectName: state.projectName,
                  mode: String(state.mode),
                  autonomy: state.autonomy,
                  thinkingLevel: state.thinkingLevel,
                  maxTokens: state.maxTokens,
                  artifactCount: state.artifacts.length,
                  pendingApprovalTool: state.pendingApproval?.toolName ?? null,
                  focus: 'map',
                }),
                ...delegationLines,
                'Session actions  /session branch  ·  /session compact  ·  /session export [markdown|json] [path]  ·  /session delete confirm',
              ].join('\n'),
            })
          } catch (error) {
            dispatch({
              type: 'ERROR',
              message:
                error instanceof Error
                  ? `Failed to load session info: ${error.message}`
                  : 'Failed to load session info.',
            })
          }
          return
        }
        if (args[0] === 'branch') {
          await branchCurrentSession()
          return
        }
        if (args[0] === 'compact') {
          if (!state.sessionId) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: 'No active session to compact.',
            })
            return
          }
          await compactSessionById(state.sessionId)
          return
        }
        if (args[0] === 'export') {
          await exportCurrentSession(args.slice(1))
          return
        }
        if (args[0] === 'delete') {
          if (args[1] !== 'confirm') {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: 'Use /session delete confirm to remove the active session.',
            })
            return
          }
          if (!state.sessionId) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: 'No active session to delete.',
            })
            return
          }
          await deleteSessionById(state.sessionId)
          return
        }
        await loadSessionById(args[0])
      },
    })
    registry.register({
      name: '/provider',
      description: 'Open the model picker, switch provider, or run setup/edit/delete',
      handler: async ({ args }) => {
        const query = args.join(' ').trim()
        if (!query) {
          openModelPicker()
          return
        }
        if (query === 'default') {
          openModelPicker('default')
          return
        }
        if (query === 'setup' || query === 'add' || query === 'config') {
          await openProviderSetup()
          return
        }
        if (query.startsWith('setup ')) {
          const presetQuery = query.slice('setup '.length).trim()
          if (!presetQuery) {
            await openProviderSetup()
            return
          }
          await openProviderSetup({ presetQuery })
          return
        }
        if (query.startsWith('edit ')) {
          const providerId = query.slice('edit '.length).trim()
          if (!providerId) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: 'Usage: /provider edit <id>',
            })
            return
          }
          await openProviderSetup({ providerId })
          return
        }
        if (query.startsWith('default ')) {
          const providerId = query.slice('default '.length).trim()
          if (!providerId) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: 'Usage: /provider default <id>',
            })
            return
          }
          await setProviderAsDefault({
            providerId,
            source: 'command',
          })
          return
        }
        if (query.startsWith('delete ')) {
          const providerId = query.slice('delete '.length).trim()
          if (!providerId) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: 'Usage: /provider delete <id>',
            })
            return
          }
          await openProviderDeleteConfirmation({ providerId })
          return
        }
        if (query === 'current' || query === 'info') {
          const currentProviderInfo =
            providers.find((provider) => provider.id === state.provider) ?? null
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: [
              `Current provider: ${state.provider}`,
              `Current model: ${state.model}`,
              currentProviderInfo
                ? `Provider health: ${formatProviderHealthSummary(currentProviderInfo)}`
                : 'Provider health: unknown',
              daemonDefaultProvider
                ? `Daemon default: ${daemonDefaultProvider}/${daemonDefaultModel || '?'}`
                : 'Daemon default: unknown',
              providers.length > 0
                ? `Configured providers: ${providers.map((provider) => provider.id).join(', ')}`
                : 'Configured providers: none loaded',
              'Use /provider or Ctrl+T to open the picker, /provider <id> to switch the current session, /provider default <id> to change daemon defaults only, or /provider setup to configure one.',
            ].join('\n'),
          })
          return
        }
        let availableProviders = providers
        if (availableProviders.length === 0) {
          try {
            availableProviders = await ensureProvidersLoaded()
          } catch (error) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: error instanceof Error ? error.message : 'Failed to load providers.',
            })
            return
          }
        }
        const match = findProviderMatch(availableProviders, query)
        if (match.ambiguousMatches.length > 0) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Multiple providers match "${query}": ${match.ambiguousMatches.map((provider) => provider.id).join(', ')}`,
          })
          return
        }
        if (!match.match) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content:
              'Usage: /provider [current|setup|setup <type>|edit <id>|default <id>|delete <id>] | /provider <id>',
          })
          return
        }
        const nextModel =
          match.match.models.find((model) => model.id === state.model) ?? match.match.models[0]
        if (!nextModel) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content:
              match.match.health.status === 'ready'
                ? `Provider ${match.match.id} has no configured chat models.`
                : `Provider ${match.match.id} is not usable yet: ${formatProviderHealthSummary(match.match)}.\nUse /provider edit ${match.match.id} to repair it.`,
          })
          return
        }
        applyProviderModelSelection({
          providerId: match.match.id,
          modelId: nextModel.id,
        })
      },
    })
    registry.register(
      createStateBoardSlashCommand({
        getSessionId: () => state.sessionId ?? undefined,
        fetchBoard: (sessionId) => httpClient.stateBoard(sessionId),
        render: (content) => dispatch({ type: 'SYSTEM_MESSAGE', content }),
      }),
    )
    registry.register({
      name: '/usage',
      description: 'Open/close the usage dashboard or set the day window',
      handler: async ({ args }) => {
        const query = args.join(' ').trim().toLowerCase()
        if (!query || ['open', 'show', 'refresh'].includes(query)) {
          await loadUsageDashboard(usageDashboardDays)
          return
        }
        if (['close', 'hide', 'off'].includes(query)) {
          closeUsageDashboard()
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'Usage dashboard closed.',
          })
          return
        }
        if (query === 'current' || query === 'info') {
          const latestBucket = usageBuckets[0]
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: [
              `Dashboard: ${usageDashboardOpen ? 'open' : 'closed'}`,
              `Window: last ${usageDashboardDays} day${usageDashboardDays === 1 ? '' : 's'}`,
              usageSummary
                ? `Totals: in ${usageSummary.inputTokens.toLocaleString()} / out ${usageSummary.outputTokens.toLocaleString()} / cost $${usageSummary.costUsd.toFixed(4)} / req ${usageSummary.requestCount.toLocaleString()}`
                : 'Totals: not loaded',
              latestBucket
                ? `Top day: ${latestBucket.date} $${latestBucket.costUsd.toFixed(4)} ${latestBucket.topProvider}/${latestBucket.topModel}`
                : 'Top day: none',
              usageDashboardLoading
                ? 'Status: loading...'
                : usageDashboardError
                  ? `Status: ${usageDashboardError}`
                  : usageSummary || usageDaily.length > 0
                    ? 'Status: ready'
                    : 'Status: not loaded',
              'Use /usage, /usage <days>, or /usage close.',
            ].join('\n'),
          })
          return
        }
        const dayMatch = query.match(/^(\d+)(d)?$/)
        if (!dayMatch) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Usage: /usage [current|close] or /usage <1-${USAGE_DASHBOARD_MAX_DAYS}>`,
          })
          return
        }
        const days = Number.parseInt(dayMatch[1], 10)
        if (!Number.isInteger(days) || days < 1 || days > USAGE_DASHBOARD_MAX_DAYS) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Usage window must be between 1 and ${USAGE_DASHBOARD_MAX_DAYS} days.`,
          })
          return
        }
        await loadUsageDashboard(days)
      },
    })
    registry.register({
      name: '/tps',
      description: 'Show token-per-second speed for the last response and current session',
      handler: ({ args }) => {
        const query = args.join(' ').trim().toLowerCase()
        if (query === 'reset') {
          resetTokenSpeedStats()
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'TPS stats reset.',
          })
          return
        }
        if (query && query !== 'current' && query !== 'info') {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'Usage: /tps [current|reset]',
          })
          return
        }
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: formatTokenSpeedStats(getTokenSpeedStats()),
        })
      },
    })
    registry.register({
      name: '/memory',
      description: 'Search memory or close the panel',
      handler: async ({ args }) => {
        const query = args.join(' ').trim()
        if (!query) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Usage: /memory <query> | /memory current | /memory close\n${TUI_FILE_MEMORY_USAGE}`,
          })
          return
        }
        if (['lifecycle', 'health', 'maintenance-status'].includes(args[0]?.toLowerCase() ?? '')) {
          try {
            const snapshot = await httpClient.memoryLifecycle()
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: formatTuiMemoryLifecycleSummary(snapshot),
            })
          } catch (error) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: error instanceof Error ? error.message : 'Failed to load memory lifecycle.',
            })
          }
          return
        }
        if (['audit', 'audits'].includes(args[0]?.toLowerCase() ?? '')) {
          const memoryId = args[1]?.startsWith('--') ? undefined : args[1]
          try {
            const entries = await httpClient.memoryAudit({
              memoryId: memoryId?.trim() || undefined,
              limit: 8,
            })
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: formatTuiMemoryAuditSummary(entries),
            })
          } catch (error) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content:
                error instanceof Error ? error.message : 'Failed to load memory audit trail.',
            })
          }
          return
        }
        if (['maintenance', 'maintain', 'cleanup'].includes(args[0]?.toLowerCase() ?? '')) {
          const apply = args.slice(1).includes('--apply')
          try {
            const result = await httpClient.runMemoryMaintenance({
              dryRun: !apply,
              reason: apply
                ? 'Run memory lifecycle maintenance from CLI TUI'
                : 'Preview memory lifecycle maintenance from CLI TUI',
            })
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: formatTuiMemoryMaintenanceSummary(result),
            })
          } catch (error) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: error instanceof Error ? error.message : 'Failed to run memory maintenance.',
            })
          }
          return
        }
        if (
          ['backlog', 'backlogs', 'open-loop', 'open-loops', 'openloops'].includes(
            args[0]?.toLowerCase() ?? '',
          )
        ) {
          const action = args[1]?.toLowerCase() ?? 'list'
          const rest = args.slice(2)
          try {
            if (['help', '--help', '-h'].includes(action)) {
              dispatch({ type: 'SYSTEM_MESSAGE', content: TUI_FILE_MEMORY_USAGE })
              return
            }

            if (['list', 'current', 'info'].includes(action)) {
              const snapshot = await httpClient.fileMemory()
              dispatch({ type: 'SYSTEM_MESSAGE', content: formatTuiBacklogSummary(snapshot) })
              return
            }

            if (action === 'add') {
              const content = rest.join(' ').trim()
              if (!content) {
                dispatch({ type: 'SYSTEM_MESSAGE', content: TUI_FILE_MEMORY_USAGE })
                return
              }
              const snapshot = await httpClient.fileMemory()
              const next = appendUniqueMemoryLine(
                findMemorySection(snapshot, TUI_OPEN_LOOP_QUEUE_SECTION),
                formatManualMemoryBacklogLine(content),
              )
              await httpClient.updateFileMemorySection(TUI_OPEN_LOOP_QUEUE_SECTION, next)
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: `Added open-loop backlog item: ${content}`,
              })
              return
            }

            if (['done', 'resolve', 'remove', 'rm'].includes(action)) {
              const queryText = rest.join(' ').trim()
              if (!queryText) {
                dispatch({ type: 'SYSTEM_MESSAGE', content: TUI_FILE_MEMORY_USAGE })
                return
              }
              const snapshot = await httpClient.fileMemory()
              const result = removeMatchingMemoryLines(
                findMemorySection(snapshot, TUI_OPEN_LOOP_QUEUE_SECTION),
                queryText,
              )
              if (result.removed.length === 0) {
                dispatch({
                  type: 'SYSTEM_MESSAGE',
                  content: `No open-loop backlog item matched: ${queryText}`,
                })
                return
              }
              await httpClient.updateFileMemorySection(
                TUI_OPEN_LOOP_QUEUE_SECTION,
                result.remaining,
              )
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: [
                  `Resolved ${result.removed.length} open-loop backlog item${result.removed.length === 1 ? '' : 's'}.`,
                  ...result.removed.map((item) => `  - ${item.replace(/^-\s*/, '')}`),
                ].join('\n'),
              })
              return
            }

            dispatch({ type: 'SYSTEM_MESSAGE', content: TUI_FILE_MEMORY_USAGE })
          } catch (error) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: error instanceof Error ? error.message : 'Failed to manage memory backlog.',
            })
          }
          return
        }
        if (['file', 'files', 'markdown'].includes(args[0]?.toLowerCase() ?? '')) {
          const action = args[1]?.toLowerCase() ?? 'list'
          const rest = args.slice(2)
          try {
            if (['help', '--help', '-h'].includes(action)) {
              dispatch({ type: 'SYSTEM_MESSAGE', content: TUI_FILE_MEMORY_USAGE })
              return
            }

            if (['list', 'current', 'info'].includes(action)) {
              const snapshot = await httpClient.fileMemory()
              dispatch({ type: 'SYSTEM_MESSAGE', content: formatTuiFileMemorySummary(snapshot) })
              return
            }

            if (action === 'today' || action === 'yesterday') {
              const snapshot = await httpClient.fileMemory()
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content:
                  action === 'today'
                    ? formatTuiDailyMemory('today', snapshot.todayNotePath, snapshot.todayNote)
                    : formatTuiDailyMemory(
                        'yesterday',
                        snapshot.yesterdayNotePath,
                        snapshot.yesterdayNote,
                      ),
              })
              return
            }

            if (action === 'show') {
              const sectionTitle = rest.join(' ').trim()
              if (!sectionTitle) {
                const snapshot = await httpClient.fileMemory()
                dispatch({ type: 'SYSTEM_MESSAGE', content: formatTuiFileMemorySummary(snapshot) })
                return
              }
              const snapshot = await httpClient.fileMemory()
              const section = snapshot.sections.find((entry) => entry.title === sectionTitle)
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: section
                  ? `## ${section.title}\n${section.content || '(empty)'}`
                  : `Memory section not found: ${sectionTitle}`,
              })
              return
            }

            if (action === 'set') {
              const parsed = splitSectionAndContent(rest)
              if (!parsed?.section || !parsed.content) {
                dispatch({ type: 'SYSTEM_MESSAGE', content: TUI_FILE_MEMORY_USAGE })
                return
              }
              const result = await httpClient.updateFileMemorySection(
                parsed.section,
                parsed.content,
              )
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: result.deleted
                  ? `Cleared markdown memory section: ${parsed.section}`
                  : `Saved markdown memory section: ${result.title}`,
              })
              return
            }

            if (['delete', 'remove', 'rm'].includes(action)) {
              const sectionTitle = rest.join(' ').trim()
              if (!sectionTitle) {
                dispatch({ type: 'SYSTEM_MESSAGE', content: TUI_FILE_MEMORY_USAGE })
                return
              }
              const result = await httpClient.deleteFileMemorySection(sectionTitle)
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: result.deleted
                  ? `Deleted markdown memory section: ${sectionTitle}`
                  : `Markdown memory section was already absent: ${sectionTitle}`,
              })
              return
            }

            dispatch({ type: 'SYSTEM_MESSAGE', content: TUI_FILE_MEMORY_USAGE })
          } catch (error) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: error instanceof Error ? error.message : 'Failed to manage markdown memory.',
            })
          }
          return
        }
        if (['close', 'hide', 'off'].includes(query.toLowerCase())) {
          closeMemorySearchPanel()
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'Memory search panel closed.',
          })
          return
        }
        if (['current', 'info'].includes(query.toLowerCase())) {
          const topResult = memorySearchResults[0]
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: [
              `Memory panel: ${memorySearchOpen ? 'open' : 'closed'}`,
              `Query: ${memorySearchQuery || 'none'}`,
              `Results: ${memorySearchResults.length}`,
              memorySearchStatus
                ? `Status: ${memorySearchStatus.status}${memorySearchStatus.configuredProviderId && memorySearchStatus.configuredModel ? ` (${memorySearchStatus.configuredProviderId}/${memorySearchStatus.configuredModel})` : ''}`
                : 'Status: unavailable',
              topResult
                ? `Top hit: ${topResult.source} ${topResult.score != null ? `(${Math.round(topResult.score * 100)}%)` : ''} ${topResult.content.replace(/\s+/g, ' ').trim().slice(0, 80)}`
                : 'Top hit: none',
              memorySearchLoading
                ? 'Search: loading...'
                : memorySearchError
                  ? `Search: ${memorySearchError}`
                  : memorySearchQuery
                    ? 'Search: ready'
                    : 'Search: not loaded',
            ].join('\n'),
          })
          return
        }
        await loadMemorySearch(query)
      },
    })
    registry.register({
      name: '/rag',
      description: 'Open, search, add, or sync local RAG sources',
      handler: async ({ args }) => {
        const action = args[0]?.toLowerCase() ?? 'open'
        try {
          if (['help', '--help', '-h'].includes(action)) {
            dispatch({ type: 'SYSTEM_MESSAGE', content: RAG_COMMAND_USAGE })
            return
          }

          if (['close', 'hide', 'off'].includes(action)) {
            closeRagPanel()
            dispatch({ type: 'SYSTEM_MESSAGE', content: 'RAG panel closed.' })
            return
          }

          if (['open', 'current', 'info', 'list', 'sources'].includes(action)) {
            await loadRagPanel('')
            const sources = await httpClient.ragFolders()
            const vector = await httpClient.ragVectorDbInfo().catch(() => null)
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: [
                `Local RAG sources: ${sources.length}`,
                formatRagVectorInfo(vector),
                sources.length > 0
                  ? sources.slice(0, 5).map(formatRagSource).join('\n')
                  : 'No local RAG sources. Use /rag add <path>.',
              ].join('\n'),
            })
            return
          }

          if (action === 'sync') {
            await syncRagPanel()
            return
          }

          if (action === 'add') {
            const sourcePath = args[1]
            if (!sourcePath) {
              dispatch({ type: 'SYSTEM_MESSAGE', content: RAG_COMMAND_USAGE })
              return
            }
            await addRagSourceFromSlash(sourcePath, args.slice(2).join(' '))
            return
          }

          const query = action === 'search' ? args.slice(1).join(' ').trim() : args.join(' ').trim()
          if (!query) {
            dispatch({ type: 'SYSTEM_MESSAGE', content: RAG_COMMAND_USAGE })
            return
          }
          await loadRagPanel(query)
        } catch (error) {
          dispatch({ type: 'ERROR', message: normalizeError(error) })
        }
      },
    })
    registry.register({
      name: '/context',
      description: 'Print a context-management summary for the current session',
      handler: async ({ args }) => {
        if (!state.sessionId) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content:
              'Context Map\nSession  none\nNext     /resume to open recent history, or type a prompt to start fresh',
          })
          return
        }
        const contextQuery = args[0]?.toLowerCase() ?? 'map'
        const focus =
          contextQuery === 'resume' ? 'resume' : contextQuery === 'rewind' ? 'rewind' : 'map'
        if (!['map', 'current', 'info', 'resume', 'rewind', 'compact'].includes(contextQuery)) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'Usage: /context [map|resume|rewind]',
          })
          return
        }
        try {
          const ctxSession = await httpClient.session(state.sessionId)
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: buildContextManagementSummary({
              session: ctxSession,
              usage: state.usage,
              contextInputTokens: currentContextTokens,
              contextWindow: effectiveContextWindow,
              contextEstimated,
              projectName: state.projectName,
              mode: String(state.mode),
              autonomy: state.autonomy,
              thinkingLevel: state.thinkingLevel,
              maxTokens: state.maxTokens,
              artifactCount: state.artifacts.length,
              pendingApprovalTool: state.pendingApproval?.toolName ?? null,
              focus,
            }),
          })
        } catch (error) {
          dispatch({
            type: 'ERROR',
            message:
              error instanceof Error
                ? `Failed to load context info: ${error.message}`
                : 'Failed to load context info.',
          })
        }
      },
    })
    registry.register({
      name: '/autonomy',
      description: 'Switch autonomy level (readonly|supervised|autonomous)',
      handler: async ({ args }) => {
        if (!args[0]) {
          openAutonomyPicker()
          return
        }
        if (args[0] === 'current' || args[0] === 'info') {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: [
              `Current autonomy: ${getAutonomyOption(state.autonomy).label} (${state.autonomy})`,
              'Available: readonly, supervised, autonomous',
              'Use /autonomy to open the picker, /autonomy <level> to switch directly, or Ctrl+A to cycle.',
            ].join('\n'),
          })
          return
        }
        const option = findAutonomyOption(args.join(' '))
        if (!option) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: 'Usage: /autonomy <readonly|supervised|autonomous>',
          })
          return
        }
        await applyAutonomySetting(option.id)
      },
    })
    registry.register({
      name: '/providers',
      description: 'List configured providers and their health',
      handler: async () => {
        try {
          const providers = await httpClient.providers()
          const lines = providers.flatMap((provider) => [
            `${provider.name} (${provider.id})${provider.id === state.provider ? ' [current]' : ''}${provider.id === daemonDefaultProvider ? ' [default]' : ''}${provider.supportsEmbedding ? ' [embeddings]' : ''} [${provider.health.status}]`,
            `  health: ${formatProviderHealthSummary(provider)}`,
            `  configured models: ${provider.configuredModelIds.join(', ') || 'none'}`,
            `  embeddings: ${provider.embeddingModelIds.join(', ') || 'unsupported'}`,
            ...provider.models.map((model) => `  - ${model.id}${formatProviderModelBadges(model)}`),
          ])
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: lines.join('\n') || 'No providers configured.',
          })
        } catch (error) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: error instanceof Error ? error.message : 'Failed to load providers.',
          })
        }
      },
    })
    registry.register({
      name: '/containers',
      description: 'List sepilot-managed Docker containers from this and previous sessions',
      handler: async () => {
        try {
          if (!(await dockerAvailable())) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: 'Docker is not installed or its daemon is not running.',
            })
            return
          }
          const list = await listManagedContainers()
          if (list.length === 0) {
            dispatch({ type: 'SYSTEM_MESSAGE', content: 'No sepilot-managed containers.' })
            return
          }
          const lines = [
            `Sepilot-managed containers (${list.length}):`,
            ...list.map(
              (c) => `- ${c.name}  [${c.state}]  ${c.image}${c.purpose ? `  — ${c.purpose}` : ''}`,
            ),
            '',
            'Remove with `sepilot containers prune` (stopped), `sepilot containers prune --all`, or `sepilot containers rm <name>`.',
          ]
          dispatch({ type: 'SYSTEM_MESSAGE', content: lines.join('\n') })
        } catch (error) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: error instanceof Error ? error.message : 'Failed to list containers.',
          })
        }
      },
    })
    for (const command of DAEMON_ADMIN_SLASH_COMMANDS) {
      if (registry.has(command.name)) continue
      registry.register({
        name: command.name,
        description: command.description,
        handler: async ({ args }) => {
          try {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: await runDaemonAdminSlashCommand(httpClient, command.name, args),
            })
          } catch (error) {
            dispatch({ type: 'ERROR', message: normalizeError(error) })
          }
        },
      })
    }
    return registry
  }, [
    applyAutonomySetting,
    applyDaemonDefaultToSession,
    applyProjectSelection,
    applyProviderModelSelection,
    applyThemeSelection,
    addRagSourceFromSlash,
    answerQuestion,
    branchCurrentSession,
    cancelQueuedFollowUps,
    closeMemorySearchPanel,
    closeRagPanel,
    closeUsageDashboard,
    lastSwarmRunId,
    setLastSwarmRunId,
    swarmClient,
    compactSessionById,
    deleteSessionById,
    exportCurrentSession,
    openSessionPicker,
    projectSelectionSource,
    projects,
    workspaceProject,
    setProjects,
    contextEstimated,
    currentContextTokens,
    effectiveContextWindow,
    daemonDefaultModel,
    daemonDefaultProvider,
    dispatch,
    ensureAgentModes,
    ensureProvidersLoaded,
    getTokenSpeedStats,
    httpClient,
    loadMemorySearch,
    loadRagPanel,
    loadSessionById,
    loadUsageDashboard,
    memorySearchError,
    memorySearchLoading,
    memorySearchOpen,
    memorySearchQuery,
    memorySearchResults,
    memorySearchStatus,
    openAutonomyPicker,
    openFilePicker,
    openSkillManager,
    openSkillStore,
    openModelPicker,
    openModePicker,
    openProviderDeleteConfirmation,
    openProviderSetup,
    pendingAttachmentPaths,
    providers,
    recapEnabled,
    rememberQueuedFollowUp,
    refreshInstalledSkills,
    resumeSession,
    resetTokenSpeedStats,
    rewindCurrentSession,
    selectAgentMode,
    setHelpOpen,
    setInputValue,
    sendMessage,
    setProviderAsDefault,
    setQueuedAttachmentPaths,
    setShowMcp,
    setTranscriptClearedAt,
    showDoctorSummary,
    startNewSession,
    syncRagPanel,
    state.artifacts,
    state.autonomy,
    state.maxTokens,
    state.mode,
    state.model,
    state.pendingApproval,
    state.pendingQuestions,
    state.projectName,
    state.provider,
    state.sessionId,
    state.thinkingLevel,
    state.usage,
    themeId,
    usageBuckets,
    usageDaily.length,
    usageDashboardDays,
    usageDashboardError,
    usageDashboardLoading,
    usageDashboardOpen,
    usageSummary,
  ])

  // Slash commands live in slashCommandRegistry. This dispatcher
  // is just a registry lookup with a fallback for unknown commands. Handlers
  // capture their own state via the registry's useMemo deps, so this hook
  // does not need a giant deps array of its own.
  const handleSlashCommand = useCallback(
    async (cmd: string) => {
      const [command, ...args] = splitSlashCommandInput(cmd)
      const registered = command ? slashCommandRegistry.get(command) : undefined
      if (registered) {
        await registered.handler({ args, rawCommand: cmd })
      } else {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Unknown command: ${command}. Type /help.`,
        })
      }
      setInputValue('')
    },
    [dispatch, setInputValue, slashCommandRegistry],
  )

  const requestExit = useCallback(() => {
    clearExitHint()
    wsClient?.close()
    exit()
  }, [clearExitHint, exit, wsClient])

  const submitInput = useCallback(
    async (text: string, options: { displayText?: string } = {}) => {
      const trimmedText = text.trim()
      if (isExitInput(trimmedText)) {
        requestExit()
        return
      }
      if (text.startsWith('/')) {
        await handleSlashCommand(text)
        return
      }
      if (['continue', 'resume'].includes(text.trim().toLowerCase())) {
        await handleSlashCommand('/resume')
        return
      }
      const rerunLastShellCommand = trimmedText === '!!'
      if (rerunLastShellCommand || text.startsWith('!')) {
        const command = rerunLastShellCommand
          ? (lastLocalShellCommandRef.current ?? '').trim()
          : text.slice(1).trim()

        if (!command) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: rerunLastShellCommand
              ? 'No local shell command to rerun yet. Use !<command> first.'
              : 'Direct shell usage: !<command>\nExample: !git status',
          })
          return
        }

        const gate = resolveLocalShellGate({ autonomy: state.autonomy })
        if (gate === 'block') {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content:
              "Local shell is disabled under autonomy 'readonly'. Change it with /autonomy or the autonomy picker.",
          })
          return
        }

        const userMessageId = crypto.randomUUID()
        const toolCallId = crypto.randomUUID()
        const startedAt = Date.now()
        const userTimestamp = Date.now()
        const invocation = buildLocalShellInvocation(command)
        const localShellCwd =
          selectedProject?.workingDirectory ?? workspaceProject?.rootDir ?? FILE_PICKER_ROOT_DIR
        lastLocalShellCommandRef.current = command

        dispatch({
          type: 'APPEND_MESSAGE',
          message: {
            id: userMessageId,
            role: 'user',
            content: options.displayText ?? text,
            timestamp: userTimestamp,
          },
        })

        dispatch({
          type: 'UPSERT_MESSAGE',
          message: {
            id: toolCallId,
            role: 'tool',
            content: '',
            timestamp: userTimestamp + 1,
            toolCall: {
              ...buildStoredToolCallState({
                id: toolCallId,
                name: 'terminal.run',
                input: {
                  executable: invocation.shell,
                  args: invocation.args,
                  cwd: localShellCwd,
                },
                status: 'running',
                meta: 'Running local shell command',
              }),
              collapsed: true,
            },
          },
        })

        setLocalShellCommand({
          command,
          startedAt,
        })

        const shellAborter = new AbortController()
        localShellAborterRef.current = shellAborter
        try {
          const result = await runLocalShellCommand(command, {
            cwd: localShellCwd,
            signal: shellAborter.signal,
          })
          const toolOutput = result.stderr
            ? [result.stdout, '[stderr]', result.stderr]
                .filter((part) => part.trim().length > 0)
                .join('\n')
            : result.stdout
          const status = result.exitCode === 0 ? 'success' : 'error'

          dispatch({
            type: 'UPSERT_MESSAGE',
            message: {
              id: toolCallId,
              role: 'tool',
              content: '',
              timestamp: userTimestamp + 1,
              toolCall: {
                ...buildStoredToolCallState({
                  id: toolCallId,
                  name: 'terminal.run',
                  input: {
                    executable: result.shell,
                    args: result.args,
                    cwd: result.cwd,
                  },
                  status,
                  output: toolOutput,
                  meta: `Local shell • exit ${result.exitCode} • ${result.durationMs}ms`,
                }),
                collapsed: true,
              },
            },
          })
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: buildLocalShellTranscript(result),
          })
          const sessionId = state.sessionId ?? crypto.randomUUID()
          try {
            const recorded = await httpClient.recordLocalShellTurn(sessionId, {
              command: result.command,
              cwd: result.cwd,
              shell: result.shell,
              args: result.args,
              stdout: result.stdout,
              stderr: result.stderr,
              exitCode: result.exitCode,
              signal: result.signal,
              durationMs: result.durationMs,
              timedOut: result.timedOut,
              maxBufferExceeded: result.maxBufferExceeded,
              title: `!${result.command}`,
              provider: state.provider,
              model: state.model,
              tags: ['local-shell'],
            })
            if (!state.sessionId) {
              dispatch({ type: 'SESSION_SET', sessionId: recorded.id })
            }
          } catch (error) {
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: `Local shell output was shown but not saved to session context: ${error instanceof Error ? error.message : String(error)}`,
            })
          }
        } finally {
          localShellAborterRef.current = null
          setLocalShellCommand(null)
        }

        return
      }

      void sendMessage(
        text,
        pendingAttachmentPaths.map((path) => ({ path })),
        { displayContent: options.displayText },
      ).then((sent) => {
        if (sent) {
          setQueuedAttachmentPaths([])
        }
      })
    },
    [
      dispatch,
      handleSlashCommand,
      httpClient,
      pendingAttachmentPaths,
      requestExit,
      sendMessage,
      selectedProject?.workingDirectory,
      state.autonomy,
      state.model,
      state.provider,
      state.sessionId,
      workspaceProject?.rootDir,
    ],
  )

  // Messages typed while the agent is streaming are queued and sent
  // automatically when the turn ends, instead of forcing the user to wait
  // (or Esc-abort) before giving the next instruction.
  const queuedInputsRef = useRef<Array<{ text: string; displayText?: string }>>([])
  const [queuedInputCount, setQueuedInputCount] = useState(0)

  const handleSend = useCallback(
    (text: string, options?: { displayText?: string }) => {
      if (isExitInput(text)) {
        requestExit()
        return
      }
      const immediateWhileBusy = shouldSubmitImmediatelyWhileBusy(text)
      if (inputBusy && !immediateWhileBusy) {
        queuedInputsRef.current.push({ text, displayText: options?.displayText })
        setQueuedInputCount(queuedInputsRef.current.length)
        return
      }
      void submitInput(text, options)
    },
    [inputBusy, requestExit, submitInput],
  )

  useEffect(() => {
    if (inputBusy) return
    const next = queuedInputsRef.current.shift()
    if (!next) return
    setQueuedInputCount(queuedInputsRef.current.length)
    void submitInput(next.text, next.displayText ? { displayText: next.displayText } : undefined)
  }, [inputBusy, submitInput])

  const autocompletePaletteCommand = useCallback(() => {
    if (!showPalette || commandPaletteItems.length === 0) return
    const selected =
      commandPaletteItems[Math.max(0, Math.min(paletteIndex, commandPaletteItems.length - 1))]
    setInputValue(selected.insertValue)
  }, [commandPaletteItems, paletteIndex, showPalette])

  const acceptPaletteCommand = useCallback((): boolean => {
    if (!showPalette || commandPaletteItems.length === 0) return false
    const selected =
      commandPaletteItems[Math.max(0, Math.min(paletteIndex, commandPaletteItems.length - 1))]
    if (!selected || selected.insertValue === inputValue) return false
    setInputValue(selected.insertValue)
    return true
  }, [commandPaletteItems, inputValue, paletteIndex, showPalette])

  const autocompleteAttachment = useCallback(() => {
    if (!activeAttachmentReference || attachmentSuggestions.length === 0) return
    const selected =
      attachmentSuggestions[
        Math.max(0, Math.min(attachmentSuggestionIndex, attachmentSuggestions.length - 1))
      ]
    setInputValue((current) =>
      applyAttachmentCompletion(current, activeAttachmentReference, selected.path),
    )
    if (!selected.isDirectory) {
      recordAttachment(selected.path)
    }
  }, [activeAttachmentReference, attachmentSuggestionIndex, attachmentSuggestions])

  const autocompleteSkill = useCallback(() => {
    if (!activeSkillReference || skillAutocompleteCandidates.length === 0) return
    const selected =
      skillAutocompleteCandidates[
        Math.max(0, Math.min(skillAutocompleteIndex, skillAutocompleteCandidates.length - 1))
      ]
    if (!selected) return
    setInputValue((current) => applySkillCompletion(current, activeSkillReference, selected.id))
  }, [activeSkillReference, skillAutocompleteCandidates, skillAutocompleteIndex])

  const exitPlanConfirmArmedRef = useRef(false)
  const togglePrimaryAgent = useCallback(() => {
    if (!state.sessionId) return
    const decision = resolvePlanModeToggle(
      primaryAgentId === 'plan' ? 'plan' : 'build',
      exitPlanConfirmArmedRef.current,
    )
    if (decision.kind === 'confirm-exit') {
      // First plan -> build toggle: arm a confirmation so the user
      // deliberately reviews the plan before the agent starts editing.
      exitPlanConfirmArmedRef.current = true
      dispatch({
        type: 'SYSTEM_MESSAGE',
        content:
          'Plan mode: review the plan above. Press the mode toggle again to exit plan mode and let the agent execute it, or keep working read-only.',
      })
      return
    }
    exitPlanConfirmArmedRef.current = false
    const next = decision.to
    setPrimaryAgentId(next)
    void httpClient.setSessionAgent(state.sessionId, next).catch(() => {
      setPrimaryAgentId(primaryAgentId)
    })
  }, [dispatch, httpClient, primaryAgentId, state.sessionId])
  const inputBoxController: Pick<
    React.ComponentProps<typeof InputBox>,
    | 'onPalettePrev'
    | 'onPaletteNext'
    | 'onPaletteAutocomplete'
    | 'onPaletteAccept'
    | 'onAttachmentPrev'
    | 'onAttachmentNext'
    | 'onAttachmentAutocomplete'
    | 'onSkillPrev'
    | 'onSkillNext'
    | 'onSkillAutocomplete'
    | 'onToggleMode'
  > = useMemo(
    () => ({
      onPalettePrev: () => {
        setPaletteIndex((current) => selectPreviousListIndex(current))
      },
      onPaletteNext: () => {
        setPaletteIndex((current) => selectNextListIndex(current, commandPaletteItems.length))
      },
      onPaletteAutocomplete: autocompletePaletteCommand,
      onPaletteAccept: acceptPaletteCommand,
      onAttachmentPrev: () => {
        setAttachmentSuggestionIndex((current) => selectPreviousListIndex(current))
      },
      onAttachmentNext: () => {
        setAttachmentSuggestionIndex((current) =>
          selectNextListIndex(current, attachmentSuggestions.length),
        )
      },
      onAttachmentAutocomplete: autocompleteAttachment,
      onSkillPrev: () => {
        setSkillAutocompleteIndex((current) => selectPreviousListIndex(current))
      },
      onSkillNext: () => {
        setSkillAutocompleteIndex((current) =>
          selectNextListIndex(current, skillAutocompleteCandidates.length),
        )
      },
      onSkillAutocomplete: autocompleteSkill,
      onToggleMode: togglePrimaryAgent,
    }),
    [
      attachmentSuggestions.length,
      acceptPaletteCommand,
      autocompleteAttachment,
      autocompletePaletteCommand,
      autocompleteSkill,
      commandPaletteItems.length,
      skillAutocompleteCandidates.length,
      togglePrimaryAgent,
    ],
  )

  const copyLastCodeBlock = useCallback(() => {
    const codeBlock = findLatestCopyableCodeBlock(state.messages, state.currentMessage)

    if (!codeBlock) {
      dispatch({
        type: 'SYSTEM_MESSAGE',
        content: 'No assistant code block is available to copy yet.',
      })
      return
    }

    const lineCount =
      codeBlock.content.length > 0 ? codeBlock.content.split(/\r\n|\r|\n/).length : 0
    const lineLabel = `${lineCount} line${lineCount === 1 ? '' : 's'}`
    const detail = codeBlock.language ? `${lineLabel}, ${codeBlock.language}` : lineLabel

    void copyTextToClipboard(codeBlock.content)
      .then(() => {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Copied last code block (${detail}) to clipboard.`,
        })
      })
      .catch((error) => {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content:
            error instanceof Error ? error.message : 'Failed to copy code block to clipboard.',
        })
      })
  }, [dispatch, state.currentMessage, state.messages])

  const handleCtrlC = useCallback(() => {
    if (state.pendingApproval) {
      // ApprovalModal owns Ctrl+C so the same request is not resolved twice by
      // the raw stdin listener and Ink's input listener.
      return
    }
    if (localShellAborterRef.current) {
      localShellAborterRef.current.abort()
      dispatch({ type: 'SYSTEM_MESSAGE', content: 'Local shell command cancelled.' })
      return
    }
    const intent = resolveCtrlCIntent({
      isStreaming: state.isStreaming,
      inputLength: inputValue.length,
      exitArmed: exitHint !== null,
      immediateExitWhenIdle: providerSetupOpen || providers.length === 0,
    })
    switch (intent.kind) {
      case 'exit':
        requestExit()
        return
      case 'cancel-stream':
        cancelStream()
        break
      case 'clear-input':
        setInputValue('')
        break
      case 'arm':
        break
    }
    showExitHint(intent.hint)
  }, [
    cancelStream,
    dispatch,
    exitHint,
    inputValue.length,
    providerSetupOpen,
    providers.length,
    requestExit,
    showExitHint,
    state.isStreaming,
    state.pendingApproval,
  ])

  useRawCtrlC(handleCtrlC)

  useKeybindings(
    {
      onPageUp: () => {
        if (
          isTranscriptNavigationBlocked(overlayState, {
            showCommandPalette,
            showAttachmentPalette,
            showSkillPalette,
            pendingApproval: Boolean(state.pendingApproval),
          })
        )
          return
        chatViewRef.current?.pageUp()
      },
      onPageDown: () => {
        if (
          isTranscriptNavigationBlocked(overlayState, {
            showCommandPalette,
            showAttachmentPalette,
            showSkillPalette,
            pendingApproval: Boolean(state.pendingApproval),
          })
        )
          return
        chatViewRef.current?.pageDown()
      },
      onCtrlP: () => {
        if (
          isTranscriptNavigationBlocked(overlayState, {
            showCommandPalette,
            showAttachmentPalette,
            showSkillPalette,
            pendingApproval: Boolean(state.pendingApproval),
          })
        )
          return
        chatViewRef.current?.pageUp()
      },
      onCtrlN: () => {
        if (
          isTranscriptNavigationBlocked(overlayState, {
            showCommandPalette,
            showAttachmentPalette,
            showSkillPalette,
            pendingApproval: Boolean(state.pendingApproval),
          })
        )
          return
        chatViewRef.current?.pageDown()
      },
      onCtrlShiftLeft: () => {
        void navigateRecentSession('older')
      },
      onCtrlShiftRight: () => {
        void navigateRecentSession('newer')
      },
      onCtrlC: handleCtrlC,
      onCtrlL: () => {
        if (composerOverlayBlocked) return
        startNewSession()
      },
      onCtrlO: () => {
        if (composerOverlayBlocked) return
        toggleCopyView()
      },
      onCtrlS: () => {
        if (shortcutInputBlocked) return
        // While a turn is streaming, Ctrl+S with in-progress composer text
        // steers the live run instead of queuing it (the default Enter
        // behavior) or opening the session picker (the non-busy binding).
        if (inputBusy && inputValue.trim()) {
          const message = inputValue.trim()
          setInputValue('')
          const sessionId = state.sessionId
          if (!sessionId) {
            dispatch({ type: 'SYSTEM_MESSAGE', content: 'No active session yet — send a message first.' })
            return
          }
          void submitSteer(httpClient, sessionId, message).then((result) => {
            if (result.ok) {
              rememberQueuedFollowUp(sessionId, result.noteId)
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: `[steer] ${formatQueuedSteerNote(
                  result.noteId,
                  result.pendingSteeringNoteCount,
                )} · Esc to undo`,
              })
              return
            }
            // Steer failed — restore the composer so the typed text isn't
            // lost. On 409 (no active run) the user can then just hit Enter
            // to send it as a normal message, matching the guidance text.
            setInputValue(result.originalMessage)
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: result.noActiveRun ? result.guidance : `Failed to steer: ${result.message}`,
            })
          })
          return
        }
        openSessionPicker()
      },
      onCtrlT: () => {
        if (modelPickerOpen) {
          closeModelPicker()
          return
        }
        if (shortcutInputBlocked) return
        openModelPicker()
      },
      onCtrlF: () => {
        if (filePickerOpen) {
          closeFilePicker()
          return
        }
        if (shortcutInputBlocked) return
        openFilePicker()
      },
      onCtrlG: () => {
        if (modePickerOpen) {
          closeModePicker()
          return
        }
        if (shortcutInputBlocked) return
        openModePicker()
      },
      onShiftTab: () => {
        if (shortcutInputBlocked) return
        void cycleAgentMode()
      },
      onCtrlA: () => {
        if (
          autonomyPickerOpen ||
          filePickerOpen ||
          modelPickerOpen ||
          providerSetupOpen ||
          providerDeleteOpen ||
          modePickerOpen ||
          sessionPickerOpen ||
          skillManagerOpen ||
          skillStoreOpen ||
          state.isStreaming ||
          state.pendingApproval
        ) {
          return
        }
        void applyAutonomySetting(nextAutonomyLevel(state.autonomy), 'shortcut')
      },
      onCtrlY: () => {
        if (composerOverlayBlocked) return
        copyLastCodeBlock()
      },
      onCtrlR: () => {
        if (composerOverlayBlocked) return
        if (attachmentSuggestions.length === 0) return
        invalidateFileIndex(FILE_PICKER_ROOT_DIR)
        void (async () => {
          const refreshed = await getFileIndex(FILE_PICKER_ROOT_DIR)
          setFileIndexPaths(new Set(refreshed.entries.map((entry) => entry.path)))
          setAttachmentReindexNotice(`reindexed (${refreshed.entries.length} entries)`)
          setTimeout(() => setAttachmentReindexNotice(null), 1500)
          if (activeAttachmentReference) {
            const items = await resolveAttachmentCandidates(
              activeAttachmentReference.path,
              FILE_PICKER_ROOT_DIR,
            )
            setAttachmentSuggestions(items)
            setAttachmentSuggestionIndex(0)
          }
        })()
      },
      onEscape: () => {
        if (approvalCommentActive) {
          // ApprovalModal's comment editor owns Esc (cancels the draft);
          // falling through here would cancel the whole run.
          return
        }
        if (state.pendingApproval) {
          // ApprovalModal maps Esc to a one-off denial. Do not also cancel the
          // surrounding run from this global key handler.
          return
        }
        if (localShellAborterRef.current) {
          localShellAborterRef.current.abort()
          dispatch({ type: 'SYSTEM_MESSAGE', content: 'Local shell command cancelled.' })
          return
        }
        if (state.isStreaming) {
          if (!inputValue.trim()) {
            const queued = [...queuedFollowUpsRef.current].reverse().find(
              (candidate) => candidate.sessionId === state.sessionId,
            )
            if (queued) {
              void cancelQueuedFollowUps({ noteId: queued.noteId })
              return
            }
          }
          cancelStream()
          return
        }
        switch (getForegroundOverlayId(overlayState)) {
          case 'help':
            setHelpOpen(false)
            return
          case 'autonomy-picker':
            closeAutonomyPicker()
            return
          case 'provider-setup':
            closeProviderSetup()
            return
          case 'provider-delete':
            closeProviderDeleteConfirm()
            return
          case 'model-picker':
            closeModelPicker()
            return
          case 'mode-picker':
            closeModePicker()
            return
          case 'file-picker':
            closeFilePicker()
            return
          case 'session-picker':
            closeSessionPicker()
            return
          case 'skill-manager':
            closeSkillManagerPicker()
            return
          case 'skill-store':
            closeSkillStorePicker()
            return
          default:
            break
        }
        if (showPalette) setInputValue('')
      },
      onApprove: () => {
        if (state.pendingApproval && !approvalCommentActive) {
          void resolveApproval(true, 'once')
        }
      },
      onApproveSession: () => {
        if (state.pendingApproval && !approvalCommentActive) {
          void resolveApproval(true, 'session')
        }
      },
      onApproveAlways: () => {
        if (state.pendingApproval && !approvalCommentActive) {
          void resolveApproval(true, 'always')
        }
      },
      onDeny: () => {
        if (state.pendingApproval && !approvalCommentActive) {
          void resolveApproval(false, 'once')
        }
      },
      onCtrlB: () => {
        if (shortcutInputBlocked) return
        setActivityPanelHidden((hidden) => !hidden)
      },
      // Ctrl+O terminal transcript mode takes over the keymap exclusively while
      // open so copy-mode keys do not accidentally fire pickers/autonomy/actions.
      copyViewActive: copyView,
      onCopyClose: closeCopyView,
    },
    !swarmAttachTarget,
  )

  if (connectionStatus === 'connecting') {
    return (
      <Box flexDirection="column" padding={1}>
        <Text color={colors.dimText}>Connecting to sepilotd at {config.url}...</Text>
      </Box>
    )
  }

  if (connectionStatus === 'error') {
    return (
      <Box flexDirection="column" padding={1}>
        <Text color={colors.error}>Cannot connect to sepilotd at {config.url}</Text>
        {connectionError ? <Text color={colors.dimText}>{connectionError}</Text> : null}
        <Text color={colors.dimText}>Run `sepilot start` to start the daemon.</Text>
      </Box>
    )
  }

  if (swarmAttachTarget) {
    return (
      <SwarmAttachView
        client={swarmClient}
        target={swarmAttachTarget}
        width={termWidth}
        height={termHeight}
        onExit={closeSwarmAttach}
      />
    )
  }

  // Layout heights
  // Reserve the very first row of the alt screen as a blank spacer.
  // Some terminals/SSH clients (Termius, some Wezterm/Ghostty configs) draw
  // chrome over row 1 of the alternate screen, which would clip the top
  // half of the header glyphs. By rendering row 1 as an empty row and
  // pushing actual content down, the header stays visible regardless of
  // the host terminal. The total output still fills exactly `termHeight`
  // rows so Ink takes the clearTerminal render path (no trailing newline
  // → no terminal scroll on full frames).
  const TOP_SPACER_ROWS = 1
  const layoutHeight = Math.max(1, termHeight)
  const INLINE_PANEL_GAP_ROWS = 1
  const MODAL_MARGIN_ROWS = 2
  const headerHeight = 2 // 1 content row + 1 borderBottom
  const maxInputEditorRows = Math.max(2, Math.min(8, Math.floor(termHeight / 4)))
  const inputHeight = calculateInputBoxHeight({
    value: inputValue,
    width: termWidth,
    maxEditorRows: maxInputEditorRows,
    isStreaming: inputBusy,
    streamLabel: activeProgressLabel,
    blockedReason: composerPresentation.blockedReason,
    blockedHint: composerPresentation.blockedHint,
    paletteOpen: composerPresentation.paletteOpen,
    attachmentAutocompleteOpen: composerPresentation.attachmentAutocompleteOpen,
    skillAutocompleteOpen: composerPresentation.skillAutocompleteOpen,
    pendingAttachments: composerPresentation.pendingAttachments,
    denialFollowup: state.denialFollowup,
    exitHint,
  })
  const statusHeight = 1
  const minimumChatHeight = state.pendingApproval
    ? 1
    : Math.max(3, Math.min(5, Math.floor(termHeight / 4)))
  let remainingInlineRows = Math.max(
    0,
    layoutHeight -
      TOP_SPACER_ROWS -
      headerHeight -
      inputHeight -
      statusHeight -
      minimumChatHeight -
      2,
  )
  const {
    showRagPanel,
    showMemorySearchPanel,
    showUsageDashboardPanel,
    showMcpPanel,
    showArtifactPanel,
    showToolActivityPanel: derivedToolActivityPanelVisible,
  } = deriveInlinePanelVisibility({
    overlayState,
    isStreaming: state.isStreaming,
    pendingApproval: Boolean(state.pendingApproval),
    showCommandPalette,
    showAttachmentPalette,
    showSkillPalette,
    ragPanelOpen,
    memorySearchOpen,
    usageDashboardOpen,
    showMcp,
    artifactCount: state.artifacts.length,
    toolCallCount: state.toolCalls.length,
  })
  const showToolActivityPanel = derivedToolActivityPanelVisible && !activityPanelHidden
  const ragPanelMaxHeight = Math.max(6, Math.min(14, Math.floor(termHeight / 3)))
  const ragHitRows =
    ragLoading || ragError || ragHits.length === 0
      ? 0
      : Math.min(ragHits.length, Math.max(0, Math.floor((ragPanelMaxHeight - 7) / 2)))
  const ragSourceRows =
    ragLoading || ragError || ragSources.length === 0
      ? 1
      : Math.min(ragSources.length, Math.max(1, ragPanelMaxHeight - 7 - ragHitRows))
  const desiredRagPanelHeight = showRagPanel
    ? Math.min(
        ragPanelMaxHeight,
        Math.max(6, 7 + (ragQuery ? 1 : 0) + (ragSyncResult ? 1 : 0) + ragHitRows + ragSourceRows),
      )
    : 0
  const ragPanelHeight = fitInlinePanelHeight(
    desiredRagPanelHeight,
    remainingInlineRows,
    6,
    INLINE_PANEL_GAP_ROWS,
  )
  const ragPanelRows = calculateRenderedPanelRows(ragPanelHeight, INLINE_PANEL_GAP_ROWS)
  remainingInlineRows -= ragPanelRows
  const memorySearchMaxHeight = Math.max(6, Math.min(12, Math.floor(termHeight / 3)))
  const memorySearchContentRows =
    memorySearchLoading || memorySearchError || memorySearchResults.length === 0
      ? 1
      : Math.min(
          memorySearchResults.length,
          Math.max(1, memorySearchMaxHeight - 4 - (memorySearchStatus ? 1 : 0)),
        )
  const desiredMemorySearchHeight = showMemorySearchPanel
    ? Math.min(
        memorySearchMaxHeight,
        Math.max(5, 4 + (memorySearchStatus ? 1 : 0) + memorySearchContentRows),
      )
    : 0
  const memorySearchHeight = fitInlinePanelHeight(
    desiredMemorySearchHeight,
    remainingInlineRows,
    5,
    INLINE_PANEL_GAP_ROWS,
  )
  const memorySearchRows = calculateRenderedPanelRows(memorySearchHeight, INLINE_PANEL_GAP_ROWS)
  remainingInlineRows -= memorySearchRows
  const usageDashboardMaxHeight = Math.max(5, Math.min(12, Math.floor(termHeight / 3)))
  const usageDashboardContentRows =
    usageDashboardLoading || usageDashboardError || usageBuckets.length === 0
      ? 1
      : Math.min(usageBuckets.length, Math.max(1, usageDashboardMaxHeight - 5))
  const desiredUsageDashboardHeight = showUsageDashboardPanel
    ? Math.min(
        usageDashboardMaxHeight,
        Math.max(
          5,
          3 + (usageSummary ? 1 : 0) + (usageDaily.length > 0 ? 1 : 0) + usageDashboardContentRows,
        ),
      )
    : 0
  const usageDashboardHeight = fitInlinePanelHeight(
    desiredUsageDashboardHeight,
    remainingInlineRows,
    5,
    INLINE_PANEL_GAP_ROWS,
  )
  const usageDashboardRows = calculateRenderedPanelRows(usageDashboardHeight, INLINE_PANEL_GAP_ROWS)
  remainingInlineRows -= usageDashboardRows
  const desiredMcpPanelHeight = showMcpPanel
    ? Math.max(4, Math.min(16, Math.floor(termHeight / 3)))
    : 0
  const mcpPanelHeight = fitInlinePanelHeight(desiredMcpPanelHeight, remainingInlineRows, 4)
  const mcpPanelRows = calculateRenderedPanelRows(mcpPanelHeight)
  remainingInlineRows -= mcpPanelRows
  const desiredToolActivityHeight = showToolActivityPanel
    ? Math.max(4, 3 + Math.min(3, state.toolCalls.length))
    : 0
  const toolActivityHeight = fitInlinePanelHeight(
    desiredToolActivityHeight,
    remainingInlineRows,
    4,
    INLINE_PANEL_GAP_ROWS,
  )
  const toolActivityRows = calculateRenderedPanelRows(toolActivityHeight, INLINE_PANEL_GAP_ROWS)
  remainingInlineRows -= toolActivityRows
  const desiredArtifactPanelHeight = showArtifactPanel ? 3 + Math.min(3, state.artifacts.length) : 0
  const artifactPanelHeight = fitInlinePanelHeight(
    desiredArtifactPanelHeight,
    remainingInlineRows,
    3,
    INLINE_PANEL_GAP_ROWS,
  )
  const artifactPanelRows = calculateRenderedPanelRows(artifactPanelHeight, INLINE_PANEL_GAP_ROWS)
  remainingInlineRows -= artifactPanelRows
  const approvalModalHeight = state.pendingApproval
    ? fitInlinePanelHeight(
        Math.max(6, Math.min(14, remainingInlineRows)),
        remainingInlineRows,
        4,
        MODAL_MARGIN_ROWS,
      )
    : 0
  const approvalModalRows = calculateRenderedPanelRows(approvalModalHeight, MODAL_MARGIN_ROWS)
  remainingInlineRows -= approvalModalRows
  const helpModalHeight = helpOpen
    ? fitInlinePanelHeight(
        Math.max(10, Math.min(28, remainingInlineRows)),
        remainingInlineRows,
        6,
        MODAL_MARGIN_ROWS,
      )
    : 0
  const helpModalRows = calculateRenderedPanelRows(helpModalHeight, MODAL_MARGIN_ROWS)
  remainingInlineRows -= helpModalRows
  const commandPaletteMaxVisibleItems = calculateListViewportCapacity(remainingInlineRows, 3, 8)
  const commandPaletteHeight = showCommandPalette
    ? fitInlinePanelHeight(
        calculateListPanelHeight({
          // Pin to max so the panel height does not shrink as the user types
          // and filters the list, which would reflow chatHeight per keystroke
          // and flicker the chat region.
          itemCount: commandPaletteMaxVisibleItems,
          reservedRows: 3,
          maxVisibleItems: commandPaletteMaxVisibleItems,
          minimumHeight: 4,
        }),
        remainingInlineRows,
        4,
      )
    : 0
  if (commandPaletteHeight > 0) {
    remainingInlineRows -= commandPaletteHeight
  }
  const attachmentPaletteMaxVisibleItems = calculateListViewportCapacity(remainingInlineRows, 3, 8)
  const attachmentPaletteHeight = showAttachmentPalette
    ? fitInlinePanelHeight(
        calculateListPanelHeight({
          // Pin to max — same rationale as commandPaletteHeight above.
          itemCount: attachmentPaletteMaxVisibleItems,
          reservedRows: 3,
          maxVisibleItems: attachmentPaletteMaxVisibleItems,
          minimumHeight: 4,
        }),
        remainingInlineRows,
        4,
      )
    : 0
  if (attachmentPaletteHeight > 0) {
    remainingInlineRows -= attachmentPaletteHeight
  }
  const skillPaletteMaxVisibleItems = calculateListViewportCapacity(remainingInlineRows, 3, 8)
  const skillPaletteHeight = showSkillPalette
    ? fitInlinePanelHeight(
        calculateListPanelHeight({
          itemCount: skillPaletteMaxVisibleItems,
          reservedRows: 3,
          maxVisibleItems: skillPaletteMaxVisibleItems,
          minimumHeight: 4,
        }),
        remainingInlineRows,
        4,
      )
    : 0
  if (skillPaletteHeight > 0) {
    remainingInlineRows -= skillPaletteHeight
  }
  const autonomyPickerMaxVisibleItems = calculateListViewportCapacity(
    remainingInlineRows,
    4,
    AUTONOMY_OPTIONS.length,
  )
  const autonomyPickerHeight = autonomyPickerOpen
    ? fitInlinePanelHeight(
        calculateListPanelHeight({
          itemCount: AUTONOMY_OPTIONS.length,
          reservedRows: 4,
          maxVisibleItems: autonomyPickerMaxVisibleItems,
          minimumHeight: 5,
        }),
        remainingInlineRows,
        5,
        MODAL_MARGIN_ROWS,
      )
    : 0
  const autonomyPickerRows = calculateRenderedPanelRows(autonomyPickerHeight, MODAL_MARGIN_ROWS)
  remainingInlineRows -= autonomyPickerRows
  const modelPickerMaxVisibleItems = calculateListViewportCapacity(remainingInlineRows, 5, 10)
  const modelPickerHeight = modelPickerOpen
    ? fitInlinePanelHeight(
        calculateListPanelHeight({
          itemCount:
            providersLoading || providersError || modelPickerList.ordered.length === 0
              ? 0
              : modelPickerList.ordered.length,
          reservedRows: 5,
          maxVisibleItems: modelPickerMaxVisibleItems,
          minimumHeight: 6,
        }),
        remainingInlineRows,
        6,
        MODAL_MARGIN_ROWS,
      )
    : 0
  const modelPickerRows = calculateRenderedPanelRows(modelPickerHeight, MODAL_MARGIN_ROWS)
  remainingInlineRows -= modelPickerRows
  const providerSetupItemCount =
    providerSetup?.step === 'preset'
      ? PROVIDER_WIZARD_PRESETS.length
      : providerSetup?.step === 'model'
        ? providerSetup.modelSuggestions.length
        : 0
  const providerSetupReservedRows =
    providerSetup?.step === 'confirm'
      ? 8
      : providerSetup?.step === 'model' || providerSetup?.step === 'headers'
        ? 7
        : 6
  const providerSetupMaxVisibleItems = calculateListViewportCapacity(
    remainingInlineRows,
    providerSetupReservedRows,
    providerSetup?.step === 'preset' ? PROVIDER_WIZARD_PRESETS.length : 8,
  )
  const providerSetupHeight = providerSetupOpen
    ? fitInlinePanelHeight(
        calculateListPanelHeight({
          itemCount: providerSetupItemCount,
          reservedRows: providerSetupReservedRows,
          maxVisibleItems: providerSetupMaxVisibleItems,
          minimumHeight: 6,
        }),
        remainingInlineRows,
        6,
        MODAL_MARGIN_ROWS,
      )
    : 0
  const providerSetupRows = calculateRenderedPanelRows(providerSetupHeight, MODAL_MARGIN_ROWS)
  remainingInlineRows -= providerSetupRows
  const providerDeleteHeight = providerDeleteOpen
    ? fitInlinePanelHeight(8, remainingInlineRows, 6, MODAL_MARGIN_ROWS)
    : 0
  const providerDeleteRows = calculateRenderedPanelRows(providerDeleteHeight, MODAL_MARGIN_ROWS)
  remainingInlineRows -= providerDeleteRows
  const modePickerMaxVisibleItems = calculateListViewportCapacity(remainingInlineRows, 5, 9)
  const modePickerHeight = modePickerOpen
    ? fitInlinePanelHeight(
        calculateListPanelHeight({
          itemCount:
            agentModesLoading || agentModesError || modePickerList.ordered.length === 0
              ? 0
              : modePickerList.ordered.length,
          reservedRows: 5,
          maxVisibleItems: modePickerMaxVisibleItems,
          minimumHeight: 6,
        }),
        remainingInlineRows,
        6,
        MODAL_MARGIN_ROWS,
      )
    : 0
  const modePickerRows = calculateRenderedPanelRows(modePickerHeight, MODAL_MARGIN_ROWS)
  remainingInlineRows -= modePickerRows
  const filePickerMaxVisibleItems = calculateListViewportCapacity(remainingInlineRows, 6, 10)
  const filePickerHeight = filePickerOpen
    ? fitInlinePanelHeight(
        calculateListPanelHeight({
          itemCount:
            filePickerLoading || filePickerError || filePickerItems.length === 0
              ? 0
              : filePickerItems.length,
          reservedRows: 6,
          maxVisibleItems: filePickerMaxVisibleItems,
          minimumHeight: 7,
        }),
        remainingInlineRows,
        7,
        MODAL_MARGIN_ROWS,
      )
    : 0
  const filePickerRows = calculateRenderedPanelRows(filePickerHeight, MODAL_MARGIN_ROWS)
  remainingInlineRows -= filePickerRows
  const skillManagerMaxVisibleItems = calculateListViewportCapacity(remainingInlineRows, 8, 10)
  const skillManagerHeight = skillManagerOpen
    ? fitInlinePanelHeight(
        calculateListPanelHeight({
          itemCount:
            installedSkillsLoading ||
            (skillManagerError ?? installedSkillsError) ||
            installedSkills.length === 0
              ? 0
              : installedSkills.length,
          reservedRows: 8,
          maxVisibleItems: skillManagerMaxVisibleItems,
          minimumHeight: 7,
        }),
        remainingInlineRows,
        7,
        MODAL_MARGIN_ROWS,
      )
    : 0
  const skillManagerRows = calculateRenderedPanelRows(skillManagerHeight, MODAL_MARGIN_ROWS)
  remainingInlineRows -= skillManagerRows
  const skillStoreMaxVisibleItems = calculateListViewportCapacity(remainingInlineRows, 8, 10)
  const skillStoreHeight = skillStoreOpen
    ? fitInlinePanelHeight(
        calculateListPanelHeight({
          itemCount:
            skillStoreLoading || skillStoreError || skillStoreResults.length === 0
              ? 0
              : skillStoreResults.length,
          reservedRows: 8,
          maxVisibleItems: skillStoreMaxVisibleItems,
          minimumHeight: 7,
        }),
        remainingInlineRows,
        7,
        MODAL_MARGIN_ROWS,
      )
    : 0
  const skillStoreRows = calculateRenderedPanelRows(skillStoreHeight, MODAL_MARGIN_ROWS)
  remainingInlineRows -= skillStoreRows
  const sessionPickerSectionRows =
    !sessionQuery.trim() && sessionPickerList.ordered.length > 0
      ? (sessionPickerList.project.length > 0 ? 1 : 0) +
        (sessionPickerList.recent.length > 0 ? 1 : 0) +
        (sessionPickerList.ordered.length >
        sessionPickerList.project.length + sessionPickerList.recent.length
          ? 1
          : 0)
      : 0
  const sessionPickerMaxVisibleItems = calculateListViewportCapacity(
    remainingInlineRows,
    5 + sessionPickerSectionRows,
    10,
  )
  const sessionPickerHeight = sessionPickerOpen
    ? fitInlinePanelHeight(
        calculateListPanelHeight({
          itemCount:
            sessionsLoading || sessionsError || sessionPickerList.ordered.length === 0
              ? 0
              : sessionPickerList.ordered.length,
          reservedRows: 5,
          maxVisibleItems: sessionPickerMaxVisibleItems,
          minimumHeight: 6,
          extraRows: sessionPickerSectionRows,
        }),
        remainingInlineRows,
        6,
        MODAL_MARGIN_ROWS,
      )
    : 0
  const sessionPickerRows = calculateRenderedPanelRows(sessionPickerHeight, MODAL_MARGIN_ROWS)
  const chatHeight = Math.max(
    1,
    layoutHeight -
      TOP_SPACER_ROWS -
      headerHeight -
      inputHeight -
      statusHeight -
      ragPanelRows -
      memorySearchRows -
      usageDashboardRows -
      mcpPanelRows -
      toolActivityRows -
      artifactPanelRows -
      approvalModalRows -
      helpModalRows -
      commandPaletteHeight -
      attachmentPaletteHeight -
      skillPaletteHeight -
      autonomyPickerRows -
      modelPickerRows -
      providerSetupRows -
      providerDeleteRows -
      modePickerRows -
      filePickerRows -
      skillManagerRows -
      skillStoreRows -
      sessionPickerRows -
      2,
  )
  const overlayPanels: React.ComponentProps<typeof OverlayPanels> = {
    commandPalette:
      showCommandPalette && commandPaletteHeight > 0
        ? {
            selectedIndex: paletteIndex,
            items: commandPaletteItems,
            maxVisibleItems: commandPaletteMaxVisibleItems,
            width: termWidth,
          }
        : null,
    attachmentPalette:
      showAttachmentPalette && attachmentPaletteHeight > 0
        ? {
            items: attachmentSuggestions,
            selectedIndex: attachmentSuggestionIndex,
            maxVisibleItems: attachmentPaletteMaxVisibleItems,
            reindexNotice: attachmentReindexNotice,
            width: termWidth,
          }
        : null,
    skillPalette:
      showSkillPalette && skillPaletteHeight > 0
        ? {
            items: skillAutocompleteCandidates,
            selectedIndex: skillAutocompleteIndex,
            loading: installedSkillsLoading,
            error: installedSkillsError,
            query: activeSkillReference?.query ?? '',
            maxVisibleItems: skillPaletteMaxVisibleItems,
            width: termWidth,
          }
        : null,
    autonomyPicker:
      autonomyPickerOpen && autonomyPickerHeight > 0
        ? {
            currentLevel: state.autonomy,
            selectedIndex: autonomyPickerIndex,
            maxVisibleItems: autonomyPickerMaxVisibleItems,
            onSelectIndex: setAutonomyPickerIndex,
            onConfirm: (autonomy) => {
              void applyAutonomySetting(autonomy, 'picker')
            },
            onClose: closeAutonomyPicker,
          }
        : null,
    modelPicker:
      modelPickerOpen && modelPickerHeight > 0
        ? {
            query: modelQuery,
            providers,
            currentProvider: state.provider,
            currentModel: state.model,
            defaultProvider: daemonDefaultProvider,
            defaultModel: daemonDefaultModel,
            mru: modelMru,
            selectedIndex: modelPickerIndex,
            loading: providersLoading,
            error: providersError,
            maxVisibleItems: modelPickerMaxVisibleItems,
            onQueryChange: setModelQuery,
            onSelectIndex: setModelPickerIndex,
            onConfirm: handleModelPickerSelection,
            onAddProvider: () => {
              void openProviderSetup({ allowFromModelPicker: true })
            },
            onClose: closeModelPicker,
          }
        : null,
    providerSetupModal:
      providerSetupOpen && providerSetup && providerSetupHeight > 0
        ? {
            mode: providerSetup.mode,
            step: providerSetup.step,
            applyToCurrentSession: providerSetup.applyToCurrentSession,
            preset: providerSetup.preset,
            presets: PROVIDER_WIZARD_PRESETS,
            selectedPresetIndex: providerSetup.selectedPresetIndex,
            providerId: providerSetup.providerId,
            baseUrl: providerSetup.baseUrl,
            apiKeyEnvVar: providerSetup.apiKeyEnvVar,
            apiKeyValue: providerSetup.apiKeyValue,
            headersText: providerSetup.headersText ?? '',
            model: providerSetup.model,
            modelSuggestions: providerSetup.modelSuggestions,
            selectedModelSuggestionIndex: providerSetup.modelSuggestionIndex,
            loading: providerSetup.loading,
            validating: providerSetup.validating,
            saving: providerSetup.saving,
            error: providerSetup.error,
            maxVisibleItems: providerSetupMaxVisibleItems,
            onSelectPresetIndex: (index) => {
              setProviderSetup((current) =>
                current ? { ...current, selectedPresetIndex: index } : current,
              )
            },
            onSelectPreset: selectProviderSetupPreset,
            onProviderIdChange: (value) => {
              setProviderSetup((current) =>
                current ? { ...current, providerId: value, error: null } : current,
              )
            },
            onBaseUrlChange: (value) => {
              setProviderSetup((current) =>
                current ? { ...current, baseUrl: value, error: null } : current,
              )
            },
            onApiKeyEnvVarChange: (value) => {
              setProviderSetup((current) =>
                current ? { ...current, apiKeyEnvVar: value, error: null } : current,
              )
            },
            onApiKeyValueChange: (value) => {
              setProviderSetup((current) =>
                current ? { ...current, apiKeyValue: value, error: null } : current,
              )
            },
            onHeadersTextChange: (value) => {
              setProviderSetup((current) =>
                current ? { ...current, headersText: value, error: null } : current,
              )
            },
            onModelChange: (value) => {
              setProviderSetup((current) =>
                current ? { ...current, model: value, error: null } : current,
              )
            },
            onSelectModelSuggestionIndex: (index) => {
              setProviderSetup((current) =>
                current ? { ...current, modelSuggestionIndex: index } : current,
              )
            },
            onApplyModelSuggestion: (index) => {
              setProviderSetup((current) => {
                if (!current) {
                  return current
                }
                const selected = current.modelSuggestions[index]
                return selected
                  ? {
                      ...current,
                      model: selected,
                      modelSuggestionIndex: index,
                      error: null,
                    }
                  : current
              })
            },
            onToggleApplyToCurrentSession: toggleProviderSetupApplyToCurrentSession,
            onNext: () => {
              void advanceProviderSetup()
            },
            onBack: goBackProviderSetup,
            onClose: closeProviderSetup,
          }
        : null,
    providerDeleteModal:
      providerDeleteOpen && providerDeleteConfirm && providerDeleteHeight > 0
        ? {
            providerId: providerDeleteConfirm.providerId,
            providerName: providerDeleteConfirm.providerName,
            nextProviderId: providerDeleteConfirm.nextProviderId,
            nextModel: providerDeleteConfirm.nextModel,
            currentProviderId: state.provider,
            currentModel: state.model,
            willSwitchCurrentSession: state.provider === providerDeleteConfirm.providerId,
            busy: providerDeleteConfirm.deleting,
            error: providerDeleteConfirm.error,
            onConfirm: () => {
              void confirmProviderDelete()
            },
            onClose: closeProviderDeleteConfirm,
          }
        : null,
    modePicker:
      modePickerOpen && modePickerHeight > 0
        ? {
            query: modeQuery,
            agents: agentModes,
            currentMode: state.mode,
            selectedIndex: modePickerIndex,
            loading: agentModesLoading,
            error: agentModesError,
            maxVisibleItems: modePickerMaxVisibleItems,
            onQueryChange: setModeQuery,
            onSelectIndex: setModePickerIndex,
            onConfirm: (agent) => {
              selectAgentMode(agent, 'picker')
            },
            onClose: closeModePicker,
          }
        : null,
    filePicker:
      filePickerOpen && filePickerHeight > 0
        ? {
            rootDir: FILE_PICKER_ROOT_DIR,
            currentDir: filePickerDir,
            items: filePickerItems,
            selectedIndex: filePickerIndex,
            selectedPaths: queuedAttachmentPaths.map((path) =>
              path.startsWith('/') ? path : join(FILE_PICKER_ROOT_DIR, path),
            ),
            selectedLabels: queuedAttachmentPaths,
            loading: filePickerLoading,
            error: filePickerError,
            maxVisibleItems: filePickerMaxVisibleItems,
            onSelectIndex: setFilePickerIndex,
            onOpenDirectory: (item) => {
              if (item.isDirectory) {
                setFilePickerDir(item.absolutePath)
              }
            },
            onNavigateUp: () => {
              const parent = dirname(filePickerDir)
              if (parent !== filePickerDir) {
                setFilePickerDir(parent)
              }
            },
            onToggleFile: (item) => {
              if (!item.isDirectory) {
                toggleQueuedAttachmentPath(item.absolutePath)
              }
            },
            onConfirm: closeFilePicker,
            onClose: closeFilePicker,
          }
        : null,
    sessionPicker:
      sessionPickerOpen && sessionPickerHeight > 0
        ? {
            query: sessionQuery,
            sessions: sessionItems,
            projectSessionIds: projectSessionIds,
            currentSessionId: state.sessionId,
            recentSessionIds: recentSessionIds,
            selectedIndex: sessionPickerIndex,
            loading: sessionsLoading,
            error: sessionsError,
            maxVisibleItems: sessionPickerMaxVisibleItems,
            onQueryChange: setSessionQuery,
            onSelectIndex: setSessionPickerIndex,
            onConfirm: (session) => {
              void loadSessionById(session.id)
            },
            onBranch: (session) => {
              void branchSessionAtTail(session.id)
            },
            onCompact: (session) => {
              void compactSessionById(session.id)
            },
            onDelete: (session) => {
              void deleteSessionById(session.id)
            },
            onExport: (session, format) => {
              closeSessionPicker()
              void exportSessionById(session.id, [format])
            },
            onRetry: retrySessionPicker,
            onClose: closeSessionPicker,
          }
        : null,
    skillManagerPicker:
      skillManagerOpen && skillManagerHeight > 0
        ? {
            query: skillManagerQuery,
            skills: installedSkills,
            selectedIndex: skillManagerIndex,
            loading: installedSkillsLoading,
            togglingSkillId: skillManagerTogglingId,
            error: skillManagerError ?? installedSkillsError,
            message: skillManagerMessage,
            maxVisibleItems: skillManagerMaxVisibleItems,
            onQueryChange: (value) => {
              setSkillManagerQuery(value)
              setSkillManagerIndex(0)
              setSkillManagerError(null)
              setSkillManagerMessage(null)
            },
            onSelectIndex: setSkillManagerIndex,
            onToggle: (skill) => {
              void toggleSkillFromManager(skill)
            },
            onRun: runSkillFromManager,
            onClose: closeSkillManagerPicker,
          }
        : null,
    skillStorePicker:
      skillStoreOpen && skillStoreHeight > 0
        ? {
            query: skillStoreQuery,
            searchedQuery: skillStoreSearchedQuery,
            results: skillStoreResults,
            selectedIndex: skillStoreIndex,
            loading: skillStoreLoading,
            installingSource: skillStoreInstallingSource,
            error: skillStoreError,
            message: skillStoreMessage,
            maxVisibleItems: skillStoreMaxVisibleItems,
            onQueryChange: (value) => {
              setSkillStoreQuery(value)
              setSkillStoreError(null)
              setSkillStoreMessage(null)
            },
            onSearch: (query) => {
              void searchSkillStore(query)
            },
            onSelectIndex: setSkillStoreIndex,
            onInstall: (result) => {
              void installSkillStoreResult(result)
            },
            onClose: closeSkillStorePicker,
          }
        : null,
    approvalModal:
      state.pendingApproval && approvalModalHeight > 0
        ? {
            approval: state.pendingApproval,
            height: approvalModalHeight,
            inputActive: !copyView && !hasForegroundOverlay(overlayState),
            onResolve: (approved, scope, note) => {
              setApprovalCommentActive(false)
              if (note === undefined) {
                void resolveApproval(approved, scope)
              } else {
                void resolveApproval(approved, scope, note)
              }
            },
            onCommentModeChange: setApprovalCommentActive,
          }
        : null,
    helpModal:
      helpOpen && helpModalHeight > 0
        ? {
            context: helpModalContext,
            height: helpModalHeight,
            onClose: () => {
              setHelpOpen(false)
            },
          }
        : null,
  }

  if (copyView) {
    return null
  }

  return (
    <Box flexDirection="column" height={layoutHeight} overflow="hidden">
      <Box height={TOP_SPACER_ROWS} flexShrink={0} />
      <Header
        version={version}
        // Before the user picks a model (or sends the first turn) the reducer
        // holds the 'default' sentinel. Show the resolved daemon default so a
        // fresh session displays the real model (e.g. qwen3.5:cloud @ ollama)
        // instead of "default @ default" (CLI_BACKLOG.md B9).
        model={state.model === 'default' && daemonDefaultModel ? daemonDefaultModel : state.model}
        provider={
          state.provider === 'default' && daemonDefaultProvider
            ? daemonDefaultProvider
            : state.provider
        }
        contextWindow={currentModelContextWindow}
        maxOutputTokens={currentModelMaxOutputTokens}
        projectName={state.projectName}
        width={termWidth}
      />
      <ChatView
        ref={chatViewRef}
        state={state}
        height={chatHeight}
        width={termWidth}
        transcriptClearedAt={transcriptClearedAt}
        activeRunLabel={streamingProgress?.label ?? state.streamStatus}
      />
      <InlineStatusPanels
        showToolActivityPanel={showToolActivityPanel}
        activities={state.activities}
        toolCalls={state.toolCalls}
        toolActivityHeight={toolActivityHeight}
        toolActivityMaxVisibleItems={3}
        runActive={state.isStreaming}
        showRagPanel={showRagPanel}
        ragQuery={ragQuery}
        ragSources={ragSources}
        ragHits={ragHits}
        ragVectorInfo={ragVectorInfo}
        ragSyncResult={ragSyncResult}
        ragLoading={ragLoading}
        ragError={ragError}
        ragSelectedIndex={ragSelectedIndex}
        ragPanelHeight={ragPanelHeight}
        onRagSelectIndex={setRagSelectedIndex}
        onRagSync={() => {
          void syncRagPanel()
        }}
        onRagClose={closeRagPanel}
        showMemorySearchPanel={showMemorySearchPanel}
        memorySearchQuery={memorySearchQuery}
        memorySearchResults={memorySearchResults}
        memorySearchStatus={memorySearchStatus}
        memorySearchLoading={memorySearchLoading}
        memorySearchError={memorySearchError}
        memorySearchHeight={memorySearchHeight}
        showUsageDashboardPanel={showUsageDashboardPanel}
        usageSummary={usageSummary}
        usageDaily={usageDaily}
        usageDashboardDays={usageDashboardDays}
        usageDashboardLoading={usageDashboardLoading}
        usageDashboardError={usageDashboardError}
        usageDashboardHeight={usageDashboardHeight}
        showMcpPanel={showMcpPanel}
        mcpPanelHeight={mcpPanelHeight}
        mcpClient={httpClient}
        showArtifactPanel={showArtifactPanel}
        artifacts={state.artifacts}
        artifactPanelHeight={artifactPanelHeight}
        plannerWorkingMemory={state.plannerWorkingMemory}
        stateBoardTodos={state.stateBoardCounts?.todos ?? null}
        editRollbacks={state.editRollbacks}
        debateRounds={state.debateRounds}
      />
      <OverlayPanels {...overlayPanels} />
      {/*
        Elastic filler. The frame *must* render exactly `layoutHeight` rows so
        Ink takes the `outputHeight >= stdout.rows` full-screen-clear path
        instead of incremental line patching — incremental updates of a frame
        that's a row short leave stale fragments behind (a previous frame's
        longer line bleeding through a shorter one). The heights above are
        budgeted to sum to `layoutHeight`, but a child that renders shorter
        than its reservation (e.g. an inline panel that bails to `null`) would
        under-fill. This grows to absorb any such slack so the total is always
        exactly `layoutHeight`; in the common case it is 0 rows and invisible.
      */}
      <Box flexGrow={1} flexShrink={1} minHeight={0} />
      <InputBox
        value={inputValue}
        onChange={setInputValue}
        onSubmit={handleSend}
        width={termWidth}
        maxEditorRows={maxInputEditorRows}
        isStreaming={inputBusy}
        streamLabel={activeProgressLabel}
        queuedCount={queuedInputCount}
        blockedReason={composerPresentation.blockedReason}
        blockedHint={composerPresentation.blockedHint}
        paletteOpen={composerPresentation.paletteOpen}
        attachmentAutocompleteOpen={composerPresentation.attachmentAutocompleteOpen}
        skillAutocompleteOpen={composerPresentation.skillAutocompleteOpen}
        pendingAttachments={composerPresentation.pendingAttachments}
        denialFollowup={state.denialFollowup}
        exitHint={exitHint}
        onExit={requestExit}
        {...inputBoxController}
      />
      <StatusBar presentation={statusBarPresentation} width={termWidth} />
    </Box>
  )
}
