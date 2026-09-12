import { readFileSync } from 'node:fs'
import { access, readFile, readdir, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { basename, dirname, extname, relative, resolve } from 'node:path'
import type {
  DaemonAgentDescriptor,
  DaemonAgentMode,
  DaemonMemoryEntry,
  DaemonMemorySemanticStatus,
  DaemonMcpServerStatus,
  DaemonDailyUsageSummary,
  DaemonProject,
  DaemonProviderInfo,
  DaemonRagFolder,
  DaemonRagSearchHit,
  DaemonRagVectorDbInfo,
  DaemonSessionMeta,
  DaemonSkill,
  DaemonUsageSummary,
  DaemonUserCommandSummary,
  SwarmClient,
} from '@sepilotd/api-client'
import { createSwarmClient, formatTokenSpeedStats } from '@sepilotd/api-client'
import { Box, Text, useApp, useInput, useStdout } from 'ink'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { DaemonClient } from '../../client/http.js'
import { loadDaemonToken } from '../../client/token.js'
import { cancelSteer, formatCancelledSteerNote } from '../../steer-shared.js'
import {
  detectOllamaModels,
  PROVIDER_WIZARD_PRESETS,
  type ProviderWizardPreset,
} from '../../utils/provider-presets.js'
import {
  formatProviderHeadersInput,
  parseProviderHeadersInput,
} from '../../utils/provider-http-options.js'
import {
  colors,
  getActiveThemeId,
  resolveThemeId,
  setActiveTheme,
  themeOptions,
  type ThemeId,
} from '../theme.js'
import type { AppConfig, Message, ToolCallState } from '../types.js'
import { ApprovalModal } from '../components/ApprovalModal.js'
import { FilePicker, type FilePickerItem } from '../components/FilePicker.js'
import { SwarmAttachView, type SwarmAttachTarget } from '../components/SwarmAttachView.js'
import { ProviderSetupModal, type ProviderSetupStep } from '../components/ProviderSetupModal.js'
import { ProviderDeleteModal } from '../components/ProviderDeleteModal.js'
import { ModelPicker } from '../components/ModelPicker.js'
import {
  buildProviderModelPickerList,
  type ProviderModelPickerItem,
} from '../utils/provider-models.js'
import { ModePicker } from '../components/ModePicker.js'
import { SessionPicker } from '../components/SessionPicker.js'
import { SkillManagerPicker } from '../components/SkillManagerPicker.js'
import { useChat } from '../hooks/useChat.js'
import { useInputHistory } from '../hooks/useInputHistory.js'
import { useTerminalSize } from '../hooks/useTerminalSize.js'
import { formatConversationForCopy } from '../utils/transcript-text.js'
import { findRewindTarget } from '../utils/session-history.js'
import {
  sessionBelongsToWorkspace,
  sessionWorkspaceLoadError,
} from '../utils/session-workspace.js'
import { buildTurnRecap, type TurnRecapSnapshot } from '../utils/turn-recap.js'
import { loadCliState, recordRecapPreference } from '../cli-state.js'
import { dockerAvailable, listManagedContainers } from '../../utils/docker.js'
import {
  buildTuiHooksUsage,
  formatTuiHookDeadLetter,
  formatTuiHookDelivery,
  formatTuiHookSummary,
  formatTuiMemoryAuditSummary,
  formatTuiMemoryLifecycleSummary,
  formatTuiMemoryMaintenanceSummary,
  formatTuiFileMemorySummary,
  formatTuiDailyMemory,
  formatTuiBacklogSummary,
  splitSectionAndContent,
  TUI_FILE_MEMORY_USAGE,
  parseTuiHookEvents,
  parseTuiHookHeaders,
  parseTuiHookLimit,
  parseTuiHookValueFlag,
} from '../utils/app-helpers.js'
import {
  A2A_COMMAND_USAGE,
  ACP_COMMAND_USAGE,
  buildA2aGuide,
  buildAcpCodexGuide,
  buildAcpConfigSnippet,
  buildAcpOpencodeGuide,
  buildAcpZedHint,
} from '../utils/integration-guides.js'
import {
  applyModelCapabilityOverride,
  describeModelCapabilities,
  parseCapabilityToggle,
  resolveCapabilityName,
  TOGGLEABLE_MODEL_CAPABILITIES,
} from '../utils/model-capabilities.js'
import { runMcpSlashCommand } from '../utils/mcp-slash.js'
import { runDaemonAdminSlashCommand, splitSlashCommandInput } from '../utils/daemon-admin-slash.js'
import { toolFilePath } from '../utils/tooling.js'
import { formatSkillInstallResult, SKILLS_COMMAND_USAGE } from '../utils/skill-store.js'
import {
  appendUniqueMemoryLine,
  findMemorySection,
  formatManualMemoryBacklogLine,
  removeMatchingMemoryLines,
  TUI_OPEN_LOOP_QUEUE_SECTION,
} from '../utils/memory.js'
import {
  buildProviderConfigRecord,
  buildProviderConfigUpdate,
  buildProviderDefaultUpdate,
  buildProviderDeleteUpdate,
  buildProviderModelSuggestions,
  buildProviderSecretEnvUpdate,
  buildProviderSetupDraft,
  findBuiltinProviderPreset,
  isValidProviderEnvVarName,
  managedProviderApiKeyEnvVar,
  reconcileProviderModelDiscovery,
  type ConfiguredProviderRecord,
  type ProviderSetupDraft,
} from '../utils/provider-setup.js'
import {
  applyAttachmentCompletion,
  findActiveAttachmentReference,
  resolveAttachmentCandidates,
  type AttachmentCandidate,
} from '../utils/attachments.js'
import {
  applySkillCompletion,
  filterSkillAutocompleteCandidates,
  findActiveSkillReference,
} from '../utils/skill-autocomplete.js'
import {
  buildLocalShellTranscript,
  resolveLocalShellGate,
  runLocalShellCommand,
  type LocalShellResult,
} from '../utils/local-shell.js'
import {
  COMMANDS,
  findByAlias,
  primaryCommands,
  type CommandDef,
} from './commands/registry.js'
import { runModeCommand, runModelCommand } from './commands/runtime-selection.js'
import { runProjectCommand, runUsageCommand } from './commands/workspace.js'
import { runSessionCommand } from './commands/session.js'
import { ChoiceDialog } from './dialogs/ChoiceDialog.js'
import { ArtifactDialog } from './dialogs/ArtifactDialog.js'
import { DialogStack, type DialogEntry } from './dialogs/DialogStack.js'
import { InfoDialog } from './dialogs/InfoDialog.js'
import { FilePreviewDialog } from './dialogs/FilePreviewDialog.js'
import { AgentLoopDialog } from './dialogs/AgentLoopDialog.js'
import { MemorySearchDialog } from './dialogs/MemorySearchDialog.js'
import { McpDialog } from './dialogs/McpDialog.js'
import { ProjectDialog } from './dialogs/ProjectDialog.js'
import { QuestionDialog } from './dialogs/QuestionDialog.js'
import { RagDialog } from './dialogs/RagDialog.js'
import { SettingsDialog } from './dialogs/SettingsDialog.js'
import { UsageDialog } from './dialogs/UsageDialog.js'
import { PlanTodoDialog } from './dialogs/PlanTodoDialog.js'
import { loadKeybindings, type KeybindingConfig } from './keys/bindings.js'
import {
  IDLE_LEADER_STATE,
  leaderHint,
  reduceLeaderKey,
  type LeaderState,
} from './keys/leader.js'
import { copyToClipboard, type ClipboardResult } from './runtime/clipboard.js'
import { createNodeClipboardDeps } from './runtime/clipboard-node.js'
import { editTextInExternalEditor } from './runtime/external-editor.js'
import { resolveComposerAttachments } from './runtime/composer-attachments.js'
import { pasteClipboardPng } from './runtime/clipboard-image-node.js'
import { formatInlineAgentProgress } from './runtime/agent-progress.js'
import {
  applyHangulInputChunk,
  composeHangul,
  createHangulInputState,
  replayInlineInputControls,
  stripBracketedPasteDelimiters,
} from '../utils/hangul.js'
import {
  clampGraphemeOffset,
  nextGraphemeOffset,
  previousGraphemeOffset,
} from '../utils/graphemes.js'
import { Composer } from './screens/Composer.js'
import { Conversation } from './screens/Conversation.js'
import { StatusLine } from './screens/StatusLine.js'
import { useActivityAnimation } from './hooks/useActivityAnimation.js'
import { useFunctionKeys } from './hooks/useFunctionKeys.js'
import { SETTINGS_ITEMS } from './settings/schema.js'
import {
  contextFillPercent,
  estimateConversationTokens,
} from '../utils/context-token-estimate.js'

const MODE_FALLBACKS = [
  { id: 'auto', name: 'Auto', description: 'Choose the best agent automatically' },
  { id: 'build', name: 'Build', description: 'Implement and validate changes' },
  { id: 'plan', name: 'Plan', description: 'Analyze and prepare a plan' },
] satisfies Array<{ id: DaemonAgentMode; name: string; description: string }>

const THINKING_OPTIONS = ['off', 'low', 'medium', 'high', 'max'].map((value) => ({
  value,
  label: value,
}))
const AUTONOMY_OPTIONS = [
  'readonly',
  'accept-edits',
  'workspace-write',
  'supervised',
  'autonomous',
].map((value) => ({ value, label: value }))
const MAX_TOKEN_OPTIONS = [
  { value: 'default', label: 'Provider default' },
  { value: '4096', label: '4,096' },
  { value: '8192', label: '8,192' },
  { value: '16384', label: '16,384' },
  { value: '32768', label: '32,768' },
]

export interface NextAppRuntime {
  client?: DaemonClient
  keybindings?: KeybindingConfig
  copy?: (text: string) => Promise<ClipboardResult>
  editText?: (text: string) => Promise<string>
  history?: string[]
  appendHistory?: (text: string) => void
  runShell?: (command: string, options: { signal: AbortSignal }) => Promise<LocalShellResult>
  attachmentCandidates?: (query: string) => Promise<AttachmentCandidate[]>
  swarmClient?: SwarmClient
  writeFile?: (path: string, content: string) => Promise<void>
  readFile?: (path: string) => Promise<string>
  pasteImage?: () => Promise<string>
  exit?: () => void
  hardwareCursor?: boolean
}

export interface NextAppProps {
  config: AppConfig
  runtime?: NextAppRuntime
}

interface InlineProviderSetupState {
  mode: 'new' | 'edit'
  step: ProviderSetupStep
  allowPresetSelection: boolean
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

interface InlineProviderDeleteState {
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

export function matchCommands(query: string): CommandDef[] {
  if (query === '/') return primaryCommands()
  const normalized = query.toLowerCase()
  return COMMANDS.filter((command) =>
    command.aliases.some((alias) => alias.toLowerCase().startsWith(normalized)),
  )
}

function preferredModelPickerIndex(options: {
  providers: DaemonProviderInfo[]
  currentProvider: string
  currentModel: string
  defaultProvider: string
  defaultModel: string
  target: 'session' | 'default'
}): number {
  const preferredProvider = options.target === 'default'
    ? options.defaultProvider
    : options.currentProvider
  const preferredModel = options.target === 'default'
    ? options.defaultModel
    : options.currentModel
  const ordered = buildProviderModelPickerList(
    options.providers,
    options.currentProvider,
    options.currentModel,
    '',
    {
      defaultProviderId: options.defaultProvider,
      defaultModelId: options.defaultModel,
    },
  ).ordered
  const index = ordered.findIndex((item) => (
    item.kind === 'model'
    && item.providerId === preferredProvider
    && item.modelId === preferredModel
  ))
  return index >= 0 ? index : 0
}

export function NextApp({ config, runtime }: NextAppProps) {
  const { exit: inkExit } = useApp()
  const exit = runtime?.exit ?? inkExit
  const { stdout } = useStdout()
  const { rows, columns: width } = useTerminalSize(stdout)
  const [client] = useState(() => runtime?.client ?? new DaemonClient(config.url))
  const [swarmClient] = useState(() => runtime?.swarmClient ?? createSwarmClient({ baseUrl: config.url, token: loadDaemonToken() }))
  const {
    state,
    dispatch,
    sendMessage,
    loadSession,
    answerQuestion,
    cancelStream,
    resolveApproval,
    getTokenSpeedStats,
    resetTokenSpeedStats,
  } = useChat(null, client)
  const [inputValue, setInputValue] = useState('')
  const [cursorOffset, setCursorOffset] = useState(0)
  const inputValueRef = useRef('')
  const cursorOffsetRef = useRef(0)
  const hangulInputStateRef = useRef(createHangulInputState())
  const applyComposerValue = useCallback((value: string, offset = value.length, resetComposition = true) => {
    const safeOffset = clampGraphemeOffset(value, offset)
    inputValueRef.current = value
    cursorOffsetRef.current = safeOffset
    if (resetComposition) hangulInputStateRef.current = createHangulInputState()
    setInputValue(value)
    setCursorOffset(safeOffset)
  }, [])
  const replaceComposerValue = useCallback((value: string) => {
    applyComposerValue(value)
  }, [applyComposerValue])
  const persistedHistory = useInputHistory()
  const history = runtime?.history ?? persistedHistory.history
  const appendHistory = runtime?.appendHistory ?? persistedHistory.append
  const historyIndexRef = useRef<number | null>(null)
  const [historyDraft, setHistoryDraft] = useState('')
  const [suggestionIndex, setSuggestionIndex] = useState(0)
  const [attachmentSuggestions, setAttachmentSuggestions] = useState<AttachmentCandidate[]>([])
  const [attachmentSuggestionIndex, setAttachmentSuggestionIndex] = useState(0)
  const [skillSuggestionIndex, setSkillSuggestionIndex] = useState(0)
  const lastShellCommand = useRef<string | null>(null)
  const shellAborter = useRef<AbortController | null>(null)
  const exitArmTimeout = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastEmptyEscapeAt = useRef(0)
  const queuedFollowUpsRef = useRef<Array<{ sessionId: string; noteId: string }>>([])
  const followUpCancelInFlightRef = useRef(false)
  const [exitArmed, setExitArmed] = useState(false)
  const [leaderState, setLeaderState] = useState<LeaderState>(IDLE_LEADER_STATE)
  const [dialogs, setDialogs] = useState<string[]>([])
  const [statusError, setStatusError] = useState<string | null>(null)
  const [info, setInfo] = useState<{ title: string; body: string } | null>(null)
  const [clearedMessageCount, setClearedMessageCount] = useState(0)
  const [providers, setProviders] = useState<DaemonProviderInfo[]>([])
  const [providerDefaults, setProviderDefaults] = useState({ provider: '', model: '' })
  const [providerMetadataLoading, setProviderMetadataLoading] = useState(true)
  const [modelQuery, setModelQuery] = useState('')
  const [modelIndex, setModelIndex] = useState(0)
  const [modelLoading, setModelLoading] = useState(false)
  const [modelError, setModelError] = useState<string | null>(null)
  const [modelPickerPurpose, setModelPickerPurpose] = useState<'session' | 'default'>('session')
  const [agents, setAgents] = useState<DaemonAgentDescriptor[]>(MODE_FALLBACKS)
  const [modeQuery, setModeQuery] = useState('')
  const [modeIndex, setModeIndex] = useState(0)
  const [modeLoading, setModeLoading] = useState(false)
  const [modeError, setModeError] = useState<string | null>(null)
  const [sessions, setSessions] = useState<DaemonSessionMeta[]>([])
  const [sessionQuery, setSessionQuery] = useState('')
  const [sessionIndex, setSessionIndex] = useState(0)
  const [sessionsLoading, setSessionsLoading] = useState(false)
  const [sessionsError, setSessionsError] = useState<string | null>(null)
  const [sessionRefresh, setSessionRefresh] = useState(0)
  const [questionBusy, setQuestionBusy] = useState(false)
  const [questionError, setQuestionError] = useState<string | null>(null)
  const [projects, setProjects] = useState<DaemonProject[]>([])
  const [projectsLoading, setProjectsLoading] = useState(false)
  const [projectsError, setProjectsError] = useState<string | null>(null)
  const [skills, setSkills] = useState<DaemonSkill[]>([])
  const [skillsQuery, setSkillsQuery] = useState('')
  const [skillsIndex, setSkillsIndex] = useState(0)
  const [skillsLoading, setSkillsLoading] = useState(false)
  const [skillsToggling, setSkillsToggling] = useState<string | null>(null)
  const [skillsError, setSkillsError] = useState<string | null>(null)
  const [skillsMessage, setSkillsMessage] = useState<string | null>(null)
  const [memoryStatus, setMemoryStatus] = useState<DaemonMemorySemanticStatus | null>(null)
  const [memoryInitialQuery, setMemoryInitialQuery] = useState('')
  const [memoryResults, setMemoryResults] = useState<DaemonMemoryEntry[]>([])
  const [memoryLoading, setMemoryLoading] = useState(false)
  const [memoryError, setMemoryError] = useState<string | null>(null)
  const [usageSummary, setUsageSummary] = useState<DaemonUsageSummary | null>(null)
  const [usageDaily, setUsageDaily] = useState<DaemonDailyUsageSummary[]>([])
  const [usageDays, setUsageDays] = useState(7)
  const [usageLoading, setUsageLoading] = useState(false)
  const [usageError, setUsageError] = useState<string | null>(null)
  const [ragInitialQuery, setRagInitialQuery] = useState('')
  const [ragRefresh, setRagRefresh] = useState(0)
  const [ragSources, setRagSources] = useState<DaemonRagFolder[]>([])
  const [ragHits, setRagHits] = useState<DaemonRagSearchHit[]>([])
  const [ragVectorInfo, setRagVectorInfo] = useState<DaemonRagVectorDbInfo | null>(null)
  const [ragLoading, setRagLoading] = useState(false)
  const [ragError, setRagError] = useState<string | null>(null)
  const [mcpInitialQuery, setMcpInitialQuery] = useState('')
  const [mcpServers, setMcpServers] = useState<DaemonMcpServerStatus[]>([])
  const [mcpLoading, setMcpLoading] = useState(false)
  const [mcpError, setMcpError] = useState<string | null>(null)
  const [themeId, setThemeId] = useState<ThemeId>(() => getActiveThemeId())
  const [autonomyPurpose, setAutonomyPurpose] = useState<'session' | 'default'>('session')
  const [defaultAutonomy, setDefaultAutonomy] = useState<typeof state.autonomy | null>(null)
  const [recapEnabled, setRecapEnabled] = useState(true)
  const [lastTurnRecap, setLastTurnRecap] = useState<string | null>(null)
  const [showToolDetails, setShowToolDetails] = useState(false)
  const [showThinking, setShowThinking] = useState(true)
  const [userCommands, setUserCommands] = useState<DaemonUserCommandSummary[]>([])
  const recapSnapshotRef = useRef<TurnRecapSnapshot | null>(null)
  const recapBaselineToolIdsRef = useRef<Set<string>>(new Set())
  const recapToolCallsRef = useRef<Map<string, ToolCallState>>(new Map())
  const idleToolIdsRef = useRef<Set<string>>(new Set())
  const fileRoot = useMemo(() => process.cwd(), [])
  const [fileDirectory, setFileDirectory] = useState(fileRoot)
  const [fileItems, setFileItems] = useState<FilePickerItem[]>([])
  const [fileIndex, setFileIndex] = useState(0)
  const [selectedFilePaths, setSelectedFilePaths] = useState<string[]>([])
  const [fileLoading, setFileLoading] = useState(false)
  const [fileError, setFileError] = useState<string | null>(null)
  const [filePreview, setFilePreview] = useState<{ path: string; content: string } | null>(null)
  const [lastSwarmRunId, setLastSwarmRunId] = useState<string | null>(null)
  const [redoSessionId, setRedoSessionId] = useState<string | null>(null)
  const [swarmAttachTarget, setSwarmAttachTarget] = useState<SwarmAttachTarget | null>(null)
  const [providerSetup, setProviderSetup] = useState<InlineProviderSetupState | null>(null)
  const [providerDelete, setProviderDelete] = useState<InlineProviderDeleteState | null>(null)

  const keybindings = useMemo(
    () => runtime?.keybindings ?? loadKeybindings((path) => readFileSync(path, 'utf8'), homedir()),
    [runtime?.keybindings],
  )
  const titles = useMemo(
    () => new Map(COMMANDS.map((command) => [command.id, command.title] as const)),
    [],
  )
  const closeTopDialog = useCallback(() => setDialogs((stack) => stack.slice(0, -1)), [])
  const openDialog = useCallback((id: string) => {
    setStatusError(null)
    setDialogs((stack) => (stack.at(-1) === id ? stack : [...stack, id]))
  }, [])
  const toggleInspectionDialog = useCallback((id: 'planTodo' | 'agentLoop') => {
    setStatusError(null)
    setDialogs((stack) => (
      stack.at(-1) === id
        ? stack.slice(0, -1)
        : [...stack.filter((entry) => entry !== id), id]
    ))
  }, [])
  const latestToolFilePath = useMemo(() => {
    const calls = [
      ...state.messages
        .slice(clearedMessageCount)
        .flatMap((message) => message.toolCall ? [message.toolCall] : []),
      ...state.toolCalls.filter(({ status }) => status === 'running' || status === 'pending'),
    ]
    for (let index = calls.length - 1; index >= 0; index -= 1) {
      const call = calls[index]
      const path = toolFilePath(call?.input, call?.name)
      if (path) return path
    }
    return null
  }, [clearedMessageCount, state.messages, state.toolCalls])
  const openLatestFilePreview = useCallback(async () => {
    if (!latestToolFilePath) {
      setStatusError('No recent file tool result is available to open.')
      return
    }
    const absolutePath = latestToolFilePath.startsWith('~/')
      ? resolve(homedir(), latestToolFilePath.slice(2))
      : resolve(fileRoot, latestToolFilePath)
    try {
      const content = runtime?.readFile
        ? await runtime.readFile(absolutePath)
        : await readFile(absolutePath, 'utf8')
      setFilePreview({ path: absolutePath, content })
      openDialog('filePreview')
    } catch (error) {
      setStatusError(`Cannot open ${latestToolFilePath}: ${errorMessage(error)}`)
    }
  }, [fileRoot, latestToolFilePath, openDialog, runtime])
  const openModelPicker = useCallback((target: 'session' | 'default' = 'session') => {
    setModelPickerPurpose(target)
    setModelQuery('')
    setModelError(null)
    setModelIndex(preferredModelPickerIndex({
      providers,
      currentProvider: state.provider,
      currentModel: state.model,
      defaultProvider: providerDefaults.provider,
      defaultModel: providerDefaults.model,
      target,
    }))
    openDialog('model')
  }, [openDialog, providerDefaults.model, providerDefaults.provider, providers, state.model, state.provider])

  useEffect(() => {
    if (config.model) dispatch({ type: 'SET_MODEL', model: config.model })
    if (config.provider) dispatch({ type: 'SET_PROVIDER', provider: config.provider })
  }, [config.model, config.provider, dispatch])

  useEffect(() => {
    let active = true
    void loadCliState().then((saved) => {
      if (active && saved.recapEnabled === false) setRecapEnabled(false)
    })
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (state.isStreaming) {
      if (!recapSnapshotRef.current) {
        setLastTurnRecap(null)
        recapSnapshotRef.current = {
          startedAt: Date.now(),
          toolCallCount: 0,
        }
        recapBaselineToolIdsRef.current = new Set(idleToolIdsRef.current)
        recapToolCallsRef.current = new Map()
      }
      for (const call of state.toolCalls) {
        if (!recapBaselineToolIdsRef.current.has(call.id)) {
          recapToolCallsRef.current.set(call.id, call)
        }
      }
      return
    }
    const snapshot = recapSnapshotRef.current
    recapSnapshotRef.current = null
    const turnToolCalls = [...recapToolCallsRef.current.values()]
    recapBaselineToolIdsRef.current = new Set()
    recapToolCallsRef.current = new Map()
    idleToolIdsRef.current = new Set(state.toolCalls.map((call) => call.id))
    if (!snapshot || !recapEnabled) return
    // Tool rows can be transient: some daemon surfaces omit them from the
    // finalized assistant message. Use the calls observed during the run so
    // the recap does not silently disappear at completion.
    const recap = buildTurnRecap(snapshot, turnToolCalls, state.usage, Date.now())
    setLastTurnRecap(recap)
  }, [recapEnabled, state.isStreaming, state.toolCalls, state.usage])

  useEffect(() => () => {
    if (exitArmTimeout.current) clearTimeout(exitArmTimeout.current)
  }, [])

  useEffect(() => {
    let active = true
    void Promise.all([client.config(), client.providers()])
      .then(([daemonConfig, nextProviders]) => {
        if (!active) return
        setProviders(nextProviders)
        const defaultProvider = daemonConfig.agent?.defaultProvider ?? ''
        const defaultModel = daemonConfig.agent?.defaultModel ?? ''
        setProviderDefaults({ provider: defaultProvider, model: defaultModel })
        if (!config.provider && defaultProvider) dispatch({ type: 'SET_PROVIDER', provider: defaultProvider })
        if (!config.model && defaultModel) dispatch({ type: 'SET_MODEL', model: defaultModel })
        if (daemonConfig.agent?.autonomy) dispatch({ type: 'SET_AUTONOMY', autonomy: daemonConfig.agent.autonomy })
        if (daemonConfig.agent?.thinkingLevel) dispatch({ type: 'SET_THINKING_LEVEL', thinkingLevel: daemonConfig.agent.thinkingLevel })
      })
      .catch(() => {
        // Startup remains usable when metadata is temporarily unavailable;
        // the model picker exposes retryable errors when opened explicitly.
      })
      .finally(() => {
        if (active) setProviderMetadataLoading(false)
      })
    return () => { active = false }
  }, [client, config.model, config.provider, dispatch])

  useEffect(() => {
    let active = true
    void client.userCommands()
      .then((commands) => { if (active) setUserCommands(commands) })
      .catch(() => { /* Custom command discovery is optional while the daemon is unavailable. */ })
    return () => { active = false }
  }, [client])

  useEffect(() => {
    if (!config.sessionId && !config.resume) return
    let active = true
    const hydrate = async () => {
      try {
        let sessionId = config.sessionId
        if (!sessionId) {
          const listed = await client.sessions(undefined, {
            perPage: 100,
            workspaceRoot: fileRoot,
          })
          sessionId = listed.items
            .filter((session) => sessionBelongsToWorkspace(session, fileRoot))
            .slice()
            .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))[0]?.id
        }
        if (!sessionId) {
          if (active) setStatusError('No session is available to resume.')
          return
        }
        const [session, artifacts] = await Promise.all([
          client.session(sessionId),
          client.sessionArtifacts(sessionId).catch(() => []),
        ])
        const workspaceError = sessionWorkspaceLoadError(session, fileRoot)
        if (workspaceError) {
          if (active) setStatusError(workspaceError)
          return
        }
        if (active) loadSession(session, artifacts)
      } catch (error) {
        if (active) setStatusError(`Could not load session: ${errorMessage(error)}`)
      }
    }
    void hydrate()
    return () => { active = false }
  }, [client, config.resume, config.sessionId, fileRoot, loadSession])

  useEffect(() => {
    if (keybindings.warnings.length > 0) setStatusError(keybindings.warnings.join(' · '))
  }, [keybindings])

  useEffect(() => {
    if (dialogs.at(-1) !== 'model') return
    let active = true
    setModelLoading(true)
    setModelError(null)
    void Promise.all([client.providers(), client.config()])
      .then(([nextProviders, daemonConfig]) => {
        if (!active) return
        const nextDefaults = {
          provider: daemonConfig.agent?.defaultProvider ?? '',
          model: daemonConfig.agent?.defaultModel ?? '',
        }
        setProviders(nextProviders)
        setProviderDefaults(nextDefaults)
        setModelIndex(preferredModelPickerIndex({
          providers: nextProviders,
          currentProvider: state.provider,
          currentModel: state.model,
          defaultProvider: nextDefaults.provider,
          defaultModel: nextDefaults.model,
          target: modelPickerPurpose,
        }))
      })
      .catch((error) => active && setModelError(errorMessage(error)))
      .finally(() => active && setModelLoading(false))
    return () => { active = false }
  }, [client, dialogs, modelPickerPurpose, state.model, state.provider])

  useEffect(() => {
    if (dialogs.at(-1) !== 'mode') return
    let active = true
    setModeLoading(true)
    setModeError(null)
    void client.agents()
      .then((nextAgents) => active && setAgents(nextAgents.length > 0 ? nextAgents : MODE_FALLBACKS))
      .catch((error) => active && setModeError(errorMessage(error)))
      .finally(() => active && setModeLoading(false))
    return () => { active = false }
  }, [client, dialogs])

  useEffect(() => {
    if (dialogs.at(-1) !== 'sessions') return
    let active = true
    const timeout = setTimeout(() => {
      setSessionsLoading(true)
      setSessionsError(null)
      void client.sessions(sessionQuery.trim() || undefined, {
        perPage: 100,
        workspaceRoot: fileRoot,
      })
        .then((result) => {
          if (!active) return
          setSessions(result.items.filter((session) => (
            sessionBelongsToWorkspace(session, fileRoot)
          )))
          setSessionIndex(0)
        })
        .catch((error) => active && setSessionsError(errorMessage(error)))
        .finally(() => active && setSessionsLoading(false))
    }, sessionQuery ? 150 : 0)
    return () => { active = false; clearTimeout(timeout) }
  }, [client, dialogs, fileRoot, sessionQuery, sessionRefresh])

  useEffect(() => {
    if (dialogs.at(-1) !== 'projects') return
    let active = true
    setProjectsLoading(true)
    setProjectsError(null)
    void client.projects()
      .then((items) => active && setProjects(items))
      .catch((error) => active && setProjectsError(errorMessage(error)))
      .finally(() => active && setProjectsLoading(false))
    return () => { active = false }
  }, [client, dialogs])

  useEffect(() => {
    if (dialogs.at(-1) !== 'skills') return
    let active = true
    setSkillsLoading(true)
    setSkillsError(null)
    setSkillsMessage('Enter toggles a skill. Tab prepares a skill run.')
    void client.skills({ includeDisabled: true, cwd: fileRoot, workspaceRoot: fileRoot })
      .then((items) => active && setSkills(items))
      .catch((error) => active && setSkillsError(errorMessage(error)))
      .finally(() => active && setSkillsLoading(false))
    return () => { active = false }
  }, [client, dialogs, fileRoot])

  useEffect(() => {
    if (dialogs.at(-1) !== 'memory') return
    let active = true
    setMemoryError(null)
    void client.memoryStatus()
      .then((status) => active && setMemoryStatus(status))
      .catch((error) => active && setMemoryError(errorMessage(error)))
    return () => { active = false }
  }, [client, dialogs])

  useEffect(() => {
    if (dialogs.at(-1) !== 'usage') return
    let active = true
    setUsageLoading(true)
    setUsageError(null)
    void Promise.all([client.usage(), client.usageDaily(usageDays)])
      .then(([summary, daily]) => {
        if (!active) return
        setUsageSummary(summary)
        setUsageDaily(daily)
      })
      .catch((error) => active && setUsageError(errorMessage(error)))
      .finally(() => active && setUsageLoading(false))
    return () => { active = false }
  }, [client, dialogs, usageDays])

  useEffect(() => {
    if (dialogs.at(-1) !== 'rag') return
    let active = true
    setRagLoading(true)
    setRagError(null)
    void Promise.allSettled([
      client.ragFolders(),
      client.ragVectorDbInfo(),
      ragInitialQuery ? client.searchRag(ragInitialQuery, 8) : Promise.resolve([]),
    ])
      .then(([sourcesResult, vectorResult, hitsResult]) => {
        if (!active) return
        setRagSources(sourcesResult.status === 'fulfilled' ? sourcesResult.value : [])
        setRagVectorInfo(vectorResult.status === 'fulfilled' ? vectorResult.value : null)
        setRagHits(hitsResult.status === 'fulfilled' ? hitsResult.value : [])
        const failures = [sourcesResult, vectorResult, hitsResult]
          .filter((result): result is PromiseRejectedResult => result.status === 'rejected')
          .map((result) => errorMessage(result.reason))
        if (failures.length > 0) setRagError(failures.join(' · '))
      })
      .finally(() => active && setRagLoading(false))
    return () => { active = false }
  }, [client, dialogs, ragInitialQuery, ragRefresh])

  useEffect(() => {
    if (dialogs.at(-1) !== 'mcp') return
    let active = true
    setMcpLoading(true)
    setMcpError(null)
    void client.mcpServers()
      .then((servers) => active && setMcpServers(servers))
      .catch((error) => active && setMcpError(errorMessage(error)))
      .finally(() => active && setMcpLoading(false))
    return () => { active = false }
  }, [client, dialogs])

  useEffect(() => {
    if (dialogs.at(-1) !== 'files') return
    let active = true
    setFileLoading(true)
    setFileError(null)
    void readdir(fileDirectory, { withFileTypes: true })
      .then((entries) => {
        if (!active) return
        setFileItems(entries
          .filter((entry) => !entry.name.startsWith('.') || entry.name === '.github')
          .map((entry) => ({
            absolutePath: resolve(fileDirectory, entry.name),
            label: entry.name,
            isDirectory: entry.isDirectory(),
          }))
          .sort((a, b) => Number(b.isDirectory) - Number(a.isDirectory) || a.label.localeCompare(b.label)))
        setFileIndex(0)
      })
      .catch((error) => active && setFileError(errorMessage(error)))
      .finally(() => active && setFileLoading(false))
    return () => { active = false }
  }, [dialogs, fileDirectory])

  const showNotice = useCallback((content: string) => {
    dispatch({ type: 'SYSTEM_MESSAGE', content })
  }, [dispatch])

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
    if (!sessionId) {
      showNotice('There is no active session with pending follow-ups.')
      return
    }
    if (followUpCancelInFlightRef.current) return
    followUpCancelInFlightRef.current = true
    try {
      const result = await cancelSteer(client, sessionId, target)
      if (result.ok) {
        const cancelledIds = new Set(result.cancelledNoteIds)
        queuedFollowUpsRef.current = queuedFollowUpsRef.current.filter(
          (queued) => !cancelledIds.has(queued.noteId),
        )
        showNotice(formatCancelledSteerNote(
          result.cancelledNoteIds,
          result.pendingSteeringNoteCount,
        ))
        return
      }
      if (
        'noteId' in target
        && ['already_consumed', 'already_cancelled', 'not_found', 'no_pending', 'no_active_run']
          .includes(result.reason)
      ) {
        queuedFollowUpsRef.current = queuedFollowUpsRef.current.filter(
          (queued) => queued.noteId !== target.noteId,
        )
      } else if (['no_pending', 'no_active_run'].includes(result.reason)) {
        queuedFollowUpsRef.current = queuedFollowUpsRef.current.filter(
          (queued) => queued.sessionId !== sessionId,
        )
      }
      showNotice(result.message)
    } finally {
      followUpCancelInFlightRef.current = false
    }
  }, [client, showNotice, state.sessionId])

  useEffect(() => {
    if (state.isStreaming || !state.sessionId) return
    queuedFollowUpsRef.current = queuedFollowUpsRef.current.filter(
      (queued) => queued.sessionId !== state.sessionId,
    )
  }, [state.isStreaming, state.sessionId])

  const openProviderSetup = useCallback(async (providerId?: string, presetQuery?: string) => {
    if (state.isStreaming) { setStatusError('Finish the active run before changing provider configuration.'); return }
    try {
      const daemonConfig = await client.config()
      const configProviders = (daemonConfig.providers ?? []) as ConfiguredProviderRecord[]
      const existing = providerId ? configProviders.find((provider) => provider.id === providerId) : undefined
      if (providerId && !existing) throw new Error(`Provider ${providerId} is not configured.`)
      const requestedPreset = presetQuery
        ? PROVIDER_WIZARD_PRESETS.find((preset) => preset.type === presetQuery || preset.label.toLowerCase().includes(presetQuery.toLowerCase()))
        : undefined
      const preset = requestedPreset ?? findBuiltinProviderPreset(existing?.type) ?? PROVIDER_WIZARD_PRESETS[0] ?? null
      if (!preset) throw new Error('No provider presets are available.')
      const draft = buildProviderSetupDraft({ provider: existing, preset, currentProvider: state.provider, currentModel: state.model })
      setProviderSetup({
        mode: existing ? 'edit' : 'new',
        step: providerId || requestedPreset ? 'providerId' : 'preset',
        allowPresetSelection: !providerId && !requestedPreset,
        applyToCurrentSession: false,
        preset,
        selectedPresetIndex: Math.max(0, PROVIDER_WIZARD_PRESETS.findIndex((entry) => entry.type === preset.type)),
        sourceProviderId: draft.sourceProviderId,
        providerId: draft.providerId,
        baseUrl: draft.baseUrl,
        apiKeyEnvVar: draft.apiKeyEnvVar,
        apiKeyValue: '',
        headersText: formatProviderHeadersInput(draft.headers),
        model: draft.model,
        modelSuggestions: preset.type === 'ollama' ? [] : buildProviderModelSuggestions(draft),
        modelSuggestionIndex: 0,
        modelAliasSource: undefined,
        configProviders,
        loading: false,
        validating: false,
        saving: false,
        error: null,
      })
      openDialog('providerSetup')
    } catch (error) { setStatusError(errorMessage(error)) }
  }, [client, openDialog, state.isStreaming, state.model, state.provider])

  const selectProviderPreset = useCallback((index: number) => {
    const preset = PROVIDER_WIZARD_PRESETS[index]
    if (!preset) return
    setProviderSetup((current) => {
      if (!current) return current
      const draft = buildProviderSetupDraft({ preset, currentProvider: state.provider, currentModel: state.model })
      return { ...current, preset, selectedPresetIndex: index, sourceProviderId: null, providerId: draft.providerId, baseUrl: draft.baseUrl, apiKeyEnvVar: draft.apiKeyEnvVar, apiKeyValue: '', headersText: formatProviderHeadersInput(draft.headers), model: draft.model, modelSuggestions: preset.type === 'ollama' ? [] : buildProviderModelSuggestions(draft), modelSuggestionIndex: 0, modelAliasSource: undefined, step: 'providerId', error: null }
    })
  }, [state.model, state.provider])

  const saveProviderSetup = useCallback(async (current: InlineProviderSetupState) => {
    if (!current.preset) return
    const draft: ProviderSetupDraft = {
      sourceProviderId: current.sourceProviderId,
      preset: current.preset,
      providerId: current.providerId.trim(),
      baseUrl: current.baseUrl.trim(),
      apiKeyEnvVar: current.apiKeyEnvVar.trim(),
      headers: parseProviderHeadersInput(current.headersText ?? ''),
      model: current.model.trim(),
      models: current.preset.type === 'custom' || current.preset.type === 'ollama' ? current.modelSuggestions : undefined,
      modelAliasSource: current.modelAliasSource,
    }
    const secretUpdate = buildProviderSecretEnvUpdate({ preset: current.preset, apiKeyEnvVar: draft.apiKeyEnvVar, apiKeyValue: current.apiKeyValue })
    setProviderSetup((value) => value ? { ...value, validating: true, error: null } : value)
    try {
      const provider = buildProviderConfigRecord({ config: { providers: current.configProviders }, draft })
      await client.validateProvider({ provider, model: draft.model, env: secretUpdate?.updates ? Object.fromEntries(Object.entries(secretUpdate.updates).filter((entry): entry is [string, string] => typeof entry[1] === 'string')) : undefined })
      setProviderSetup((value) => value ? { ...value, validating: false, saving: true } : value)
      if (secretUpdate) await client.updateConfigEnv(secretUpdate)
      await client.updateConfig(buildProviderConfigUpdate({ config: { providers: current.configProviders }, draft }))
      await client.refreshProviderModels(draft.providerId).catch(() => undefined)
      const refreshed = await client.providers().catch(() => [])
      setProviders(refreshed)
      setProviderDefaults({ provider: draft.providerId, model: draft.model })
      if (current.applyToCurrentSession) {
        dispatch({ type: 'SET_PROVIDER', provider: draft.providerId })
        dispatch({ type: 'SET_MODEL', model: draft.model })
      }
      setProviderSetup(null)
      closeTopDialog()
      showNotice(`${current.mode === 'edit' ? 'Updated' : 'Added'} provider ${draft.providerId}/${draft.model}.`)
    } catch (error) {
      setProviderSetup((value) => value ? { ...value, validating: false, saving: false, error: errorMessage(error) } : value)
    }
  }, [client, closeTopDialog, dispatch, showNotice])

  const advanceProviderSetup = useCallback(async () => {
    const current = providerSetup
    if (!current || !current.preset || current.loading || current.validating || current.saving) return
    const required = (message: string) => setProviderSetup((value) => value ? { ...value, error: message } : value)
    if (current.step === 'preset') { selectProviderPreset(current.selectedPresetIndex); return }
    if (current.step === 'providerId') {
      if (!current.providerId.trim()) { required('Provider id is required.'); return }
      setProviderSetup({ ...current, providerId: current.providerId.trim(), step: current.preset.type === 'ollama' || current.preset.type === 'custom' ? 'baseUrl' : 'apiKeyValue', error: null })
      return
    }
    if (current.step === 'baseUrl') {
      const raw = current.baseUrl.trim().replace(/\/+$/, '')
      if (!raw) { required('Base URL is required.'); return }
      const baseUrl = current.preset.type === 'custom' && !raw.endsWith('/v1') ? `${raw}/v1` : raw
      if (current.preset.type === 'custom') {
        setProviderSetup({ ...current, baseUrl, step: 'apiKeyValue', error: null })
        return
      }
      setProviderSetup({ ...current, baseUrl, loading: true, error: null })
      const discovered = await detectOllamaModels(baseUrl).catch(() => [])
      setProviderSetup((value) => {
        if (!value) return value
        const selection = reconcileProviderModelDiscovery({
          currentModel: value.model,
          discoveredModels: discovered,
          fallbackModels: value.modelSuggestions.length > 0
            ? value.modelSuggestions
            : current.preset?.suggestedModels,
          defaultModel: current.preset?.type === 'ollama' ? 'llama3.3' : undefined,
        })
        return {
          ...value,
          baseUrl,
          loading: false,
          ...selection,
          step: 'model',
          error: null,
        }
      })
      return
    }
    if (current.step === 'apiKeyEnv') {
      const apiKeyEnvVar = current.apiKeyEnvVar.trim()
      if (!apiKeyEnvVar && current.preset.type !== 'custom') { required('API key env var is required.'); return }
      if (apiKeyEnvVar && !isValidProviderEnvVarName(apiKeyEnvVar)) { required('API key env var must match [A-Za-z_][A-Za-z0-9_]*.'); return }
      setProviderSetup({ ...current, apiKeyEnvVar, step: current.preset.type === 'custom' ? 'headers' : 'model', error: null })
      return
    }
    if (current.step === 'apiKeyValue') {
      if (!current.apiKeyValue.length) {
        setProviderSetup({ ...current, apiKeyValue: '', step: 'apiKeyEnv', error: null })
        return
      }
      setProviderSetup({
        ...current,
        apiKeyEnvVar: managedProviderApiKeyEnvVar({
          providerId: current.providerId,
          preset: current.preset,
          currentEnvVar: current.apiKeyEnvVar,
        }),
        step: current.preset.type === 'custom' ? 'headers' : 'model',
        error: null,
      })
      return
    }
    if (current.step === 'headers') {
      let headers: Record<string, string>
      try {
        headers = parseProviderHeadersInput(current.headersText ?? '')
      } catch (error) {
        required(errorMessage(error))
        return
      }
      setProviderSetup({ ...current, headersText: (current.headersText ?? '').trim(), loading: true, error: null })
      const discovered = await client.discoverProviderModels({
        type: current.preset.type,
        baseUrl: current.baseUrl,
        apiKey: current.apiKeyValue.length > 0
          ? current.apiKeyValue
          : (current.apiKeyEnvVar ? `\${${current.apiKeyEnvVar}}` : undefined),
        headers,
      }).then((result) => result.models).catch(() => [])
      const selection = reconcileProviderModelDiscovery({
        currentModel: current.model,
        discoveredModels: discovered,
        fallbackModels: current.modelSuggestions,
      })
      setProviderSetup({
        ...current,
        headersText: (current.headersText ?? '').trim(),
        loading: false,
        ...selection,
        step: 'model',
        error: null,
      })
      return
    }
    if (current.step === 'model') {
      if (!current.model.trim()) { required('Model name is required.'); return }
      setProviderSetup({ ...current, model: current.model.trim(), step: 'confirm', error: null })
      return
    }
    await saveProviderSetup(current)
  }, [providerSetup, saveProviderSetup, selectProviderPreset])

  const backProviderSetup = useCallback(() => setProviderSetup((current) => {
    if (!current || current.loading || current.validating || current.saving) return current
    const previous: Partial<Record<ProviderSetupStep, ProviderSetupStep>> = {
      confirm: 'model',
      model: current.preset?.type === 'ollama'
        ? 'baseUrl'
        : current.preset?.type === 'custom'
          ? 'headers'
          : 'apiKeyValue',
      headers: current.apiKeyValue.length > 0 ? 'apiKeyValue' : 'apiKeyEnv',
      apiKeyValue: current.preset?.type === 'custom' ? 'baseUrl' : 'providerId',
      baseUrl: 'providerId',
      apiKeyEnv: 'apiKeyValue',
      providerId: 'preset',
    }
    const step = previous[current.step]
    if (!step || (current.step === 'providerId' && !current.allowPresetSelection)) return null
    return { ...current, step, error: null }
  }), [])

  const confirmProviderDelete = useCallback(async () => {
    const current = providerDelete
    if (!current || current.deleting) return
    setProviderDelete({ ...current, deleting: true, error: null })
    try {
      const updates = buildProviderDeleteUpdate({
        config: { providers: current.configProviders, agent: { defaultProvider: current.defaultProviderId ?? undefined, defaultModel: current.defaultModel ?? undefined } },
        providerId: current.providerId,
        preferredFallbackProviderId: current.nextProviderId,
        preferredFallbackModel: current.nextModel,
      })
      await client.updateConfig(updates)
      const nextProvider = String(updates['agent.defaultProvider'])
      const nextModel = String(updates['agent.defaultModel'])
      setProviderDefaults({ provider: nextProvider, model: nextModel })
      setProviders(await client.providers().catch(() => []))
      if (state.provider === current.providerId) {
        dispatch({ type: 'SET_PROVIDER', provider: nextProvider })
        dispatch({ type: 'SET_MODEL', model: nextModel })
      }
      setProviderDelete(null)
      closeTopDialog()
      showNotice(`Deleted provider ${current.providerId}. Default is now ${nextProvider}/${nextModel}.`)
    } catch (error) { setProviderDelete((value) => value ? { ...value, deleting: false, error: errorMessage(error) } : value) }
  }, [client, closeTopDialog, dispatch, providerDelete, showNotice, state.provider])

  const switchDaemonDefaultAndSession = useCallback(async (
    selection: Extract<ProviderModelPickerItem, { kind: 'model' }>,
  ) => {
    setModelError(null)
    try {
      const result = await client.switchDefaultModel({
        target: `${selection.providerId}/${selection.modelId}`,
      })
      if (!result.ok) {
        setModelError(result.message)
        return
      }
      setProviderDefaults({ provider: result.providerId, model: result.model })
      dispatch({ type: 'SET_PROVIDER', provider: result.providerId })
      dispatch({ type: 'SET_MODEL', model: result.model })
      closeTopDialog()
      showNotice(result.message)
    } catch (error) {
      setModelError(errorMessage(error))
    }
  }, [client, closeTopDialog, dispatch, showNotice])

  const setProviderDefaultFromPicker = useCallback(async (
    providerId: string,
    modelId?: string,
  ) => {
    setModelError(null)
    try {
      const snapshot = await client.config()
      const updates = buildProviderDefaultUpdate({
        config: { providers: snapshot.providers ?? [], agent: snapshot.agent },
        providerId,
        preferredModel: modelId,
      })
      await client.updateConfig(updates)
      const nextProvider = String(updates['agent.defaultProvider'])
      const nextModel = String(updates['agent.defaultModel'])
      setProviderDefaults({ provider: nextProvider, model: nextModel })
      closeTopDialog()
      showNotice(`Daemon default set to ${nextProvider}/${nextModel}.`)
    } catch (error) {
      setModelError(errorMessage(error))
    }
  }, [client, closeTopDialog, showNotice])

  const openProviderDeleteFromPicker = useCallback(async (providerId: string) => {
    setModelError(null)
    try {
      const snapshot = await client.config()
      const configProviders = (snapshot.providers ?? []) as ConfiguredProviderRecord[]
      const target = configProviders.find((provider) => provider.id === providerId)
      if (!target) throw new Error(`Provider ${providerId} is not configured.`)
      const updates = buildProviderDeleteUpdate({
        config: { providers: configProviders, agent: snapshot.agent },
        providerId,
      })
      setProviderDelete({
        providerId,
        providerName: providers.find((provider) => provider.id === providerId)?.name ?? providerId,
        configProviders,
        defaultProviderId: snapshot.agent?.defaultProvider ?? null,
        defaultModel: snapshot.agent?.defaultModel ?? null,
        nextProviderId: String(updates['agent.defaultProvider']),
        nextModel: String(updates['agent.defaultModel']),
        deleting: false,
        error: null,
      })
      openDialog('providerDelete')
    } catch (error) {
      setModelError(errorMessage(error))
    }
  }, [client, openDialog, providers])

  const handleModelPickerSelection = useCallback((
    selection: ProviderModelPickerItem,
    target: 'session' | 'default',
  ) => {
    if (selection.kind === 'model') {
      if (modelPickerPurpose === 'default' || target === 'default') {
        void switchDaemonDefaultAndSession(selection)
        return
      }
      dispatch({ type: 'SET_PROVIDER', provider: selection.providerId })
      dispatch({ type: 'SET_MODEL', model: selection.modelId })
      closeTopDialog()
      showNotice(`Model set to ${selection.providerId}/${selection.modelId}.`)
      return
    }

    switch (selection.action) {
      case 'sync-session-default':
        if (!selection.providerId || !selection.modelId) {
          setModelError('The daemon default model is unavailable.')
          return
        }
        dispatch({ type: 'SET_PROVIDER', provider: selection.providerId })
        dispatch({ type: 'SET_MODEL', model: selection.modelId })
        closeTopDialog()
        showNotice(`Model set to daemon default ${selection.providerId}/${selection.modelId}.`)
        return
      case 'setup':
        void openProviderSetup()
        return
      case 'setup-preset':
        void openProviderSetup(undefined, selection.presetType)
        return
      case 'edit-provider':
        if (!selection.providerId) {
          setModelError('Provider edit action is missing a provider id.')
          return
        }
        void openProviderSetup(selection.providerId)
        return
      case 'set-default-provider':
        if (!selection.providerId) {
          setModelError('Provider default action is missing a provider id.')
          return
        }
        void setProviderDefaultFromPicker(selection.providerId, selection.modelId)
        return
      case 'delete-provider':
        if (!selection.providerId) {
          setModelError('Provider delete action is missing a provider id.')
          return
        }
        void openProviderDeleteFromPicker(selection.providerId)
    }
  }, [
    closeTopDialog,
    dispatch,
    modelPickerPurpose,
    openProviderDeleteFromPicker,
    openProviderSetup,
    setProviderDefaultFromPicker,
    showNotice,
    switchDaemonDefaultAndSession,
  ])

  const copyText = useCallback(async (text: string): Promise<boolean> => {
    const result = runtime?.copy
      ? await runtime.copy(text)
      : await copyToClipboard(text, createNodeClipboardDeps(stdout))
    setStatusError(result.ok ? null : (result.error ?? 'clipboard failed'))
    return result.ok
  }, [runtime, stdout])

  const runCommand = useCallback(async (command: CommandDef, args = '') => {
    setStatusError(null)
    switch (command.id) {
      case 'shell.help':
        setInfo({
          title: 'Help',
          body: 'Enter send · Ctrl+J newline · Ctrl+C cancel/clear; twice exit · Ctrl+D exit · Ctrl+X leader\nF11 plan/todo · F12 agent loop · Ctrl+O latest file · Esc return\n/new fresh session · /clear local transcript · /compact context · /doctor health',
        })
        openDialog('info')
        return
      case 'settings.open': openDialog('settings'); return
      case 'model.pick':
        await runModelCommand(args, {
          providers, currentProvider: state.provider, currentModel: state.model,
          defaultProvider: providerDefaults.provider, defaultModel: providerDefaults.model,
          fetchProviders: () => client.providers(),
          openPicker: openModelPicker,
          select: (provider, model) => { dispatch({ type: 'SET_PROVIDER', provider }); dispatch({ type: 'SET_MODEL', model }) },
          saveDefault: async (provider, model) => { await client.updateConfig({ 'agent.defaultProvider': provider, 'agent.defaultModel': model }); setProviderDefaults({ provider, model }) },
          setError: setStatusError,
          showNotice,
        })
        return
      case 'mode.pick':
        await runModeCommand(args, {
          agents, fallbacks: MODE_FALLBACKS, currentMode: state.mode,
          fetchAgents: () => client.agents(), openPicker: () => openDialog('mode'),
          select: (mode) => dispatch({ type: 'SET_MODE', mode }), setError: setStatusError, showNotice,
        })
        return
      case 'session.new':
        if (state.isStreaming) { setStatusError('Cancel the active run before starting a new session.'); return }
        setRedoSessionId(null)
        dispatch({ type: 'NEW_SESSION' })
        setClearedMessageCount(0)
        return
      case 'session.clear':
        setClearedMessageCount(state.messages.length)
        return
      case 'session.compact':
        if (!state.sessionId) { setStatusError('No active session to compact.'); return }
        if (state.isStreaming) { setStatusError('Cancel or finish the active run before compacting.'); return }
        try {
          const result = await client.compactSession(state.sessionId)
          showNotice(`Compacted context: ${result.originalTokens.toLocaleString()} → ${result.compactedTokens.toLocaleString()} tokens (saved ${result.savedTokens.toLocaleString()}).`)
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      case 'session.resume': openDialog('sessions'); return
      case 'project.pick': {
        await runProjectCommand(args, {
          workspaceRoot: fileRoot, sessionId: state.sessionId,
          currentProjectId: state.projectId, currentProjectName: state.projectName,
          fetchProjects: () => client.projects(), attachSession: (projectId, sessionId) => client.attachSessionToProject(projectId, sessionId),
          cacheProjects: setProjects, openPicker: () => openDialog('projects'),
          select: (project) => dispatch({ type: 'SET_PROJECT', projectId: project?.id ?? null, projectName: project?.name ?? null }),
          setError: setStatusError, showNotice,
        })
        return
      }
      case 'skills.open': {
        const [action = 'installed', ...rest] = args.split(/\s+/).filter(Boolean)
        if (['help', '-h', '--help'].includes(action)) { showNotice(SKILLS_COMMAND_USAGE); return }
        if (action === 'install') {
          const source = rest.join(' ').trim()
          if (!source) { setStatusError('Usage: /skills install <source>'); return }
          try {
            const result = await client.installSkill({ source })
            setSkills(await client.skills({ includeDisabled: true, cwd: fileRoot, workspaceRoot: fileRoot }))
            showNotice(formatSkillInstallResult(result.installed))
          } catch (error) { setStatusError(errorMessage(error)) }
          return
        }
        if (action === 'enable' || action === 'disable') {
          const id = rest.join(' ').trim()
          if (!id) { setStatusError(`Usage: /skills ${action} <id>`); return }
          try {
            const result = await client.setSkillEnabled(id, action === 'enable')
            setSkills(await client.skills({ includeDisabled: true, cwd: fileRoot, workspaceRoot: fileRoot }))
            showNotice(`Skill ${result.enabled ? 'enabled' : 'disabled'}: ${result.id}`)
          } catch (error) { setStatusError(errorMessage(error)) }
          return
        }
        if (action === 'search' || action === 'store') {
          const query = rest.join(' ').trim()
          try {
            const results = await client.searchMarketplaceSkills(query)
            setInfo({ title: 'Skill marketplace', body: results.length > 0 ? results.slice(0, 20).map((result) => `${result.metadata.name} · ${result.source}${result.installed ? ' [installed]' : ''}\n  ${result.metadata.description}`).join('\n') : 'No marketplace skills matched.' })
            openDialog('info')
          } catch (error) { setStatusError(errorMessage(error)) }
          return
        }
        setSkillsQuery(action === 'installed' || action === 'list' || action === 'manage' ? rest.join(' ') : args)
        openDialog('skills')
        return
      }
      case 'memory.open':
        if (['backlog', 'backlogs', 'open-loop', 'open-loops', 'openloops'].includes(args.trim().split(/\s+/, 1)[0]?.toLowerCase() ?? '')) {
          const [, action = 'list', ...rest] = args.trim().split(/\s+/)
          try {
            const snapshot = await client.fileMemory()
            if (['list', 'current', 'info'].includes(action)) { showNotice(formatTuiBacklogSummary(snapshot)); return }
            if (action === 'add') {
              const content = rest.join(' ').trim()
              if (!content) { setStatusError(TUI_FILE_MEMORY_USAGE); return }
              const next = appendUniqueMemoryLine(findMemorySection(snapshot, TUI_OPEN_LOOP_QUEUE_SECTION), formatManualMemoryBacklogLine(content))
              await client.updateFileMemorySection(TUI_OPEN_LOOP_QUEUE_SECTION, next)
              showNotice(`Added open-loop backlog item: ${content}`)
              return
            }
            if (['done', 'resolve', 'remove', 'rm'].includes(action)) {
              const query = rest.join(' ').trim()
              if (!query) { setStatusError(TUI_FILE_MEMORY_USAGE); return }
              const result = removeMatchingMemoryLines(findMemorySection(snapshot, TUI_OPEN_LOOP_QUEUE_SECTION), query)
              if (result.removed.length === 0) { showNotice(`No open-loop backlog item matched: ${query}`); return }
              await client.updateFileMemorySection(TUI_OPEN_LOOP_QUEUE_SECTION, result.remaining)
              showNotice(`Resolved ${result.removed.length} open-loop backlog item${result.removed.length === 1 ? '' : 's'}.`)
              return
            }
            setStatusError(TUI_FILE_MEMORY_USAGE)
          } catch (error) { setStatusError(errorMessage(error)) }
          return
        }
        if (['file', 'files', 'markdown'].includes(args.trim().split(/\s+/, 1)[0]?.toLowerCase() ?? '')) {
          const [, action = 'list', ...rest] = args.trim().split(/\s+/)
          try {
            if (action === 'set') {
              const parsed = splitSectionAndContent(rest)
              if (!parsed) { setStatusError(TUI_FILE_MEMORY_USAGE); return }
              const result = await client.updateFileMemorySection(parsed.section, parsed.content)
              showNotice(result.deleted ? `Cleared markdown memory section: ${parsed.section}` : `Saved markdown memory section: ${result.title}`)
              return
            }
            if (['delete', 'remove', 'rm'].includes(action)) {
              const title = rest.join(' ').trim()
              if (!title) { setStatusError(TUI_FILE_MEMORY_USAGE); return }
              const result = await client.deleteFileMemorySection(title)
              showNotice(result.deleted ? `Deleted markdown memory section: ${title}` : `Markdown memory section was already absent: ${title}`)
              return
            }
            const snapshot = await client.fileMemory()
            if (action === 'today' || action === 'yesterday') {
              showNotice(action === 'today' ? formatTuiDailyMemory('today', snapshot.todayNotePath, snapshot.todayNote) : formatTuiDailyMemory('yesterday', snapshot.yesterdayNotePath, snapshot.yesterdayNote))
              return
            }
            if (action === 'show' && rest.length > 0) {
              const title = rest.join(' ')
              const section = snapshot.sections.find((entry) => entry.title === title)
              showNotice(section ? `## ${section.title}\n${section.content || '(empty)'}` : `Memory section not found: ${title}`)
              return
            }
            showNotice(formatTuiFileMemorySummary(snapshot))
          } catch (error) { setStatusError(errorMessage(error)) }
          return
        }
        if (['lifecycle', 'health', 'maintenance-status'].includes(args.trim().split(/\s+/, 1)[0]?.toLowerCase() ?? '')) {
          try { showNotice(formatTuiMemoryLifecycleSummary(await client.memoryLifecycle())) }
          catch (error) { setStatusError(errorMessage(error)) }
          return
        }
        if (['audit', 'audits'].includes(args.trim().split(/\s+/, 1)[0]?.toLowerCase() ?? '')) {
          const memoryId = args.trim().split(/\s+/)[1]
          try { showNotice(formatTuiMemoryAuditSummary(await client.memoryAudit({ memoryId, limit: 8 }))) }
          catch (error) { setStatusError(errorMessage(error)) }
          return
        }
        if (['maintenance', 'maintain', 'cleanup'].includes(args.trim().split(/\s+/, 1)[0]?.toLowerCase() ?? '')) {
          const apply = args.split(/\s+/).includes('--apply')
          try { showNotice(formatTuiMemoryMaintenanceSummary(await client.runMemoryMaintenance({ dryRun: !apply, reason: apply ? 'Run memory lifecycle maintenance from CLI TUI' : 'Preview memory lifecycle maintenance from CLI TUI' }))) }
          catch (error) { setStatusError(errorMessage(error)) }
          return
        }
        setMemoryInitialQuery(args.trim())
        openDialog('memory')
        if (args.trim()) {
          setMemoryLoading(true)
          setMemoryError(null)
          void client.searchMemory(args.trim(), { type: 'hybrid', limit: 20 })
            .then(setMemoryResults)
            .catch((error) => setMemoryError(errorMessage(error)))
            .finally(() => setMemoryLoading(false))
        }
        return
      case 'usage.open': {
        runUsageCommand(args, {
          days: usageDays, summary: usageSummary, dashboardOpen: dialogs.includes('usage'), dashboardOnTop: dialogs.at(-1) === 'usage',
          setDays: setUsageDays, openDashboard: () => openDialog('usage'), closeDashboard: closeTopDialog,
          setError: setStatusError, showNotice,
        })
        return
      }
      case 'artifacts.open': openDialog('artifacts'); return
      case 'rag.open':
        if (args.trim().toLowerCase() === 'sync') {
          try {
            const result = await client.syncRag()
            showNotice(`RAG sync: ${result.folders} folders · ${result.indexed} indexed · ${result.deleted} deleted · ${result.skipped} skipped${result.errors.length ? ` · ${result.errors.length} errors` : ''}`)
            setRagRefresh((value) => value + 1)
          } catch (error) { setStatusError(errorMessage(error)) }
          return
        }
        if (args.trim().toLowerCase().startsWith('add ')) {
          const [, sourcePath, ...nameParts] = args.trim().split(/\s+/)
          if (!sourcePath) { setStatusError('Usage: /rag add <path> [name]'); return }
          try {
            const absolutePath = resolve(sourcePath)
            const folder = await client.upsertRagFolder({ name: nameParts.join(' ') || basename(absolutePath), path: absolutePath, sourceType: 'git' })
            await client.syncRag()
            showNotice(`RAG source added: ${folder.name}`)
            setRagRefresh((value) => value + 1)
          } catch (error) { setStatusError(errorMessage(error)) }
          return
        }
        setRagInitialQuery(args.trim().replace(/^search\s+/i, '').replace(/^(open|list|sources|current|info)$/i, ''))
        openDialog('rag')
        return
      case 'mcp.open':
        if (['list', 'add', 'search', 'install', 'playwright', 'enable', 'disable', 'trust-manifest', 'remove', 'tools', 'prompts', 'resources', 'complete', 'logging', 'metrics', 'call', 'marketplace', 'help'].includes(args.trim().split(/\s+/, 1)[0] ?? '')) {
          try { showNotice(await runMcpSlashCommand(client, args.trim().split(/\s+/))) }
          catch (error) { setStatusError(errorMessage(error)) }
        } else {
          setMcpInitialQuery(args.trim())
          openDialog('mcp')
        }
        return
      case 'diagnostics.doctor':
        openDialog('info')
        setInfo({ title: 'Doctor', body: 'Running daemon diagnostics…' })
        try { setInfo({ title: 'Doctor', body: await client.healthReport('markdown') }) }
        catch (error) { setInfo({ title: 'Doctor', body: `Diagnostics failed: ${errorMessage(error)}` }) }
        return
      case 'run.skill': {
        const [skillName, ...promptParts] = args.split(/\s+/).filter(Boolean)
        if (!skillName) { setStatusError('Usage: /run <skill> [prompt]'); return }
        if (state.isStreaming) { setStatusError('A run is already active. Press Ctrl+C to cancel it first.'); return }
        const prompt = promptParts.join(' ') || `Execute the "${skillName}" skill`
        await sendMessage(prompt, [], { displayContent: prompt, skillRefs: [{ name: skillName }] })
        return
      }
      case 'run.steer': {
        const message = args.trim()
        if (!message) { setStatusError('Usage: /steer <message>'); return }
        if (!state.sessionId) { setStatusError('No active session yet — send a message first.'); return }
        try {
          const result = await client.steerSession(state.sessionId, message)
          rememberQueuedFollowUp(state.sessionId, result.noteId)
          showNotice(`Steering note queued${result.pendingSteeringNoteCount === undefined ? '' : ` (${result.pendingSteeringNoteCount} pending)`}. Esc to undo.`)
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'run.followup': {
        const [action = '', target = 'latest', ...extra] = args.trim().split(/\s+/).filter(Boolean)
        if (!['cancel', 'undo'].includes(action) || extra.length > 0) {
          setStatusError('Usage: /followup cancel [latest|all|<note-id>]')
          return
        }
        await cancelQueuedFollowUps(
          target === 'latest' || target === 'all' ? { selector: target } : { noteId: target },
        )
        return
      }
      case 'thinking.pick': {
        if (!args) { openDialog('thinking'); return }
        const level = args.toLowerCase()
        if (level === 'current' || level === 'info') { showNotice(`Thinking level: ${state.thinkingLevel}\nReasoning display: ${showThinking ? 'shown' : 'hidden'}`); return }
        if (['show', 'visible', 'hide', 'hidden', 'toggle'].includes(level)) {
          const nextShowThinking = level === 'toggle' ? !showThinking : ['show', 'visible'].includes(level)
          setShowThinking(nextShowThinking)
          showNotice(`Reasoning display ${nextShowThinking ? 'shown' : 'hidden'}.`)
          return
        }
        if (!THINKING_OPTIONS.some(({ value }) => value === level)) { setStatusError('Usage: /thinking <off|low|medium|high|max|show|hide|toggle>'); return }
        dispatch({ type: 'SET_THINKING_LEVEL', thinkingLevel: level as typeof state.thinkingLevel })
        showNotice(`Thinking level: ${level}`)
        return
      }
      case 'maxTokens.pick': {
        const value = args.toLowerCase()
        if (!value) { openDialog('maxTokens'); return }
        if (value === 'current' || value === 'info') { showNotice(`Max output tokens: ${state.maxTokens?.toLocaleString() ?? 'provider default'}`); return }
        if (['off', 'none', 'default', 'clear'].includes(value)) {
          dispatch({ type: 'SET_MAX_TOKENS', maxTokens: null })
          showNotice('Max output tokens: provider default')
          return
        }
        const parsed = Number.parseInt(value, 10)
        if (!Number.isInteger(parsed) || parsed <= 0) { setStatusError('Usage: /max-tokens <positive integer|default>'); return }
        dispatch({ type: 'SET_MAX_TOKENS', maxTokens: parsed })
        showNotice(`Max output tokens: ${parsed.toLocaleString()}`)
        return
      }
      case 'memory.remember': {
        const content = args.trim()
        if (!content) { setStatusError('Usage: /remember <content>'); return }
        try {
          const result = await client.addMemory(content)
          showNotice(`Saved memory ${result.id.slice(0, 8)}.`)
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'approvals.list': {
        const [action, scopeArg] = args.split(/\s+/).filter(Boolean)
        if (action === 'clear') {
          const scope = scopeArg === 'session' || scopeArg === 'always' ? scopeArg : undefined
          try {
            await client.clearRememberedApprovals({ scope })
            showNotice(scope ? `Cleared remembered ${scope}-scope approvals.` : 'Cleared all remembered approvals.')
          } catch (error) { setStatusError(errorMessage(error)) }
          return
        }
        const pending = state.pendingApproval
        try {
          const { decisions } = await client.listRememberedApprovals()
          const lines = pending ? [`Pending: ${pending.toolName} (${pending.state})`, `Request: ${pending.requestId}`, ''] : []
          lines.push(decisions.length > 0
            ? `Remembered approvals (${decisions.length}):\n${decisions.map((decision) => `${decision.approved ? 'allow' : 'deny'} ${decision.tool}: ${decision.pattern} [${decision.scope}]`).join('\n')}`
            : 'No remembered approvals.')
          setInfo({ title: 'Approvals', body: lines.join('\n') })
          openDialog('info')
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'questions.list':
        setInfo({
          title: 'Questions',
          body: state.pendingQuestions.length > 0
            ? state.pendingQuestions.map((question) => `${question.id}: ${question.prompt}${question.choices?.length ? `\n  ${question.choices.join(' · ')}` : ''}`).join('\n\n')
            : 'No pending questions.',
        })
        openDialog('info')
        return
      case 'questions.answer': {
        const [questionId, ...answerParts] = args.split(/\s+/).filter(Boolean)
        const answer = answerParts.join(' ')
        if (!questionId || !answer) { setStatusError('Usage: /answer <question-id> <reply>'); return }
        try { await answerQuestion(questionId, answer) }
        catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'context.show': {
        if (!state.sessionId) { showNotice('Context Map\nSession  none\nNext     /resume or send a prompt'); return }
        try {
          const session = await client.session(state.sessionId)
          showNotice([
            'Context Map',
            `Session  ${state.sessionId}`,
            `Messages ${session.messageCount ?? state.messages.length}`,
            `Tokens   ${(state.usage.input + state.usage.output).toLocaleString()}`,
            `Model    ${state.provider}/${state.model}`,
            `Mode     ${state.mode}`,
            `Autonomy ${state.autonomy}`,
          ].join('\n'))
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'workspace.diff': {
        try {
          const result = runtime?.runShell
            ? await runtime.runShell('git diff --no-ext-diff --', { signal: new AbortController().signal })
            : await runLocalShellCommand('git diff --no-ext-diff --')
          if (result.exitCode !== 0) { setStatusError(result.stderr.trim() || `git diff exited with ${result.exitCode}`); return }
          setInfo({ title: 'Working tree diff', body: result.stdout.trim() || 'No unstaged changes.' })
          openDialog('info')
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'session.manage':
      case 'session.fork': {
        if (command.id === 'session.fork' && args.trim()) { setStatusError('Usage: /fork'); return }
        const activate = async (sessionId: string) => {
          const [session, artifacts] = await Promise.all([client.session(sessionId), client.sessionArtifacts(sessionId).catch(() => [])])
          const workspaceError = sessionWorkspaceLoadError(session, fileRoot)
          if (workspaceError) throw new Error(workspaceError)
          loadSession(session, artifacts)
          setClearedMessageCount(0)
        }
        await runSessionCommand(command.id === 'session.fork' ? 'branch' : args, {
          sessionId: state.sessionId, messageCount: state.messages.length, provider: state.provider, model: state.model,
          isStreaming: state.isStreaming, cwd: process.cwd(), openPicker: () => openDialog('sessions'),
          inspect: (sessionId) => client.session(sessionId), activate,
          branch: (sessionId) => client.branchSession(sessionId), compact: (sessionId) => client.compactSession(sessionId),
          exportSession: (sessionId, format) => format === 'json' ? client.sessionExport(sessionId, 'json') : client.sessionExport(sessionId, 'markdown'),
          writeFile: (path, content) => runtime?.writeFile ? runtime.writeFile(path, content) : writeFile(path, content, 'utf8'),
          deleteSession: (sessionId) => client.deleteSession(sessionId),
          resetSession: () => { dispatch({ type: 'NEW_SESSION' }); setClearedMessageCount(0) },
          setError: setStatusError, showNotice,
        })
        return
      }
      case 'session.rewind': {
        const turns = args ? Number.parseInt(args, 10) : 1
        if (!Number.isInteger(turns) || turns <= 0) { setStatusError('Usage: /rewind [positive turn count]'); return }
        if (!state.sessionId) { setStatusError('No active session to rewind.'); return }
        if (state.isStreaming || state.pendingApproval) { setStatusError('Finish the active run or approval before rewinding.'); return }
        try {
          const sourceId = state.sessionId
          const source = await client.session(sourceId)
          const target = findRewindTarget(source.events, turns)
          if (!target) { setStatusError('No user messages found in the current session yet.'); return }
          const branch = await client.branchSession(sourceId, { fromEventIndex: target.fromEventIndex })
          const [session, artifacts] = await Promise.all([
            client.session(branch.branchId),
            client.sessionArtifacts(branch.branchId).catch(() => []),
          ])
          loadSession(session, artifacts)
          setClearedMessageCount(0)
          setRedoSessionId(sourceId)
          showNotice(`Rewound ${target.turns} user turn${target.turns === 1 ? '' : 's'} into branch ${branch.branchId}.`)
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'session.redo': {
        if (args.trim()) { setStatusError('Usage: /redo'); return }
        if (!redoSessionId) { setStatusError('Nothing to redo. Use /undo first.'); return }
        if (state.isStreaming || state.pendingApproval) { setStatusError('Finish the active run or approval before redoing.'); return }
        try {
          const [session, artifacts] = await Promise.all([
            client.session(redoSessionId),
            client.sessionArtifacts(redoSessionId).catch(() => []),
          ])
          loadSession(session, artifacts)
          setClearedMessageCount(0)
          showNotice(`Restored session ${redoSessionId}.`)
          setRedoSessionId(null)
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'session.editLast': {
        if (args.trim()) { setStatusError('Usage: /edit-last'); return }
        if (!state.sessionId) { setStatusError('No active session to edit.'); return }
        if (state.isStreaming || state.pendingApproval) { setStatusError('Finish the active run or approval before editing a prompt.'); return }
        try {
          const sourceId = state.sessionId
          const source = await client.session(sourceId)
          const target = findRewindTarget(source.events, 1)
          const previousPrompt = source.events.slice().reverse().find((event) => event.type === 'user_message')
          if (!target || !previousPrompt || previousPrompt.type !== 'user_message') { setStatusError('No previous user prompt is available to edit.'); return }
          const branch = await client.branchSession(sourceId, { fromEventIndex: target.fromEventIndex })
          const [session, artifacts] = await Promise.all([
            client.session(branch.branchId),
            client.sessionArtifacts(branch.branchId).catch(() => []),
          ])
          loadSession(session, artifacts)
          setClearedMessageCount(0)
          setRedoSessionId(sourceId)
          replaceComposerValue(previousPrompt.content)
          showNotice(`Editing the previous prompt in branch ${branch.branchId}. The original session is preserved.`)
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'session.rename': {
        const title = args.trim()
        if (!title) { setStatusError('Usage: /rename <title>'); return }
        if (!state.sessionId) { setStatusError('No active session to rename.'); return }
        try {
          const updated = await client.updateSession(state.sessionId, { title })
          showNotice(`Session renamed: ${updated.title}`)
          setSessionRefresh((value) => value + 1)
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'session.share': {
        if (!state.sessionId) { setStatusError('No active session to share.'); return }
        const parts = args.trim().toLowerCase().split(/\s+/).filter(Boolean)
        const mode = parts[0] || 'knowledge'
        if (mode !== 'knowledge' && mode !== 'public') { setStatusError('Usage: /share [knowledge|public confirm]'); return }
        if (mode === 'public' && parts[1] !== 'confirm') {
          setStatusError('Public sharing exposes the session through a temporary URL. Use /share public confirm.')
          return
        }
        try {
          const result = await client.shareSession(state.sessionId, mode)
          if (!result.shared) { setStatusError(result.reason ?? `Session could not be shared in ${mode} mode.`); return }
          showNotice(mode === 'public'
            ? `Public session URL: ${result.shareUrl ?? '(unavailable)'}${result.expiresAt ? `\nExpires: ${result.expiresAt}` : ''}`
            : `Session saved to shared knowledge${result.path ? `: ${result.path}` : '.'}`)
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'model.capability': {
        const parts = args.split(/\s+/).filter(Boolean)
        const usage = `Usage: /capability [current] | /capability <${TOGGLEABLE_MODEL_CAPABILITIES.join('|')}> <on|off> [provider/model]`
        try {
          const configSnapshot = await client.config()
          if (parts.length === 0 || ['current', 'info'].includes(parts[0]!)) {
            const views = describeModelCapabilities(configSnapshot.providers, state.provider, state.model)
            if (!views) { setStatusError(`Provider "${state.provider}" not found in daemon config.`); return }
            setInfo({
              title: `Capabilities · ${state.provider}/${state.model}`,
              body: views.map(({ capability, enabled, source }) => `${capability}: ${enabled ? 'on' : 'off'} (${source})`).join('\n'),
            })
            openDialog('info')
            return
          }
          const capability = resolveCapabilityName(parts[0] ?? '')
          const enabled = parseCapabilityToggle(parts[1] ?? '')
          if (!capability || enabled === null) { setStatusError(usage); return }
          const target = parts.slice(2).join(' ')
          const separator = target.indexOf('/')
          const targetProvider = target && separator > 0 ? target.slice(0, separator) : state.provider
          const targetModel = target && separator > 0 ? target.slice(separator + 1) : (target || state.model)
          const result = applyModelCapabilityOverride(configSnapshot.providers, targetProvider, targetModel, capability, enabled)
          if (!result) { setStatusError(`Provider "${targetProvider}" not found in daemon config.`); return }
          if (result.changed) await client.updateConfig({ providers: result.providers })
          showNotice(`${capability} is ${enabled ? 'on' : 'off'} for ${targetProvider}/${targetModel}${result.changed ? ' (saved)' : ''}.`)
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'diagnostics.tps': {
        const action = args.toLowerCase()
        if (action === 'reset') {
          resetTokenSpeedStats()
          showNotice('TPS stats reset.')
          return
        }
        if (action && !['current', 'info'].includes(action)) { setStatusError('Usage: /tps [current|reset]'); return }
        setInfo({ title: 'Token speed', body: formatTokenSpeedStats(getTokenSpeedStats()) })
        openDialog('info')
        return
      }
      case 'conversation.details':
        setShowToolDetails((current) => {
          showNotice(`Tool execution details ${current ? 'hidden' : 'shown'}.`)
          return !current
        })
        return
      case 'commands.manage': {
        try {
          const commandArgs = splitSlashCommandInput(`/commands ${args}`).slice(1)
          const result = await runDaemonAdminSlashCommand(client, '/commands', commandArgs)
          showNotice(result)
          const action = commandArgs[0]?.toLowerCase() ?? 'list'
          if (['create', 'delete', 'remove'].includes(action) && !result.startsWith('Confirmation required')) {
            setUserCommands(await client.userCommands())
          }
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'session.recap': {
        const action = args.toLowerCase() || 'status'
        if (action === 'status' || action === 'current' || action === 'info') {
          showNotice(`Recap is ${recapEnabled ? 'on' : 'off'}. Usage: /recap <on|off>`)
          return
        }
        if (!['on', 'off'].includes(action)) { setStatusError('Usage: /recap <on|off|status>'); return }
        const enabled = action === 'on'
        setRecapEnabled(enabled)
        try { await recordRecapPreference(enabled) } catch { /* session setting still applies */ }
        showNotice(`Recap ${enabled ? 'enabled' : 'disabled'}.`)
        return
      }
      case 'files.open':
        setFileDirectory(fileRoot)
        setSelectedFilePaths([])
        openDialog('files')
        return
      case 'files.image': {
        if (state.isStreaming) { setStatusError('Queue text follow-ups while the run is active; attach images after it finishes.'); return }
        const value = args.trim()
        if (!value) { openDialog('files'); return }
        try {
          const fromClipboard = value === 'paste' || value === 'clipboard'
          const imagePath = fromClipboard
            ? await (runtime?.pasteImage ? runtime.pasteImage() : pasteClipboardPng())
            : resolve(value.replace(/^['"]|['"]$/g, ''))
          if (!['.png', '.jpg', '.jpeg', '.gif', '.webp'].includes(extname(imagePath).toLowerCase())) {
            setStatusError('Supported image formats: PNG, JPEG, GIF, and WebP.')
            return
          }
          if (!fromClipboard) await access(imagePath)
          const reference = /\s/.test(imagePath) ? `@"${imagePath}"` : `@${imagePath}`
          replaceComposerValue(`${inputValue}${inputValue && !/\s$/.test(inputValue) ? ' ' : ''}${reference} `)
          showNotice(`Image attached: ${imagePath}`)
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'integration.acp': {
        const action = args.toLowerCase() || 'help'
        let body: string
        if (['help', '-h', '--help'].includes(action)) body = ACP_COMMAND_USAGE
        else if (action === 'config') body = buildAcpConfigSnippet(process.env.SEPILOTD_URL)
        else if (action === 'opencode') body = buildAcpOpencodeGuide()
        else if (action === 'codex') body = buildAcpCodexGuide()
        else if (action === 'zed') body = buildAcpZedHint(process.env.SEPILOTD_URL)
        else if (['status', 'current', 'info'].includes(action)) {
          try {
            const health = await client.health()
            body = `ACP status\nDaemon: ${health.status} (${health.version})\nTransport: JSON-RPC 2.0 over stdio Content-Length framing\nMethods: initialize, session/new, session/prompt, session/cancel`
          } catch (error) { setStatusError(errorMessage(error)); return }
        } else { setStatusError(`Unknown /acp action: ${action}`); return }
        setInfo({ title: 'ACP integration', body })
        openDialog('info')
        return
      }
      case 'integration.a2a': {
        const action = args.toLowerCase() || 'help'
        let body: string
        if (['help', '-h', '--help'].includes(action)) body = A2A_COMMAND_USAGE
        else if (['card', 'config'].includes(action)) body = buildA2aGuide(process.env.SEPILOTD_URL)
        else if (action === 'send') body = 'Use daemon tool `a2a.send`. Required: agentCardUrl, message. Optional: headers, timeoutMs.'
        else if (['status', 'current', 'info'].includes(action)) {
          try {
            const health = await client.health()
            body = `A2A status\nDaemon: ${health.status} (${health.version})\n${buildA2aGuide(process.env.SEPILOTD_URL)}`
          } catch (error) { setStatusError(errorMessage(error)); return }
        } else { setStatusError(`Unknown /a2a action: ${action}`); return }
        setInfo({ title: 'A2A integration', body })
        openDialog('info')
        return
      }
      case 'hooks.manage': {
        const parts = args.split(/\s+/).filter(Boolean)
        const action = parts[0]?.toLowerCase() ?? 'list'
        try {
          if (['list', 'ls', 'current', 'info'].includes(action)) {
            const hooks = await client.outboundWebhooks()
            setInfo({ title: 'Outbound hooks', body: hooks.length > 0 ? hooks.map(formatTuiHookSummary).join('\n') : `No outbound hooks configured.\n${buildTuiHooksUsage()}` })
            openDialog('info')
            return
          }
          if (action === 'add') {
            const rest = parts.slice(1)
            const url = rest.find((entry) => !entry.startsWith('--'))
            if (!url) { setStatusError(buildTuiHooksUsage()); return }
            const events: string[] = []
            const headers: string[] = []
            let secret: string | undefined
            let disabled = false
            for (let index = 0; index < rest.length;) {
              const value = rest[index]!
              if (value === url) { index += 1; continue }
              if (value === '--disabled') { disabled = true; index += 1; continue }
              if (value === '--event' || value.startsWith('--event=')) {
                const parsed = parseTuiHookValueFlag(rest, index, '--event')
                if (parsed.value) events.push(parsed.value)
                index = parsed.nextIndex
                continue
              }
              if (value === '--header' || value.startsWith('--header=')) {
                const parsed = parseTuiHookValueFlag(rest, index, '--header')
                if (parsed.value) headers.push(parsed.value)
                index = parsed.nextIndex
                continue
              }
              if (value === '--secret' || value.startsWith('--secret=')) {
                const parsed = parseTuiHookValueFlag(rest, index, '--secret')
                secret = parsed.value ?? undefined
                index = parsed.nextIndex
                continue
              }
              throw new Error(`Unknown /hooks add flag: ${value}`)
            }
            await client.upsertOutboundWebhook({ enabled: !disabled, url, events: parseTuiHookEvents(events), headers: parseTuiHookHeaders(headers), ...(secret ? { secret } : {}) })
            showNotice(`Outbound hook saved: ${url}`)
            return
          }
          if (['remove', 'delete', 'rm'].includes(action)) {
            if (!parts[1]) { setStatusError('Usage: /hooks remove <id>'); return }
            await client.deleteOutboundWebhook(parts[1])
            showNotice(`Outbound hook removed: ${parts[1]}`)
            return
          }
          if (action === 'enable' || action === 'disable') {
            if (!parts[1]) { setStatusError(`Usage: /hooks ${action} <id>`); return }
            await client.setOutboundWebhookEnabled(parts[1], action === 'enable')
            showNotice(`Outbound hook ${action}d: ${parts[1]}`)
            return
          }
          if (action === 'deliveries') {
            const limitIndex = parts.findIndex((part) => part === '--limit' || part.startsWith('--limit='))
            const limit = limitIndex >= 0 ? parseTuiHookLimit(parseTuiHookValueFlag(parts, limitIndex, '--limit').value ?? undefined) : undefined
            const statusPart = parts.find((part) => part.startsWith('--status='))?.slice('--status='.length)
            const id = parts.slice(1).find((part) => !part.startsWith('--') && part !== parts[limitIndex + 1])
            const deliveries = await client.outboundWebhookDeliveries({ id, status: statusPart === 'success' || statusPart === 'error' ? statusPart : undefined, limit: limit ?? 20 })
            setInfo({ title: 'Hook deliveries', body: deliveries.length > 0 ? deliveries.map(formatTuiHookDelivery).join('\n') : 'No outbound hook deliveries recorded.' })
            openDialog('info')
            return
          }
          if (action === 'dead-letters') {
            const statePart = parts.find((part) => part.startsWith('--state='))?.slice('--state='.length)
            const stateFilter = statePart === 'acknowledged' || statePart === 'all' ? statePart : 'open'
            const id = parts.slice(1).find((part) => !part.startsWith('--'))
            const deadLetters = await client.outboundWebhookDeadLetters({ id, state: stateFilter, limit: 20 })
            setInfo({ title: 'Hook dead letters', body: deadLetters.length > 0 ? deadLetters.map(formatTuiHookDeadLetter).join('\n') : 'No outbound hook dead letters.' })
            openDialog('info')
            return
          }
          if (action === 'replay') {
            const deliveryId = parts.slice(1).find((part) => !part.startsWith('--'))
            if (!deliveryId) { setStatusError('Usage: /hooks replay <deliveryId> [--force]'); return }
            const replay = await client.replayOutboundWebhookDelivery(deliveryId, parts.includes('--force') ? { force: true } : undefined)
            showNotice(`Outbound hook replay created.\n${formatTuiHookDelivery(replay)}`)
            return
          }
          if (action === 'replay-failed') {
            const id = parts.slice(1).find((part) => !part.startsWith('--'))
            const replays = await client.replayOutboundWebhookDeadLetters({ id, state: 'open', limit: 20, force: parts.includes('--force') })
            showNotice(replays.length > 0 ? `Replayed ${replays.length} outbound hook dead letter${replays.length === 1 ? '' : 's'}.` : 'No outbound hook dead letters were replayed.')
            return
          }
          if (action === 'ack') {
            const rootDeliveryId = parts[1]
            if (!rootDeliveryId) { setStatusError('Usage: /hooks ack <rootDeliveryId> [note...]'); return }
            const note = parts.slice(2).join(' ').trim()
            const deadLetter = await client.acknowledgeOutboundWebhookDeadLetter(rootDeliveryId, note ? { note } : undefined)
            showNotice(`Outbound hook dead letter acknowledged.\n${formatTuiHookDeadLetter(deadLetter)}`)
            return
          }
          setStatusError(buildTuiHooksUsage())
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'containers.list': {
        try {
          if (!(await dockerAvailable())) { showNotice('Docker is not installed or its daemon is not running.'); return }
          const containers = await listManagedContainers()
          showNotice(containers.length > 0
            ? [`Sepilot-managed containers (${containers.length}):`, ...containers.map((container) => `- ${container.name} [${container.state}] ${container.image}${container.purpose ? ` — ${container.purpose}` : ''}`)].join('\n')
            : 'No sepilot-managed containers.')
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'swarm.manage': {
        const [subcommand = 'help', ...rest] = args.split(/\s+/).filter(Boolean)
        const sub = subcommand.toLowerCase()
        const help = [
          '/swarm run <goal> · start supervised run',
          '/swarm run! <goal> · start user-driven run',
          '/swarm list · active runs',
          '/swarm history [n] · recent runs',
          '/swarm status|agents|logs|kill [runId]',
        ].join('\n')
        try {
          if (['help', '?'].includes(sub)) { setInfo({ title: 'Swarm', body: help }); openDialog('info'); return }
          if (sub === 'list' || sub === 'history') {
            const runs = sub === 'list' ? await swarmClient.list() : await swarmClient.history(Number(rest[0]) || 20)
            setInfo({ title: sub === 'list' ? 'Active swarm runs' : 'Swarm history', body: runs.length > 0 ? runs.map((run) => `${run.id}  ${run.status}  ${run.goal}`).join('\n') : '(no swarm runs)' })
            openDialog('info')
            return
          }
          if (sub === 'run' || sub === 'run!') {
            const goal = rest.join(' ').trim()
            if (!goal) { setStatusError('Usage: /swarm run <goal>'); return }
            const { runId } = await swarmClient.createRun({ goal, cwd: process.cwd(), warmPool: ['claude'], ...(sub === 'run!' ? { noSupervisor: true } : {}) })
            setLastSwarmRunId(runId)
            showNotice(`Swarm started ${runId}. Follow-up commands may omit the run id.`)
            return
          }
          const id = rest[0] ?? lastSwarmRunId
          if (!id) { setStatusError(`Usage: /swarm ${sub} <runId>`); return }
          setLastSwarmRunId(id)
          if (sub === 'status' || sub === 'agents') {
            const run = await swarmClient.get(id)
            const body = sub === 'status'
              ? `id: ${run.id}\nstatus: ${run.status}\ngoal: ${run.goal}\nactive: ${run.activeHandle ?? '-'}\nagents: ${run.agents.length}`
              : run.agents.map((agent) => `${run.activeHandle === agent.handle ? '●' : ' '} ${agent.handle}  ${agent.agent}  ${agent.status}  ${agent.role ?? ''}`.trimEnd()).join('\n') || '(no agents)'
            setInfo({ title: `Swarm ${sub}`, body })
            openDialog('info')
            return
          }
          if (sub === 'logs') {
            const events = await swarmClient.events(id)
            const limit = Number(rest[1]) || 20
            setInfo({ title: 'Swarm logs', body: events.slice(-limit).map((event) => `${new Date(event.ts).toISOString().slice(11, 19)} [${event.type}]`).join('\n') || '(no events)' })
            openDialog('info')
            return
          }
          if (sub === 'kill') {
            await swarmClient.cancel(id)
            if (lastSwarmRunId === id) setLastSwarmRunId(null)
            showNotice(`Cancelled swarm ${id}.`)
            return
          }
          if (sub === 'attach') {
            const run = await swarmClient.get(id)
            const requested = rest[1]
            const agent = requested
              ? run.agents.find((entry) => entry.handle === requested || entry.role === requested)
              : (run.agents.find((entry) => entry.handle === run.activeHandle) ?? run.agents[0])
            if (!agent) { setStatusError('No swarm agent is available to attach.'); return }
            setSwarmAttachTarget({ runId: run.id, handle: agent.handle, agent: agent.agent, role: agent.role, tmuxSessionName: agent.tmuxSessionName })
            openDialog('swarmAttach')
            return
          }
          setStatusError(`Unknown /swarm subcommand: ${sub}\n${help}`)
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'copy.last': {
        const last = [...state.messages].reverse().find(({ role }) => role === 'assistant')
        if (!last) { setStatusError('Nothing from the assistant to copy yet.'); return }
        if (await copyText(last.content)) showNotice('Copied the last assistant reply.')
        return
      }
      case 'copy.all':
        if (await copyText(formatConversationForCopy(state.messages, state.sessionId))) showNotice('Copied the conversation.')
        return
      case 'composer.editor':
        try {
          const edited = runtime?.editText
            ? await runtime.editText(inputValue)
            : await editTextInExternalEditor(inputValue, {
                setRawMode: typeof process.stdin.setRawMode === 'function'
                  ? (enabled) => process.stdin.setRawMode(enabled)
                  : undefined,
              })
          replaceComposerValue(edited.replace(/\s+$/, ''))
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      case 'shell.exit': exit(); return
      case 'autonomy.pick':
        if (args === 'current' || args === 'info') { showNotice(`Current autonomy: ${state.autonomy}`); return }
        if (args) {
          if (!AUTONOMY_OPTIONS.some(({ value }) => value === args)) { setStatusError('Usage: /autonomy <readonly|accept-edits|workspace-write|supervised|autonomous>'); return }
          dispatch({ type: 'SET_AUTONOMY', autonomy: args as typeof state.autonomy })
          showNotice(`Autonomy: ${args}`)
          return
        }
        setAutonomyPurpose('session'); openDialog('autonomy'); return
      case 'theme.pick': {
        if (!args) { openDialog('theme'); return }
        if (args === 'current' || args === 'info') { showNotice(`Theme: ${themeOptions.find(({ id }) => id === themeId)?.label ?? themeId}`); return }
        const nextTheme = resolveThemeId(args, themeId)
        if (!nextTheme) { setStatusError(`Unknown theme: ${args}`); return }
        setActiveTheme(nextTheme)
        setThemeId(nextTheme)
        showNotice(`Theme: ${themeOptions.find(({ id }) => id === nextTheme)?.label ?? nextTheme}`)
        return
      }
      case 'providers.open': {
        const [action, value] = args.split(/\s+/).filter(Boolean)
        if (!action) { openModelPicker('session'); return }
        if (action === 'setup' || action === 'add' || action === 'config') { await openProviderSetup(undefined, value); return }
        if (action === 'edit') {
          if (!value) { setStatusError('Usage: /provider edit <id>'); return }
          await openProviderSetup(value)
          return
        }
        if (action === 'default') {
          if (!value) { openModelPicker('default'); return }
          try {
            const snapshot = await client.config()
            const updates = buildProviderDefaultUpdate({ config: { providers: snapshot.providers ?? [], agent: snapshot.agent }, providerId: value })
            await client.updateConfig(updates)
            const nextProvider = String(updates['agent.defaultProvider'])
            const nextModel = String(updates['agent.defaultModel'])
            setProviderDefaults({ provider: nextProvider, model: nextModel })
            showNotice(`Daemon default set to ${nextProvider}/${nextModel}.`)
          } catch (error) { setStatusError(errorMessage(error)) }
          return
        }
        if (action === 'delete') {
          if (!value) { setStatusError('Usage: /provider delete <id>'); return }
          try {
            const snapshot = await client.config()
            const configProviders = (snapshot.providers ?? []) as ConfiguredProviderRecord[]
            const target = configProviders.find((provider) => provider.id === value)
            if (!target) throw new Error(`Provider ${value} is not configured.`)
            const updates = buildProviderDeleteUpdate({ config: { providers: configProviders, agent: snapshot.agent }, providerId: value })
            setProviderDelete({
              providerId: value,
              providerName: providers.find((provider) => provider.id === value)?.name ?? value,
              configProviders,
              defaultProviderId: snapshot.agent?.defaultProvider ?? null,
              defaultModel: snapshot.agent?.defaultModel ?? null,
              nextProviderId: String(updates['agent.defaultProvider']),
              nextModel: String(updates['agent.defaultModel']),
              deleting: false,
              error: null,
            })
            openDialog('providerDelete')
          } catch (error) { setStatusError(errorMessage(error)) }
          return
        }
        const match = providers.find((provider) => provider.id === action)
        if (!match) { setStatusError(`Unknown provider: ${action}`); return }
        dispatch({ type: 'SET_PROVIDER', provider: match.id })
        dispatch({ type: 'SET_MODEL', model: match.models[0]?.id ?? state.model })
        showNotice(`Provider set to ${match.id}.`)
        return
      }
      case 'providers.inspect': {
        try {
          const available = await client.providers()
          setProviders(available)
          setInfo({
            title: 'Providers',
            body: available.length > 0
              ? available.map((provider) => `${provider.name} (${provider.id})${provider.id === state.provider ? ' [current]' : ''}\n  health: ${provider.health.status}\n  models: ${provider.models.map((model) => model.id).join(', ') || 'none'}`).join('\n\n')
              : 'No providers configured.',
          })
          openDialog('info')
        } catch (error) { setStatusError(errorMessage(error)) }
        return
      }
      case 'agent.planner':
        openDialog('planTodo')
        return
      case 'agent.loop':
        openDialog('agentLoop')
        return
      case 'agent.rollbacks':
        setInfo({ title: 'Edit rollback checkpoints', body: (state.editRollbacks ?? []).length > 0 ? JSON.stringify(state.editRollbacks, null, 2) : 'No edit rollback checkpoints for this session.' })
        openDialog('info')
        return
      case 'agent.debate':
        setInfo({ title: 'Debate rounds', body: (state.debateRounds ?? []).length > 0 ? JSON.stringify(state.debateRounds, null, 2) : 'No debate rounds for this session.' })
        openDialog('info')
        return
      default:
        setStatusError(`No inline handler is registered for ${command.id}.`)
    }
  }, [answerQuestion, cancelQueuedFollowUps, client, copyText, dispatch, exit, fileRoot, getTokenSpeedStats, inputValue, loadSession, openDialog, openModelPicker, recapEnabled, rememberQueuedFollowUp, replaceComposerValue, resetTokenSpeedStats, runtime, sendMessage, showNotice, state.autonomy, state.debateRounds, state.editRollbacks, state.isStreaming, state.messages, state.mode, state.model, state.pendingApproval, state.pendingQuestions, state.plannerWorkingMemory, state.provider, state.sessionId, state.thinkingLevel, state.usage.input, state.usage.output, themeId])

  const submitInput = useCallback(async () => {
    const text = composeHangul(replayInlineInputControls(inputValueRef.current)).normalize('NFC').trim()
    if (!text) return
    const [alias, ...rest] = text.split(/\s+/)
    const command = text.startsWith('/') ? findByAlias(alias.toLowerCase()) : undefined
    appendHistory(text)
    historyIndexRef.current = null
    setHistoryDraft('')
    replaceComposerValue('')
    if (text === '!!' || text.startsWith('!')) {
      const rerunLastShellCommand = text === '!!'
      const shellCommand = rerunLastShellCommand
        ? lastShellCommand.current?.trim()
        : text.slice(1).trim()
      if (!shellCommand) {
        setStatusError(rerunLastShellCommand ? 'No local shell command to rerun.' : 'Usage: !<command>')
        return
      }
      const gate = resolveLocalShellGate({ autonomy: state.autonomy })
      if (gate === 'block') {
        setStatusError("Local shell is disabled under autonomy 'readonly'.")
        return
      }
      lastShellCommand.current = shellCommand
      const aborter = new AbortController()
      shellAborter.current = aborter
      showNotice(`Running local shell…\n$ ${shellCommand}`)
      try {
        const result = runtime?.runShell
          ? await runtime.runShell(shellCommand, { signal: aborter.signal })
          : await runLocalShellCommand(shellCommand, { signal: aborter.signal })
        showNotice(buildLocalShellTranscript(result))
        const sessionId = state.sessionId ?? crypto.randomUUID()
        try {
          const recorded = await client.recordLocalShellTurn(sessionId, {
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
          showNotice(
            `Local shell output was shown but not saved to session context: ${errorMessage(error)}`,
          )
        }
      } catch (error) {
        setStatusError(errorMessage(error))
      } finally {
        if (shellAborter.current === aborter) shellAborter.current = null
      }
      return
    }
    if (text.startsWith('/')) {
      if (command) await runCommand(command, rest.join(' '))
      else await sendMessage(text)
      return
    }
    if (state.isStreaming) {
      if (!state.sessionId) {
        setStatusError('The active run has no session id yet. Wait a moment or press Ctrl+C to cancel it.')
        replaceComposerValue(text)
        return
      }
      try {
        const result = await client.steerSession(state.sessionId, text)
        setRedoSessionId(null)
        rememberQueuedFollowUp(state.sessionId, result.noteId)
        showNotice(`Follow-up queued for the active run${result.pendingSteeringNoteCount === undefined ? '' : ` (${result.pendingSteeringNoteCount} pending)`}. Esc to undo.`)
      } catch (error) {
        setStatusError(errorMessage(error))
        replaceComposerValue(text)
      }
      return
    }
    const attachments = await resolveComposerAttachments(text)
    const sent = attachments.length > 0 ? await sendMessage(text, attachments) : await sendMessage(text)
    if (sent) setRedoSessionId(null)
  }, [appendHistory, client, dispatch, inputValue, rememberQueuedFollowUp, replaceComposerValue, runCommand, runtime, sendMessage, showNotice, state.autonomy, state.isStreaming, state.model, state.provider, state.sessionId])

  const pendingQuestion = state.pendingQuestions[0] ?? null
  const dialogOpen = dialogs.length > 0 || Boolean(state.pendingApproval) || Boolean(pendingQuestion)
  const togglePlanTodo = useCallback(() => {
    if (!state.pendingApproval && !pendingQuestion) toggleInspectionDialog('planTodo')
  }, [pendingQuestion, state.pendingApproval, toggleInspectionDialog])
  const toggleAgentLoop = useCallback(() => {
    if (!state.pendingApproval && !pendingQuestion) toggleInspectionDialog('agentLoop')
  }, [pendingQuestion, state.pendingApproval, toggleInspectionDialog])
  useFunctionKeys({ onF11: togglePlanTodo, onF12: toggleAgentLoop })
  const customCommandDefs = useMemo<CommandDef[]>(() => userCommands
    .filter((command) => !findByAlias(`/${command.id}`))
    .map((command) => ({
      id: `custom:${command.id}`,
      title: `${command.description}${command.args === 'required' ? ' · args required' : command.args === 'optional' ? ' · args optional' : ''}`,
      category: 'run', aliases: [`/${command.id}`], keys: [], surface: 'secondary', run: () => {},
    })), [userCommands])
  const historyNavigationActive = historyIndexRef.current !== null
  const suggestions = !historyNavigationActive && inputValue.startsWith('/')
    ? [...matchCommands(inputValue), ...customCommandDefs.filter((command) => inputValue === '/' || command.aliases[0]!.toLowerCase().startsWith(inputValue.toLowerCase()))]
    : []
  const selectedSuggestionIndex = Math.min(suggestionIndex, Math.max(0, suggestions.length - 1))
  const suggestionWindowStart = Math.max(0, selectedSuggestionIndex - 7)
  const visibleSuggestions = suggestions.slice(suggestionWindowStart, suggestionWindowStart + 8)
  const completeSelectedSuggestion = useCallback(() => {
    const suggestion = suggestions[selectedSuggestionIndex]
    const alias = suggestion?.aliases[0]
    if (!alias) return false
    replaceComposerValue(`${alias} `)
    setSuggestionIndex(0)
    return true
  }, [replaceComposerValue, selectedSuggestionIndex, suggestions])
  const activeAttachmentReference = useMemo(
    () => historyNavigationActive || inputValue.startsWith('/') || inputValue.startsWith('!')
      ? null
      : findActiveAttachmentReference(inputValue),
    [historyNavigationActive, inputValue],
  )
  const activeSkillReference = useMemo(
    () => historyNavigationActive || inputValue.startsWith('/') || inputValue.startsWith('!')
      ? null
      : findActiveSkillReference(inputValue),
    [historyNavigationActive, inputValue],
  )
  const skillSuggestions = useMemo(
    () => activeSkillReference
      ? filterSkillAutocompleteCandidates(skills, activeSkillReference.query)
      : [],
    [activeSkillReference, skills],
  )
  const selectedSkillIndex = Math.min(skillSuggestionIndex, Math.max(0, skillSuggestions.length - 1))
  const selectedAttachmentIndex = Math.min(
    attachmentSuggestionIndex,
    Math.max(0, attachmentSuggestions.length - 1),
  )
  const completeSelectedAttachment = useCallback(() => {
    if (!activeAttachmentReference) return false
    const selected = attachmentSuggestions[selectedAttachmentIndex]
    if (!selected) return false
    replaceComposerValue(applyAttachmentCompletion(
      inputValueRef.current,
      activeAttachmentReference,
      selected.path,
    ))
    return true
  }, [activeAttachmentReference, attachmentSuggestions, replaceComposerValue, selectedAttachmentIndex])
  const completeSelectedSkill = useCallback(() => {
    if (!activeSkillReference) return false
    const selected = skillSuggestions[selectedSkillIndex]
    if (!selected) return false
    replaceComposerValue(applySkillCompletion(
      inputValueRef.current,
      activeSkillReference,
      selected.id,
    ))
    return true
  }, [activeSkillReference, replaceComposerValue, selectedSkillIndex, skillSuggestions])

  useEffect(() => {
    if (!activeAttachmentReference) {
      setAttachmentSuggestions([])
      return
    }
    let active = true
    const timeout = setTimeout(() => {
      const load = runtime?.attachmentCandidates
        ? runtime.attachmentCandidates(activeAttachmentReference.path)
        : resolveAttachmentCandidates(activeAttachmentReference.path)
      void load
        .then((items) => { if (active) setAttachmentSuggestions(items) })
        .catch(() => { if (active) setAttachmentSuggestions([]) })
    }, 80)
    return () => { active = false; clearTimeout(timeout) }
  }, [activeAttachmentReference, runtime])

  useEffect(() => {
    if (!activeSkillReference || skills.length > 0) return
    let active = true
    void client.skills({ includeDisabled: true, cwd: fileRoot, workspaceRoot: fileRoot })
      .then((items) => { if (active) setSkills(items) })
      .catch(() => { /* The explicit skill manager exposes retryable errors. */ })
    return () => { active = false }
  }, [activeSkillReference, client, fileRoot, skills.length])

  useEffect(() => { setSuggestionIndex(0) }, [inputValue])
  useEffect(() => { setAttachmentSuggestionIndex(0) }, [activeAttachmentReference?.path])
  useEffect(() => { setSkillSuggestionIndex(0) }, [activeSkillReference?.query])

  useInput((input, key) => {
    if (key.escape && !dialogOpen && inputValueRef.current.startsWith('!')) {
      historyIndexRef.current = null
      setHistoryDraft('')
      replaceComposerValue('')
      setStatusError(null)
      return
    }
    if (key.escape && !dialogOpen && !inputValue) {
      const queued = [...queuedFollowUpsRef.current].reverse().find(
        (candidate) => candidate.sessionId === state.sessionId,
      )
      if (state.isStreaming && queued) {
        void cancelQueuedFollowUps({ noteId: queued.noteId })
        return
      }
      const now = Date.now()
      if (now - lastEmptyEscapeAt.current <= 700) {
        lastEmptyEscapeAt.current = 0
        const command = COMMANDS.find(({ id }) => id === 'session.editLast')
        if (command) void runCommand(command)
      } else {
        lastEmptyEscapeAt.current = now
      }
      return
    }
    if (key.ctrl && input.toLowerCase() === 'c') {
      if (shellAborter.current) shellAborter.current.abort()
      else if (state.isStreaming) cancelStream()
      else if (inputValue) {
        historyIndexRef.current = null
        replaceComposerValue('')
      }
      else if (exitArmed) {
        if (exitArmTimeout.current) clearTimeout(exitArmTimeout.current)
        exitArmTimeout.current = null
        exit()
      }
      else {
        setExitArmed(true)
        setStatusError('Press Ctrl+C again to exit, or Ctrl+D.')
        if (exitArmTimeout.current) clearTimeout(exitArmTimeout.current)
        exitArmTimeout.current = setTimeout(() => {
          setExitArmed(false)
          setStatusError((error) => error === 'Press Ctrl+C again to exit, or Ctrl+D.' ? null : error)
        }, 1_500)
      }
      return
    }
    if (key.ctrl && input.toLowerCase() === 'd' && !inputValue && !state.isStreaming) {
      if (exitArmTimeout.current) clearTimeout(exitArmTimeout.current)
      exitArmTimeout.current = null
      exit()
      return
    }
    if (key.ctrl && input.toLowerCase() === 'j') {
      historyIndexRef.current = null
      const current = inputValueRef.current
      const cursor = cursorOffsetRef.current
      const next = `${current.slice(0, cursor)}\n${current.slice(cursor)}`
      applyComposerValue(next, cursor + 1)
      return
    }
    if (key.ctrl && input.toLowerCase() === 'a') {
      applyComposerValue(inputValueRef.current, inputValueRef.current.startsWith('!') ? 1 : 0)
      return
    }
    if (key.ctrl && input.toLowerCase() === 'e') {
      applyComposerValue(inputValueRef.current)
      return
    }
    if (key.ctrl && input.toLowerCase() === 'o') {
      void openLatestFilePreview()
      return
    }
    if (key.ctrl && input.toLowerCase() === 'p') {
      historyIndexRef.current = null
      replaceComposerValue('/')
      setSuggestionIndex(0)
      return
    }
    const leaderResult = reduceLeaderKey(
      leaderState,
      { key: input, ctrl: Boolean(key.ctrl), now: Date.now() },
      { leaderKey: keybindings.leaderKey, bindings: keybindings.bindings },
    )
    setLeaderState(leaderResult.state)
    if (leaderResult.action === 'arm' || leaderResult.action === 'disarm') return
    if (leaderResult.action === 'run') {
      const command = COMMANDS.find(({ id }) => id === leaderResult.commandId)
      if (command) void runCommand(command)
      return
    }
    if (leaderResult.action === 'literal') {
      historyIndexRef.current = null
      const current = inputValueRef.current
      const cursor = cursorOffsetRef.current
      const next = `${current.slice(0, cursor)}${leaderResult.text}${current.slice(cursor)}`
      applyComposerValue(next, cursor + leaderResult.text.length)
      return
    }
    if (suggestions.length > 0 && key.upArrow) {
      setSuggestionIndex((index) => Math.max(0, index - 1))
      return
    }
    if (suggestions.length > 0 && key.downArrow) {
      setSuggestionIndex((index) => Math.min(suggestions.length - 1, index + 1))
      return
    }
    if (suggestions.length > 0 && key.tab) {
      completeSelectedSuggestion()
      return
    }
    if (attachmentSuggestions.length > 0 && activeAttachmentReference && key.upArrow) {
      setAttachmentSuggestionIndex((index) => Math.max(0, index - 1))
      return
    }
    if (attachmentSuggestions.length > 0 && activeAttachmentReference && key.downArrow) {
      setAttachmentSuggestionIndex((index) => Math.min(attachmentSuggestions.length - 1, index + 1))
      return
    }
    if (
      attachmentSuggestions.length > 0
      && activeAttachmentReference
      && (key.tab || key.return)
    ) {
      completeSelectedAttachment()
      return
    }
    if (skillSuggestions.length > 0 && activeSkillReference && key.upArrow) {
      setSkillSuggestionIndex((index) => Math.max(0, index - 1))
      return
    }
    if (skillSuggestions.length > 0 && activeSkillReference && key.downArrow) {
      setSkillSuggestionIndex((index) => Math.min(skillSuggestions.length - 1, index + 1))
      return
    }
    if (
      skillSuggestions.length > 0
      && activeSkillReference
      && (key.tab || key.return)
    ) {
      completeSelectedSkill()
      return
    }
    if (
      state.isStreaming &&
      key.tab &&
      inputValue.trim() &&
      !inputValue.startsWith('/') &&
      !inputValue.startsWith('!') &&
      !activeAttachmentReference &&
      !activeSkillReference
    ) {
      void submitInput()
      return
    }
    if (key.leftArrow) {
      const current = inputValueRef.current
      applyComposerValue(
        current,
        Math.max(
          current.startsWith('!') ? 1 : 0,
          previousGraphemeOffset(current, cursorOffsetRef.current),
        ),
      )
      return
    }
    if (key.rightArrow) {
      const current = inputValueRef.current
      applyComposerValue(
        current,
        nextGraphemeOffset(current, cursorOffsetRef.current),
      )
      return
    }
    if (key.upArrow && history.length > 0) {
      const currentIndex = historyIndexRef.current
      const nextIndex = currentIndex === null ? history.length - 1 : Math.max(0, currentIndex - 1)
      if (currentIndex === null) setHistoryDraft(inputValue)
      historyIndexRef.current = nextIndex
      replaceComposerValue(history[nextIndex] ?? '')
      return
    }
    if (key.downArrow && historyIndexRef.current !== null) {
      if (historyIndexRef.current < history.length - 1) {
        const nextIndex = historyIndexRef.current + 1
        historyIndexRef.current = nextIndex
        replaceComposerValue(history[nextIndex] ?? '')
      } else {
        historyIndexRef.current = null
        replaceComposerValue(historyDraft)
      }
      return
    }
    if (key.return) {
      const current = inputValueRef.current
      const alias = current.trim().split(/\s+/, 1)[0]?.toLowerCase()
      const exactCustomCommand = customCommandDefs.some((command) => command.aliases[0] === alias?.toLowerCase())
      if (current.startsWith('/') && !findByAlias(alias ?? '') && !exactCustomCommand && completeSelectedSuggestion()) return
      void submitInput()
      return
    }
    // Ink 5 names the DEL byte (0x7f) `delete`, although most terminals emit
    // it for Backspace. Preserve terminal Backspace semantics for both flags.
    if (key.backspace || key.delete) {
      const current = inputValueRef.current
      const cursor = cursorOffsetRef.current
      if (cursor > 0) {
        historyIndexRef.current = null
        const previousOffset = previousGraphemeOffset(current, cursor)
        applyComposerValue(
          `${current.slice(0, previousOffset)}${current.slice(cursor)}`,
          previousOffset,
        )
      }
      return
    }
    if (input && !key.ctrl && !key.meta) {
      setExitArmed(false)
      historyIndexRef.current = null
      // Ink forwards a paste as one input string. PTYs and automation can also
      // coalesce the following Enter byte into that same string, so parse the
      // trailing submit delimiter instead of inserting a literal carriage
      // return into the prompt. Bracketed paste newlines remain prompt text.
      const bracketedPaste = input.includes('\x1b[200~')
        || input.includes('\x1b[201~')
        || input.includes('[200~')
        || input.includes('[201~')
      // This composer intentionally treats LF as a multiline insertion
      // (Ctrl+J) and CR as submit (Enter).
      const submitAfterInsertion = !bracketedPaste && /(?:\r\n|\r)$/u.test(input)
      const insertionInput = submitAfterInsertion
        ? input.replace(/(?:\r\n|\r)+$/u, '')
        : input
      const inserted = stripBracketedPasteDelimiters(insertionInput)
      if (!inserted) {
        if (submitAfterInsertion) void submitInput()
        return
      }
      const applied = applyHangulInputChunk(
        inputValueRef.current,
        cursorOffsetRef.current,
        inserted,
        hangulInputStateRef.current,
      )
      inputValueRef.current = applied.value
      cursorOffsetRef.current = applied.cursorOffset
      hangulInputStateRef.current = applied.state
      setInputValue(applied.value)
      setCursorOffset(applied.cursorOffset)
      if (submitAfterInsertion) void submitInput()
    }
  }, { isActive: !dialogOpen })

  const visibleMessages = useMemo(() => {
    const start = Math.min(clearedMessageCount, state.messages.length)
    const result: Message[] = state.messages.slice(start)
    if (showThinking && state.thinkingText) {
      const reasoningMessage: Message = {
        id: '__active-reasoning__',
        role: 'system',
        variant: 'thinking',
        content: state.thinkingText,
        timestamp: Date.now(),
      }
      if (state.currentMessage) {
        result.push(reasoningMessage)
      } else {
        const finalAssistantIndex = result.findLastIndex(({ role }) => role === 'assistant')
        result.splice(finalAssistantIndex >= 0 ? finalAssistantIndex : result.length, 0, reasoningMessage)
      }
    }
    if (state.currentMessage) {
      result.push({
        id: '__active-assistant__',
        role: 'assistant',
        content: state.currentMessage,
        timestamp: Date.now(),
        toolCalls: state.toolCalls,
      })
    }
    return result
  }, [clearedMessageCount, showThinking, state.currentMessage, state.messages, state.thinkingText, state.toolCalls])
  const currentModel = useMemo(() => {
    const model = providers
      .find(({ id }) => id === state.provider)
      ?.models.find(({ id }) => id === state.model)
    return model ?? null
  }, [providers, state.model, state.provider])
  const conversationContextEstimate = useMemo(
    () => state.messages.length > 0
      ? estimateConversationTokens(state.messages)
      : null,
    [state.messages],
  )
  const contextTokens = state.contextUsage?.inputTokens ?? conversationContextEstimate
  const contextWindow = state.contextUsage?.contextWindowTokens ?? currentModel?.contextWindow ?? null
  const contextPercent = contextTokens == null
    ? null
    : contextFillPercent(contextTokens, contextWindow)
  const contextEstimated = state.contextUsage?.source === 'estimated'
    || (state.contextUsage == null && conversationContextEstimate != null)
  const humanInputPending = Boolean(state.pendingApproval || pendingQuestion)
  const humanInputLabel = state.pendingApproval
    ? `approval required · ${state.pendingApproval.toolName}`
    : pendingQuestion
      ? 'answer required · respond to continue'
      : null
  const activityAnimation = useActivityAnimation(state.isStreaming && !humanInputPending)
  const progressLabel = formatInlineAgentProgress({
    phase: state.currentPhase,
    streamStatus: state.streamStatus,
    counts: state.stateBoardCounts,
  })
  const interactiveProgressLabel = humanInputLabel
    ?? (state.isStreaming ? progressLabel ?? 'waiting for model events' : progressLabel)

  const settingsSelect = useCallback((itemId: string) => {
    if (itemId.startsWith('command:')) {
      const command = COMMANDS.find(({ id }) => id === itemId.slice('command:'.length))
      if (command) {
        closeTopDialog()
        void runCommand(command)
      }
      return
    }
    const item = SETTINGS_ITEMS.find(({ id }) => id === itemId)
    if (!item) return
    if (item.id === 'model.default') {
      openModelPicker('default')
      return
    }
    if (item.id === 'permissions.defaultAutonomy') {
      setAutonomyPurpose('default')
      openDialog('autonomy')
      void client.config()
        .then((config) => setDefaultAutonomy((config.agent?.autonomy as typeof state.autonomy | undefined) ?? null))
        .catch((error) => setStatusError(errorMessage(error)))
      return
    }
    if (item.id === 'agent.maxTokens') { openDialog('maxTokens'); return }
    if (item.id === 'diagnostics.doctor') {
      setInfo({ title: 'Doctor', body: 'Running daemon diagnostics…' })
      openDialog('info')
      void client.healthReport('markdown')
        .then((body) => setInfo({ title: 'Doctor', body }))
        .catch((error) => setInfo({ title: 'Doctor', body: `Diagnostics failed: ${errorMessage(error)}` }))
      return
    }
    if (item.dialogId) { openDialog(item.dialogId); return }
    if (item.sessionKey === 'model') openModelPicker('session')
    else if (item.sessionKey === 'mode') openDialog('mode')
    else if (item.sessionKey === 'thinkingLevel') openDialog('thinking')
    else if (item.sessionKey === 'autonomy') { setAutonomyPurpose('session'); openDialog('autonomy') }
    else setStatusError(`No settings handler is registered for ${item.id}.`)
  }, [client, closeTopDialog, openDialog, openModelPicker, runCommand])

  const loadSelectedSession = useCallback(async (sessionId: string) => {
    setSessionsError(null)
    try {
      const [session, artifacts] = await Promise.all([
        client.session(sessionId),
        client.sessionArtifacts(sessionId).catch(() => []),
      ])
      const workspaceError = sessionWorkspaceLoadError(session, fileRoot)
      if (workspaceError) {
        setSessionsError(workspaceError)
        return
      }
      loadSession(session, artifacts)
      setClearedMessageCount(0)
      closeTopDialog()
    } catch (error) {
      setSessionsError(errorMessage(error))
    }
  }, [client, closeTopDialog, fileRoot, loadSession])

  const answerPendingQuestion = useCallback(async (answer: string) => {
    if (!pendingQuestion || questionBusy) return
    setQuestionBusy(true)
    setQuestionError(null)
    try {
      await answerQuestion(pendingQuestion.id, answer)
    } catch (error) {
      setQuestionError(errorMessage(error))
    } finally {
      setQuestionBusy(false)
    }
  }, [answerQuestion, pendingQuestion, questionBusy])

  const searchMemory = useCallback((query: string) => {
    setMemoryLoading(true)
    setMemoryError(null)
    void client.searchMemory(query, { type: 'hybrid', limit: 20 })
      .then(setMemoryResults)
      .catch((error) => setMemoryError(errorMessage(error)))
      .finally(() => setMemoryLoading(false))
  }, [client])

  const searchRag = useCallback((query: string) => {
    setRagInitialQuery(query)
    setRagRefresh((value) => value + 1)
  }, [])

  const dialogEntries: DialogEntry[] = dialogs.map((id) => ({
    id,
    node: id === 'providerSetup' && providerSetup ? (
      <ProviderSetupModal
        key={id}
        mode={providerSetup.mode}
        step={providerSetup.step}
        applyToCurrentSession={providerSetup.applyToCurrentSession}
        preset={providerSetup.preset}
        presets={PROVIDER_WIZARD_PRESETS}
        selectedPresetIndex={providerSetup.selectedPresetIndex}
        providerId={providerSetup.providerId}
        baseUrl={providerSetup.baseUrl}
        apiKeyEnvVar={providerSetup.apiKeyEnvVar}
        apiKeyValue={providerSetup.apiKeyValue}
        headersText={providerSetup.headersText ?? ''}
        model={providerSetup.model}
        modelSuggestions={providerSetup.modelSuggestions}
        selectedModelSuggestionIndex={providerSetup.modelSuggestionIndex}
        loading={providerSetup.loading}
        validating={providerSetup.validating}
        saving={providerSetup.saving}
        error={providerSetup.error}
        maxVisibleItems={Math.max(4, Math.min(8, rows - 10))}
        onSelectPresetIndex={(index) => setProviderSetup((current) => current ? { ...current, selectedPresetIndex: index } : current)}
        onSelectPreset={selectProviderPreset}
        onProviderIdChange={(providerId) => setProviderSetup((current) => current ? { ...current, providerId, error: null } : current)}
        onBaseUrlChange={(baseUrl) => setProviderSetup((current) => current ? { ...current, baseUrl, error: null } : current)}
        onApiKeyEnvVarChange={(apiKeyEnvVar) => setProviderSetup((current) => current ? { ...current, apiKeyEnvVar, error: null } : current)}
        onApiKeyValueChange={(apiKeyValue) => setProviderSetup((current) => current ? { ...current, apiKeyValue, error: null } : current)}
        onHeadersTextChange={(headersText) => setProviderSetup((current) => current ? { ...current, headersText, error: null } : current)}
        onModelChange={(model) => setProviderSetup((current) => current ? { ...current, model, modelAliasSource: undefined, error: null } : current)}
        onSelectModelSuggestionIndex={(modelSuggestionIndex) => setProviderSetup((current) => current ? { ...current, modelSuggestionIndex } : current)}
        onApplyModelSuggestion={(index) => setProviderSetup((current) => {
          const model = current?.modelSuggestions[index]
          if (!current || !model) return current
          return {
            ...current,
            model,
            modelSuggestionIndex: index,
            modelAliasSource: model === current.model ? current.modelAliasSource : undefined,
            error: null,
          }
        })}
        onToggleApplyToCurrentSession={() => setProviderSetup((current) => current ? { ...current, applyToCurrentSession: !current.applyToCurrentSession } : current)}
        onNext={() => { void advanceProviderSetup() }}
        onBack={backProviderSetup}
        onClose={() => { setProviderSetup(null); closeTopDialog() }}
      />
    ) : id === 'providerDelete' && providerDelete ? (
      <ProviderDeleteModal
        key={id}
        providerId={providerDelete.providerId}
        providerName={providerDelete.providerName}
        nextProviderId={providerDelete.nextProviderId}
        nextModel={providerDelete.nextModel}
        currentProviderId={state.provider}
        currentModel={state.model}
        willSwitchCurrentSession={state.provider === providerDelete.providerId}
        busy={providerDelete.deleting}
        error={providerDelete.error}
        onConfirm={() => { void confirmProviderDelete() }}
        onClose={() => { setProviderDelete(null); closeTopDialog() }}
      />
    ) : id === 'swarmAttach' && swarmAttachTarget ? (
      <SwarmAttachView
        key={id}
        client={swarmClient}
        target={swarmAttachTarget}
        width={width}
        height={Math.max(8, rows - 2)}
        onExit={(reason, message) => {
          setSwarmAttachTarget(null)
          closeTopDialog()
          showNotice(`Swarm attach ${reason}${message ? `: ${message}` : ''}`)
        }}
      />
    ) : id === 'filePreview' && filePreview ? (
      <FilePreviewDialog
        key={id}
        path={filePreview.path}
        content={filePreview.content}
        width={width}
        height={Math.max(8, rows - 2)}
        onClose={() => {
          setFilePreview(null)
          closeTopDialog()
        }}
      />
    ) : id === 'agentLoop' ? (
      <AgentLoopDialog
        key={id}
        entries={state.graphTrace ?? []}
        currentPhase={state.currentPhase}
        isStreaming={state.isStreaming}
        streamStatus={state.streamStatus}
        width={width}
        height={Math.max(8, rows - 2)}
        onClose={closeTopDialog}
      />
    ) : id === 'planTodo' ? (
      <PlanTodoDialog
        key={id}
        workingMemory={state.plannerWorkingMemory}
        progress={state.runWorkProgress}
        isStreaming={state.isStreaming}
        width={width}
        height={Math.max(8, rows - 2)}
        onClose={closeTopDialog}
      />
    ) : id === 'files' ? (
      <FilePicker
        key={id}
        rootDir={fileRoot}
        currentDir={fileDirectory}
        items={fileItems}
        selectedIndex={fileIndex}
        selectedPaths={selectedFilePaths}
        selectedLabels={selectedFilePaths.map((path) => relative(fileRoot, path) || basename(path))}
        loading={fileLoading}
        error={fileError}
        maxVisibleItems={Math.max(4, Math.min(12, rows - 10))}
        onSelectIndex={setFileIndex}
        onOpenDirectory={(item) => setFileDirectory(item.absolutePath)}
        onNavigateUp={() => {
          if (fileDirectory !== fileRoot) setFileDirectory(dirname(fileDirectory))
        }}
        onToggleFile={(item) => setSelectedFilePaths((current) => current.includes(item.absolutePath)
          ? current.filter((path) => path !== item.absolutePath)
          : [...current, item.absolutePath])}
        onConfirm={() => {
          if (selectedFilePaths.length === 0) { setFileError('Select at least one file first.'); return }
          const references = selectedFilePaths.map((path) => {
            const displayPath = relative(fileRoot, path) || basename(path)
            return /\s/.test(displayPath) ? `@"${displayPath}"` : `@${displayPath}`
          })
          replaceComposerValue([inputValue.trim(), ...references].filter(Boolean).join(' '))
          closeTopDialog()
        }}
        onClose={closeTopDialog}
      />
    ) : id === 'settings' ? (
      <SettingsDialog key={id} width={width} onSelectItem={settingsSelect} onClose={closeTopDialog} errorText={statusError} />
    ) : id === 'model' ? (
      <ModelPicker
        key={id}
        query={modelQuery}
        providers={providers}
        currentProvider={state.provider}
        currentModel={state.model}
        defaultProvider={providerDefaults.provider}
        defaultModel={providerDefaults.model}
        selectedIndex={modelIndex}
        loading={modelLoading}
        error={modelError}
        onQueryChange={setModelQuery}
        onSelectIndex={setModelIndex}
        onConfirm={handleModelPickerSelection}
        onAddProvider={() => { void openProviderSetup() }}
        onClose={closeTopDialog}
      />
    ) : id === 'mode' ? (
      <ModePicker
        key={id}
        query={modeQuery}
        agents={agents}
        currentMode={state.mode}
        selectedIndex={modeIndex}
        loading={modeLoading}
        error={modeError}
        onQueryChange={setModeQuery}
        onSelectIndex={setModeIndex}
        onConfirm={(agent) => { dispatch({ type: 'SET_MODE', mode: agent.id }); closeTopDialog() }}
        onClose={closeTopDialog}
      />
    ) : id === 'sessions' ? (
      <SessionPicker
        key={id}
        query={sessionQuery}
        sessions={sessions}
        projectSessionIds={[]}
        currentSessionId={state.sessionId}
        recentSessionIds={sessions.map(({ id: sessionId }) => sessionId)}
        selectedIndex={sessionIndex}
        loading={sessionsLoading}
        error={sessionsError}
        onQueryChange={setSessionQuery}
        onSelectIndex={setSessionIndex}
        onConfirm={(session) => { void loadSelectedSession(session.id) }}
        onBranch={(session) => {
          void client.branchSession(session.id)
            .then(({ branchId }) => loadSelectedSession(branchId))
            .catch((error) => setSessionsError(errorMessage(error)))
        }}
        onCompact={(session) => {
          void client.compactSession(session.id)
            .then((result) => setSessionsError(`Compacted ${result.originalTokens.toLocaleString()} → ${result.compactedTokens.toLocaleString()} tokens.`))
            .catch((error) => setSessionsError(errorMessage(error)))
        }}
        onDelete={(session) => {
          void client.deleteSession(session.id)
            .then(() => setSessionRefresh((value) => value + 1))
            .catch((error) => setSessionsError(errorMessage(error)))
        }}
        onExport={(session, format) => {
          void client.sessionExport(session.id, format as 'markdown')
            .then((content) => {
              setInfo({
                title: `Session export · ${session.id.slice(0, 8)}`,
                body: typeof content === 'string' ? content : JSON.stringify(content, null, 2),
              })
              openDialog('info')
            })
            .catch((error) => setSessionsError(errorMessage(error)))
        }}
        onRetry={() => setSessionRefresh((value) => value + 1)}
        onClose={closeTopDialog}
      />
    ) : id === 'projects' ? (
      <ProjectDialog
        key={id}
        projects={projects}
        currentProjectId={state.projectId}
        width={width}
        loading={projectsLoading}
        error={projectsError}
        onSelect={(project) => {
          const apply = async () => {
            if (project && state.sessionId && !project.sessionIds.includes(state.sessionId)) {
              await client.attachSessionToProject(project.id, state.sessionId)
            }
            dispatch({ type: 'SET_PROJECT', projectId: project?.id ?? null, projectName: project?.name ?? null })
            closeTopDialog()
          }
          void apply().catch((error) => setProjectsError(errorMessage(error)))
        }}
        onClose={closeTopDialog}
      />
    ) : id === 'skills' ? (
      <SkillManagerPicker
        key={id}
        query={skillsQuery}
        skills={skills}
        selectedIndex={skillsIndex}
        loading={skillsLoading}
        togglingSkillId={skillsToggling}
        error={skillsError}
        message={skillsMessage}
        onQueryChange={(value) => { setSkillsQuery(value); setSkillsIndex(0) }}
        onSelectIndex={setSkillsIndex}
        onToggle={(skill) => {
          const enabled = skill.enabled === false
          setSkillsToggling(skill.id)
          setSkillsError(null)
          void client.setSkillEnabled(skill.id, enabled)
            .then((result) => {
              setSkills((items) => items.map((item) => item.id === result.id ? { ...item, enabled: result.enabled } : item))
              setSkillsMessage(`${result.enabled ? 'Enabled' : 'Disabled'} ${result.id}.`)
            })
            .catch((error) => setSkillsError(errorMessage(error)))
            .finally(() => setSkillsToggling(null))
        }}
        onRun={(skill) => {
          if (skill.enabled === false) { setSkillsMessage('Enable this skill before running it.'); return }
          closeTopDialog()
          replaceComposerValue(`/run ${skill.id} `)
        }}
        onClose={closeTopDialog}
      />
    ) : id === 'memory' ? (
      <MemorySearchDialog
        key={id}
        initialQuery={memoryInitialQuery}
        width={width}
        status={memoryStatus}
        results={memoryResults}
        loading={memoryLoading}
        error={memoryError}
        onSearch={searchMemory}
        onClose={closeTopDialog}
      />
    ) : id === 'usage' ? (
      <UsageDialog
        key={id}
        summary={usageSummary}
        daily={usageDaily}
        days={usageDays}
        loading={usageLoading}
        error={usageError}
        height={Math.max(6, rows - 4)}
        onClose={closeTopDialog}
      />
    ) : id === 'artifacts' ? (
      <ArtifactDialog
        key={id}
        artifacts={state.artifacts}
        width={width}
        onCopy={(artifact) => copyText(artifact.content)}
        onClose={closeTopDialog}
      />
    ) : id === 'rag' ? (
      <RagDialog
        key={`${id}:${ragInitialQuery}`}
        initialQuery={ragInitialQuery}
        sources={ragSources}
        hits={ragHits}
        vectorInfo={ragVectorInfo}
        loading={ragLoading}
        error={ragError}
        width={width}
        onSearch={searchRag}
        onClose={closeTopDialog}
      />
    ) : id === 'mcp' ? (
      <McpDialog
        key={`${id}:${mcpInitialQuery}`}
        servers={mcpServers}
        initialQuery={mcpInitialQuery}
        loading={mcpLoading}
        error={mcpError}
        width={width}
        onClose={closeTopDialog}
      />
    ) : id === 'theme' ? (
      <ChoiceDialog
        key={id}
        title="Theme"
        options={themeOptions.map((option) => ({ value: option.id, label: option.label, description: option.description }))}
        current={themeId}
        width={width}
        onSelect={(value) => {
          const nextTheme = value as ThemeId
          setActiveTheme(nextTheme)
          setThemeId(nextTheme)
          closeTopDialog()
        }}
        onClose={closeTopDialog}
      />
    ) : id === 'thinking' ? (
      <ChoiceDialog key={id} title="Thinking level" options={THINKING_OPTIONS} current={state.thinkingLevel} width={width} onSelect={(value) => { dispatch({ type: 'SET_THINKING_LEVEL', thinkingLevel: value as typeof state.thinkingLevel }); closeTopDialog() }} onClose={closeTopDialog} />
    ) : id === 'autonomy' ? (
      <ChoiceDialog key={id} title={autonomyPurpose === 'default' ? 'Default autonomy' : 'Autonomy'} options={AUTONOMY_OPTIONS} current={autonomyPurpose === 'default' ? (defaultAutonomy ?? state.autonomy) : state.autonomy} width={width} onSelect={(value) => {
        if (autonomyPurpose === 'default') {
          void client.updateConfig({ 'agent.autonomy': value as typeof state.autonomy })
            .then(() => { setDefaultAutonomy(value as typeof state.autonomy); closeTopDialog() })
            .catch((error) => setStatusError(errorMessage(error)))
        } else {
          dispatch({ type: 'SET_AUTONOMY', autonomy: value as typeof state.autonomy })
          closeTopDialog()
        }
      }} onClose={closeTopDialog} />
    ) : id === 'maxTokens' ? (
      <ChoiceDialog key={id} title="Max output tokens" options={MAX_TOKEN_OPTIONS} current={state.maxTokens?.toString() ?? 'default'} width={width} onSelect={(value) => { dispatch({ type: 'SET_MAX_TOKENS', maxTokens: value === 'default' ? null : Number(value) }); closeTopDialog() }} onClose={closeTopDialog} />
    ) : (
      <InfoDialog key={id} title={info?.title ?? 'Information'} body={info?.body ?? 'Information unavailable.'} width={width} onClose={closeTopDialog} />
    ),
  }))

  return (
    <Box flexDirection="column" width={width}>
      <Conversation
        messages={visibleMessages}
        isStreaming={state.isStreaming}
        width={width}
        runningTools={state.toolCalls}
        compactTools={!showToolDetails}
        showBrandMark={!dialogOpen}
        activityFrame={activityAnimation.frame}
        suspended={dialogOpen}
      />
      {state.pendingApproval ? (
        <ApprovalModal compact approval={state.pendingApproval} onResolve={(approved, scope, note) => { void resolveApproval(approved, scope, note) }} />
      ) : pendingQuestion ? (
        <QuestionDialog
          question={pendingQuestion}
          width={width}
          busy={questionBusy}
          error={questionError}
          onAnswer={(answer) => { void answerPendingQuestion(answer) }}
        />
      ) : <DialogStack entries={dialogEntries} />}
      {!dialogOpen && suggestions.length > 0 ? (
        <Box flexDirection="column">
          {visibleSuggestions.map((command, index) => {
            const absoluteIndex = suggestionWindowStart + index
            return (
            <Text key={command.id} color={absoluteIndex === selectedSuggestionIndex ? colors.primary : colors.dimText}>
              {`${absoluteIndex === selectedSuggestionIndex ? '› ' : '  '}${command.aliases[0]}  ${command.title}`}
            </Text>
            )
          })}
        </Box>
      ) : null}
      {!dialogOpen && suggestions.length === 0 && activeAttachmentReference && attachmentSuggestions.length > 0 ? (
        <Box flexDirection="column">
          {attachmentSuggestions.slice(0, 8).map((candidate, index) => (
            <Text key={candidate.path} color={index === selectedAttachmentIndex ? colors.primary : colors.dimText}>
              {`${index === selectedAttachmentIndex ? '› ' : '  '}@${candidate.path}${candidate.isDirectory ? ' [dir]' : ''}`}
            </Text>
          ))}
          <Text color={colors.dimText}>↑/↓ move  Enter/Tab attach</Text>
        </Box>
      ) : null}
      {!dialogOpen && suggestions.length === 0 && activeSkillReference && skillSuggestions.length > 0 ? (
        <Box flexDirection="column">
          {skillSuggestions.map((skill, index) => (
            <Text key={skill.id} color={index === selectedSkillIndex ? colors.primary : colors.dimText}>
              {`${index === selectedSkillIndex ? '› ' : '  '}@skill:${skill.id} · ${skill.name}`}
            </Text>
          ))}
          <Text color={colors.dimText}>↑/↓ move  Enter/Tab select</Text>
        </Box>
      ) : null}
      {!dialogOpen ? <Composer value={inputValue} cursorOffset={cursorOffset} leaderArmed={leaderState.armed} leaderHintText={leaderHint(keybindings.bindings, titles)} busy={state.isStreaming} busyLabel={interactiveProgressLabel} activityFrame={activityAnimation.frame} activityColor={activityAnimation.color} width={width} hardwareCursor={runtime?.hardwareCursor} /> : null}
      {!dialogOpen && lastTurnRecap ? (
        <Text color={colors.dimText}>{lastTurnRecap}</Text>
      ) : null}
      <StatusLine
        model={providerMetadataLoading && state.provider === 'default' && state.model === 'default'
          ? 'Connecting to daemon…'
          : `${state.provider}/${state.model}`}
        mode={state.mode}
        autonomy={state.autonomy}
        contextPercent={contextPercent}
        contextTokens={contextTokens}
        contextWindow={contextWindow}
        contextEstimated={contextEstimated}
        costLabel={state.usage.cost > 0 ? `$${state.usage.cost.toFixed(4)}` : null}
        progressLabel={interactiveProgressLabel}
        attention={humanInputPending}
        streamStartedAt={humanInputPending ? null : state.streamStartedAt}
        error={statusError ?? state.error}
        width={width}
      />
    </Box>
  )
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
