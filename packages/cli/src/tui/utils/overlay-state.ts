export type OverlayId =
  | 'help'
  | 'autonomy-picker'
  | 'provider-setup'
  | 'provider-delete'
  | 'model-picker'
  | 'mode-picker'
  | 'file-picker'
  | 'session-picker'
  | 'skill-manager'
  | 'skill-store'

export type OverlayOpenTarget =
  | 'session-picker'
  | 'autonomy-picker'
  | 'model-picker'
  | 'mode-picker'
  | 'file-picker'
  | 'skill-store'
  | 'skill-manager'
  | 'provider-setup'
  | 'provider-delete'

export interface OverlayStateSnapshot {
  helpOpen: boolean
  autonomyPickerOpen: boolean
  providerSetupOpen: boolean
  providerDeleteOpen: boolean
  modelPickerOpen: boolean
  modePickerOpen: boolean
  filePickerOpen: boolean
  sessionPickerOpen: boolean
  skillManagerOpen: boolean
  skillStoreOpen: boolean
}

interface OverlayTargetConfig {
  action: string
  blockers: OverlayId[]
}

export interface InlinePanelVisibilityOptions {
  overlayState: OverlayStateSnapshot
  isStreaming: boolean
  pendingApproval: boolean
  showCommandPalette: boolean
  showAttachmentPalette: boolean
  showSkillPalette?: boolean
  memorySearchOpen: boolean
  ragPanelOpen: boolean
  usageDashboardOpen: boolean
  showMcp: boolean
  artifactCount: number
  toolCallCount: number
}

export interface InlinePanelVisibility {
  showMemorySearchPanel: boolean
  showRagPanel: boolean
  showUsageDashboardPanel: boolean
  showMcpPanel: boolean
  showArtifactPanel: boolean
  showToolActivityPanel: boolean
}

const OVERLAY_PRIORITY: OverlayId[] = [
  'help',
  'autonomy-picker',
  'provider-setup',
  'provider-delete',
  'model-picker',
  'mode-picker',
  'file-picker',
  'session-picker',
  'skill-manager',
  'skill-store',
]

const OVERLAY_OPEN_KEYS: Record<OverlayId, keyof OverlayStateSnapshot> = {
  help: 'helpOpen',
  'autonomy-picker': 'autonomyPickerOpen',
  'provider-setup': 'providerSetupOpen',
  'provider-delete': 'providerDeleteOpen',
  'model-picker': 'modelPickerOpen',
  'mode-picker': 'modePickerOpen',
  'file-picker': 'filePickerOpen',
  'session-picker': 'sessionPickerOpen',
  'skill-manager': 'skillManagerOpen',
  'skill-store': 'skillStoreOpen',
}

const OVERLAY_LABELS: Record<OverlayId, string> = {
  help: 'help modal',
  'autonomy-picker': 'autonomy picker',
  'provider-setup': 'provider setup',
  'provider-delete': 'provider delete confirmation',
  'model-picker': 'model picker',
  'mode-picker': 'mode picker',
  'file-picker': 'file picker',
  'session-picker': 'session picker',
  'skill-manager': 'installed skill manager',
  'skill-store': 'skill catalog',
}

const OVERLAY_TARGETS: Record<OverlayOpenTarget, OverlayTargetConfig> = {
  'session-picker': {
    action: 'switching sessions',
    blockers: [
      'file-picker',
      'model-picker',
      'mode-picker',
      'autonomy-picker',
      'provider-setup',
      'provider-delete',
      'skill-manager',
      'skill-store',
    ],
  },
  'skill-manager': {
    action: 'managing installed skills',
    blockers: [
      'session-picker',
      'file-picker',
      'model-picker',
      'mode-picker',
      'autonomy-picker',
      'provider-setup',
      'provider-delete',
      'skill-store',
    ],
  },
  'skill-store': {
    action: 'opening the skill catalog',
    blockers: [
      'session-picker',
      'file-picker',
      'model-picker',
      'mode-picker',
      'autonomy-picker',
      'provider-setup',
      'provider-delete',
      'skill-manager',
    ],
  },
  'autonomy-picker': {
    action: 'opening the autonomy picker',
    blockers: [
      'session-picker',
      'mode-picker',
      'model-picker',
      'file-picker',
      'provider-setup',
      'provider-delete',
      'skill-manager',
      'skill-store',
    ],
  },
  'model-picker': {
    action: 'opening the model picker',
    blockers: [
      'session-picker',
      'file-picker',
      'autonomy-picker',
      'mode-picker',
      'provider-setup',
      'provider-delete',
      'skill-manager',
      'skill-store',
    ],
  },
  'mode-picker': {
    action: 'opening the mode picker',
    blockers: [
      'session-picker',
      'file-picker',
      'model-picker',
      'autonomy-picker',
      'provider-setup',
      'provider-delete',
      'skill-manager',
      'skill-store',
    ],
  },
  'file-picker': {
    action: 'opening the file picker',
    blockers: [
      'session-picker',
      'mode-picker',
      'model-picker',
      'autonomy-picker',
      'provider-setup',
      'provider-delete',
      'skill-manager',
      'skill-store',
    ],
  },
  'provider-setup': {
    action: 'opening provider setup',
    blockers: [
      'session-picker',
      'file-picker',
      'autonomy-picker',
      'mode-picker',
      'provider-delete',
      'model-picker',
      'skill-manager',
      'skill-store',
    ],
  },
  'provider-delete': {
    action: 'deleting providers',
    blockers: [
      'provider-setup',
      'provider-delete',
      'model-picker',
      'skill-manager',
      'skill-store',
    ],
  },
}

const INLINE_PANEL_OVERLAYS: OverlayId[] = [
  'autonomy-picker',
  'model-picker',
  'mode-picker',
  'file-picker',
  'session-picker',
  'skill-manager',
  'skill-store',
]

const TOOL_ACTIVITY_OVERLAYS: OverlayId[] = [
  ...INLINE_PANEL_OVERLAYS,
  'provider-setup',
  'provider-delete',
  'help',
]

function isOverlayOpen(
  state: OverlayStateSnapshot,
  overlay: OverlayId,
): boolean {
  return state[OVERLAY_OPEN_KEYS[overlay]]
}

function hasAnyOverlay(
  state: OverlayStateSnapshot,
  overlays: readonly OverlayId[],
): boolean {
  return overlays.some((overlay) => isOverlayOpen(state, overlay))
}

export function getForegroundOverlayId(
  state: OverlayStateSnapshot,
): OverlayId | null {
  return OVERLAY_PRIORITY.find((overlay) => isOverlayOpen(state, overlay)) ?? null
}

export function hasForegroundOverlay(
  state: OverlayStateSnapshot,
): boolean {
  return getForegroundOverlayId(state) !== null
}

export function getComposerBlockedReason(
  state: OverlayStateSnapshot,
): string | null {
  const foreground = getForegroundOverlayId(state)
  return foreground ? `${OVERLAY_LABELS[foreground]} is open` : null
}

export function getOverlayOpenBlockMessage(
  state: OverlayStateSnapshot,
  target: OverlayOpenTarget,
  options: {
    allowModelPicker?: boolean
  } = {},
): string | null {
  const allowedBlockers = new Set<OverlayId>()
  if (options.allowModelPicker) {
    allowedBlockers.add('model-picker')
  }

  const config = OVERLAY_TARGETS[target]
  const blocker = config.blockers.find((overlay) => (
    !allowedBlockers.has(overlay) && isOverlayOpen(state, overlay)
  ))

  if (!blocker) {
    return null
  }

  return `Close ${OVERLAY_LABELS[blocker]} before ${config.action}.`
}

export function isTranscriptNavigationBlocked(
  state: OverlayStateSnapshot,
  options: {
    showCommandPalette: boolean
    showAttachmentPalette: boolean
    showSkillPalette?: boolean
    pendingApproval: boolean
  },
): boolean {
  return (
    hasForegroundOverlay(state)
    || options.showCommandPalette
    || options.showAttachmentPalette
    || Boolean(options.showSkillPalette)
    || options.pendingApproval
  )
}

export function deriveInlinePanelVisibility(
  options: InlinePanelVisibilityOptions,
): InlinePanelVisibility {
  const inlinePanelBlocked = (
    options.pendingApproval
    || hasAnyOverlay(options.overlayState, INLINE_PANEL_OVERLAYS)
  )

  const showRagPanel = (
    options.ragPanelOpen
    && !inlinePanelBlocked
  )
  const showMemorySearchPanel = (
    options.memorySearchOpen
    && !showRagPanel
    && !inlinePanelBlocked
  )
  const showUsageDashboardPanel = (
    options.usageDashboardOpen
    && !showRagPanel
    && !showMemorySearchPanel
    && !inlinePanelBlocked
  )
  const showMcpPanel = (
    options.showMcp
    && !showRagPanel
    && !showMemorySearchPanel
    && !showUsageDashboardPanel
    && !inlinePanelBlocked
  )
  // CLI artifact browsing is on-demand via /artifacts to keep the shell footer
  // focused on the current run rather than previously saved outputs.
  const showArtifactPanel = false
  const showToolActivityPanel = (
    (options.toolCallCount > 0 || options.isStreaming)
    && !showRagPanel
    && !showMemorySearchPanel
    && !showUsageDashboardPanel
    && !showMcpPanel
    && !showArtifactPanel
    && !options.pendingApproval
    && !options.showCommandPalette
    && !options.showAttachmentPalette
    && !options.showSkillPalette
    && !hasAnyOverlay(options.overlayState, TOOL_ACTIVITY_OVERLAYS)
  )

  return {
    showRagPanel,
    showMemorySearchPanel,
    showUsageDashboardPanel,
    showMcpPanel,
    showArtifactPanel,
    showToolActivityPanel,
  }
}
