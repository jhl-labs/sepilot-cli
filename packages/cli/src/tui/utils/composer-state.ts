export interface ComposerActivityOptions {
  isStreaming: boolean
  streamLabel: string | null
  localShellActive: boolean
  localShellProgressLabel: string | null
}

export interface ComposerActivityState {
  inputBusy: boolean
  activeProgressLabel: string | null
}

export interface ComposerPresentationOptions {
  composerBlockedReason: string | null
  pendingApprovalToolName: string | null
  localShellActive: boolean
  hydratingSession: boolean
  showPalette: boolean
  isBangInput: boolean
  attachmentSuggestionCount: number
  skillAutocompleteActive?: boolean
  composerOverlayBlocked: boolean
  pendingAttachmentPaths: string[]
}

export interface ComposerPresentationState {
  blockedReason: string | null
  blockedHint: string | null
  paletteOpen: boolean
  attachmentAutocompleteOpen: boolean
  skillAutocompleteOpen: boolean
  pendingAttachments: string[]
}

export function shouldSubmitImmediatelyWhileBusy(text: string): boolean {
  return /^\/(?:answer|questions)\b/.test(text.trim())
}

export function deriveComposerActivityState(
  options: ComposerActivityOptions,
): ComposerActivityState {
  return {
    inputBusy: options.isStreaming || options.localShellActive,
    activeProgressLabel: options.isStreaming
      ? options.streamLabel
      : options.localShellProgressLabel,
  }
}

export function deriveComposerPresentationState(
  options: ComposerPresentationOptions,
): ComposerPresentationState {
  const blockedReason = options.composerBlockedReason
    ? options.composerBlockedReason
    : options.pendingApprovalToolName
      ? `${options.pendingApprovalToolName} requires approval`
      : options.localShellActive
        ? 'running local shell command...'
        : options.hydratingSession
          ? 'loading session history...'
          : null

  const blockedHint = options.composerBlockedReason
    ? 'Esc:close'
    : options.pendingApprovalToolName
      ? '↑/↓:choose Enter:select y/s/a/n'
      : blockedReason
        ? 'Please wait'
        : null

  return {
    blockedReason,
    blockedHint,
    paletteOpen: options.showPalette,
    attachmentAutocompleteOpen:
      !options.showPalette &&
      !options.isBangInput &&
      !options.skillAutocompleteActive &&
      options.attachmentSuggestionCount > 0 &&
      !options.composerOverlayBlocked,
    skillAutocompleteOpen:
      !options.showPalette &&
      !options.isBangInput &&
      Boolean(options.skillAutocompleteActive) &&
      !options.composerOverlayBlocked,
    pendingAttachments: options.isBangInput ? [] : options.pendingAttachmentPaths,
  }
}
