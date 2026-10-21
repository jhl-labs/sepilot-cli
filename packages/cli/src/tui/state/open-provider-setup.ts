import type { DaemonProviderInfo } from '@sepilotd/api-client'
import type { OverlayStateSnapshot } from '../utils/overlay-state.js'
import { getOverlayOpenBlockMessage } from '../utils/overlay-state.js'
import {
  PROVIDER_WIZARD_PRESETS,
  type ProviderWizardPreset,
} from '../../utils/provider-presets.js'
import { findBuiltinProviderPreset } from '../utils/provider-setup.js'

export interface OpenProviderSetupRequest {
  presetQuery?: string | null
  providerId?: string | null
  allowFromModelPicker?: boolean
}

type OpenProviderSetupBlockReason =
  | 'streaming'
  | 'pending-approval'
  | 'overlay'
  | 'invalid-preset'

export type OpenProviderSetupDecision =
  | { kind: 'block'; reason: OpenProviderSetupBlockReason; message: string }
  | {
      kind: 'proceed'
      requestedPreset: ProviderWizardPreset | null
      fallbackPreset: ProviderWizardPreset | null
    }

export function decideOpenProviderSetupRequest(opts: {
  request: OpenProviderSetupRequest
  isStreaming: boolean
  hasPendingApproval: boolean
  overlayState: OverlayStateSnapshot
  currentProvider: string
}): OpenProviderSetupDecision {
  const { request, isStreaming, hasPendingApproval, overlayState, currentProvider } = opts

  if (isStreaming) {
    return {
      kind: 'block',
      reason: 'streaming',
      message: 'Wait for the current stream to finish before changing providers.',
    }
  }

  if (hasPendingApproval) {
    return {
      kind: 'block',
      reason: 'pending-approval',
      message: 'Resolve the pending approval before changing providers.',
    }
  }

  const overlayMessage = getOverlayOpenBlockMessage(
    overlayState,
    'provider-setup',
    { allowModelPicker: request.allowFromModelPicker },
  )
  if (overlayMessage) {
    return { kind: 'block', reason: 'overlay', message: overlayMessage }
  }

  const requestedPreset = request.presetQuery
    ? findBuiltinProviderPreset(request.presetQuery)
    : null
  if (request.presetQuery && !requestedPreset) {
    const supported = PROVIDER_WIZARD_PRESETS.map((preset) => preset.type).join(', ')
    return {
      kind: 'block',
      reason: 'invalid-preset',
      message: `Unsupported provider "${request.presetQuery}". Use one of: ${supported}.`,
    }
  }

  const fallbackPreset = requestedPreset
    ?? findBuiltinProviderPreset(currentProvider)
    ?? PROVIDER_WIZARD_PRESETS[0]
    ?? null

  return { kind: 'proceed', requestedPreset, fallbackPreset }
}

type AutoOpenProviderSetupReason =
  | 'providers-empty'
  | 'provider-missing'
  | 'provider-unready'
  | 'model-missing'
  | 'model-unavailable'

export type AutoOpenProviderSetupDecision =
  | { kind: 'noop' }
  | {
      kind: 'open'
      key: string
      reason: AutoOpenProviderSetupReason
      providerId: string | null
      presetQuery: string | null
    }

function cleanOptionalId(value: string | null | undefined): string {
  return typeof value === 'string' ? value.trim() : ''
}

function modelIsChatCapable(model: DaemonProviderInfo['models'][number]): boolean {
  const capabilities = model.capabilities
  return capabilities?.embedding !== true || capabilities.toolUse === true
}

function buildAutoOpenProviderSetupDecision(opts: {
  daemonUrl: string
  reason: AutoOpenProviderSetupReason
  providerId: string
  modelId: string
  setupProviderId?: string | null
  openedKey: string | null
}): AutoOpenProviderSetupDecision {
  const key = [
    opts.daemonUrl,
    'provider-setup',
    opts.reason,
    opts.providerId || 'none',
    opts.modelId || 'none',
  ].join(':')

  if (opts.openedKey === key) {
    return { kind: 'noop' }
  }

  const setupProviderId = opts.setupProviderId === undefined
    ? opts.providerId || null
    : opts.setupProviderId
  const presetQuery = setupProviderId
    ? null
    : findBuiltinProviderPreset(opts.providerId)?.type ?? null

  return {
    kind: 'open',
    key,
    reason: opts.reason,
    providerId: setupProviderId,
    presetQuery,
  }
}

export function decideAutoOpenProviderSetup(opts: {
  daemonUrl: string
  providersLoaded: boolean
  providers: DaemonProviderInfo[]
  defaultProviderId?: string | null
  defaultModelId?: string | null
  initialSessionId?: string | null
  openedKey: string | null
}): AutoOpenProviderSetupDecision {
  if (!opts.providersLoaded) {
    return { kind: 'noop' }
  }

  const providerId = cleanOptionalId(opts.defaultProviderId)
  const modelId = cleanOptionalId(opts.defaultModelId)

  if (opts.providers.length === 0) {
    return buildAutoOpenProviderSetupDecision({
      daemonUrl: opts.daemonUrl,
      reason: 'providers-empty',
      providerId,
      modelId,
      setupProviderId: null,
      openedKey: opts.openedKey,
    })
  }

  if (!providerId) {
    return buildAutoOpenProviderSetupDecision({
      daemonUrl: opts.daemonUrl,
      reason: 'provider-missing',
      providerId,
      modelId,
      setupProviderId: null,
      openedKey: opts.openedKey,
    })
  }

  const provider = opts.providers.find((item) => item.id === providerId) ?? null
  if (!provider) {
    return buildAutoOpenProviderSetupDecision({
      daemonUrl: opts.daemonUrl,
      reason: 'provider-missing',
      providerId,
      modelId,
      setupProviderId: null,
      openedKey: opts.openedKey,
    })
  }

  if (provider.health.status !== 'ready') {
    return buildAutoOpenProviderSetupDecision({
      daemonUrl: opts.daemonUrl,
      reason: 'provider-unready',
      providerId,
      modelId,
      openedKey: opts.openedKey,
    })
  }

  const chatModels = provider.models.filter(modelIsChatCapable)
  if (!modelId || chatModels.length === 0) {
    return buildAutoOpenProviderSetupDecision({
      daemonUrl: opts.daemonUrl,
      reason: 'model-missing',
      providerId,
      modelId,
      openedKey: opts.openedKey,
    })
  }

  if (!chatModels.some((model) => model.id === modelId)) {
    return buildAutoOpenProviderSetupDecision({
      daemonUrl: opts.daemonUrl,
      reason: 'model-unavailable',
      providerId,
      modelId,
      openedKey: opts.openedKey,
    })
  }

  return { kind: 'noop' }
}

export function getAutoProviderSetupBannerMessage(
  reason: AutoOpenProviderSetupReason,
): string | null {
  if (reason === 'providers-empty') {
    return 'provider가 없습니다 — 초기 설정을 시작합니다.'
  }
  return null
}

export interface OpenProviderDeleteRequest {
  providerId: string
  allowFromModelPicker?: boolean
}

type OpenProviderDeleteBlockReason =
  | 'streaming'
  | 'pending-approval'
  | 'provider-setup-open'
  | 'overlay'

export type OpenProviderDeleteDecision =
  | { kind: 'block'; reason: OpenProviderDeleteBlockReason; message: string }
  | { kind: 'noop' }
  | { kind: 'proceed' }

export function decideOpenProviderDeleteConfirmation(opts: {
  request: OpenProviderDeleteRequest
  isStreaming: boolean
  hasPendingApproval: boolean
  providerSetupOpen: boolean
  providerDeleteOpen: boolean
  overlayState: OverlayStateSnapshot
}): OpenProviderDeleteDecision {
  const {
    request,
    isStreaming,
    hasPendingApproval,
    providerSetupOpen,
    providerDeleteOpen,
    overlayState,
  } = opts

  if (isStreaming) {
    return {
      kind: 'block',
      reason: 'streaming',
      message: 'Wait for the current stream to finish before deleting providers.',
    }
  }
  if (hasPendingApproval) {
    return {
      kind: 'block',
      reason: 'pending-approval',
      message: 'Resolve the pending approval before deleting providers.',
    }
  }
  if (providerSetupOpen) {
    return {
      kind: 'block',
      reason: 'provider-setup-open',
      message: 'Close provider setup before deleting providers.',
    }
  }
  if (providerDeleteOpen) {
    return { kind: 'noop' }
  }
  const overlayMessage = getOverlayOpenBlockMessage(
    overlayState,
    'provider-delete',
    { allowModelPicker: request.allowFromModelPicker },
  )
  if (overlayMessage) {
    return { kind: 'block', reason: 'overlay', message: overlayMessage }
  }
  return { kind: 'proceed' }
}
