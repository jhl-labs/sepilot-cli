import type {
  OverlayOpenTarget,
  OverlayStateSnapshot,
} from '../utils/overlay-state.js'
import { getOverlayOpenBlockMessage } from '../utils/overlay-state.js'

type StreamApprovalBlockReason = 'streaming' | 'pending-approval'

export type StreamApprovalGateDecision =
  | { kind: 'block'; reason: StreamApprovalBlockReason; message: string }
  | { kind: 'proceed' }

export function decideStreamApprovalGate(opts: {
  isStreaming: boolean
  hasPendingApproval: boolean
  action: string
}): StreamApprovalGateDecision {
  if (opts.isStreaming) {
    return {
      kind: 'block',
      reason: 'streaming',
      message: `Wait for the current stream to finish before ${opts.action}.`,
    }
  }
  if (opts.hasPendingApproval) {
    return {
      kind: 'block',
      reason: 'pending-approval',
      message: `Resolve the pending approval before ${opts.action}.`,
    }
  }
  return { kind: 'proceed' }
}

type OpenOverlayBlockReason = StreamApprovalBlockReason | 'overlay'

export type OpenOverlayGateDecision =
  | { kind: 'block'; reason: OpenOverlayBlockReason; message: string }
  | { kind: 'proceed' }

export function decideOpenOverlayGate(opts: {
  isStreaming: boolean
  hasPendingApproval: boolean
  action: string
  overlayState: OverlayStateSnapshot
  target: OverlayOpenTarget
  allowFromModelPicker?: boolean
}): OpenOverlayGateDecision {
  const streamApproval = decideStreamApprovalGate({
    isStreaming: opts.isStreaming,
    hasPendingApproval: opts.hasPendingApproval,
    action: opts.action,
  })
  if (streamApproval.kind === 'block') return streamApproval

  const overlayMessage = getOverlayOpenBlockMessage(
    opts.overlayState,
    opts.target,
    { allowModelPicker: opts.allowFromModelPicker },
  )
  if (overlayMessage) {
    return { kind: 'block', reason: 'overlay', message: overlayMessage }
  }
  return { kind: 'proceed' }
}
