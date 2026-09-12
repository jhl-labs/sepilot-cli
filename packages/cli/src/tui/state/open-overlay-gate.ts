// Pulled out of App.tsx so the open-overlay and other gated
// callbacks share one named pre-condition instead of each
// duplicating the same templated guard returns inline. Two helpers:
//
//   - decideStreamApprovalGate:   streaming + pending-approval only
//                                 (rewind, branch, set-as-default).
//   - decideOpenOverlayGate:      streaming + pending-approval +
//                                 overlay-block (the five picker
//                                 openers).
//
// Both are pure. Callers translate `block` outcomes into a
// SYSTEM_MESSAGE dispatch and an early return; `proceed` means the
// caller can run the actual side effects.

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
  // Gerund or noun phrase that completes "Wait for the current
  // stream to finish before ___." and "Resolve the pending approval
  // before ___." e.g. "switching sessions", "rewinding".
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
