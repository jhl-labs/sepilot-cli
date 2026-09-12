// Single source of truth for the "is this a Node.js fs error with
// the given errno code?" type guard. Previously duplicated in 10
// places across daemon (audit-logger, auth, extension-tokens,
// extensions/registry, primary-agent-store, channel-pairing-
// persistence, runs/checkpoints/tool-executions stores,
// approval-decisions). Inline copies are easy to keep consistent
// today but drift over time — one of them quietly evolved to
// `(err as NodeJS.ErrnoException).code` while the others stuck
// with the explicit `'code' in err` shape.
//
// The helper is pure type narrowing — no fs/process imports —
// so it lives in utils/ alongside other small daemon-internal
// shims.

export function isNodeFsError(err: unknown, code: string): boolean {
  return (
    typeof err === 'object'
    && err !== null
    && 'code' in err
    && (err as { code?: unknown }).code === code
  )
}
