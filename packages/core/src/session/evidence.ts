/**
 * Public session-evidence artifact kinds shared by the daemon schema and every
 * client surface. Keep this runtime tuple as the single source of truth so a
 * new daemon artifact cannot silently fall outside CLI/Desktop/Mobile types.
 */
export const SESSION_EVIDENCE_ARTIFACT_KINDS = [
  'run_contract',
  'completion_verdict',
  'observation',
  'action_receipt',
  'validation',
  'file_change',
  'artifact_readback',
  'approval',
  'todo',
  'assistant_claim',
  'post_edit_findings',
  'session_end',
] as const

export type SessionEvidenceArtifactKind =
  (typeof SESSION_EVIDENCE_ARTIFACT_KINDS)[number]
