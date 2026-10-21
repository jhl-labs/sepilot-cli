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
