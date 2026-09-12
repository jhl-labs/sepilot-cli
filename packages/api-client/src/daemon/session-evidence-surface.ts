import type {
  DaemonSessionDetail,
  DaemonSessionEvaluationGateStatus,
  DaemonSessionEvaluationStageStatus,
  DaemonSessionEvidenceArtifactStatus,
  DaemonSessionEvidenceManifestStatus,
} from './types.js'

export type SessionEvidenceSurfaceState =
  | 'in_progress'
  | 'ready'
  | 'passed'
  | 'attention_needed'

export interface SessionEvidenceSurfaceReadback {
  id: string
  label: string
  status: DaemonSessionEvidenceArtifactStatus
  timestamp: string
  path?: string
  tool?: string
  hash?: string
}

export interface SessionEvidenceSurfaceModel {
  state: SessionEvidenceSurfaceState
  manifestStatus: DaemonSessionEvidenceManifestStatus | null
  evaluationStatus: DaemonSessionEvaluationGateStatus | null
  approved: boolean | null
  verdictReason: string | null
  summary: {
    artifacts: number
    validationRuns: number
    validationFailures: number
    filesChanged: number
    acceptanceCriteria: number
    supportedAcceptanceCriteria: number
    failedAcceptanceCriteria: number
    blockedAcceptanceCriteria: number
    unverifiedAcceptanceCriteria: number
  }
  stages: {
    mechanical: DaemonSessionEvaluationStageStatus
    semantic: DaemonSessionEvaluationStageStatus
    consensus: DaemonSessionEvaluationStageStatus
  } | null
  readbacks: SessionEvidenceSurfaceReadback[]
}

type SessionEvidenceSurfaceInput = Pick<
  DaemonSessionDetail,
  'evidenceManifest' | 'evaluationGate'
>

function isTerminalEvaluation(status: DaemonSessionEvaluationGateStatus): boolean {
  return status === 'passed' || status === 'failed' || status === 'blocked' || status === 'unverified'
}

function resolveSurfaceState(
  manifestStatus: DaemonSessionEvidenceManifestStatus | null,
  evaluationStatus: DaemonSessionEvaluationGateStatus | null,
  approved: boolean | null,
): SessionEvidenceSurfaceState {
  if (evaluationStatus && isTerminalEvaluation(evaluationStatus)) {
    return evaluationStatus === 'passed' && approved === true ? 'passed' : 'attention_needed'
  }
  if (manifestStatus === 'attention_needed') return 'attention_needed'
  if (evaluationStatus) return 'in_progress'
  return manifestStatus === 'ready' ? 'ready' : 'in_progress'
}

/**
 * Projects daemon-owned evidence into one surface-neutral view model.
 *
 * The daemon's verdict boolean is false while a gate is still running, so it
 * is only exposed after a terminal evaluation status. This prevents clients
 * from presenting an in-progress run as a failed or rejected one.
 */
export function buildSessionEvidenceSurfaceModel(
  input: SessionEvidenceSurfaceInput | null | undefined,
): SessionEvidenceSurfaceModel | null {
  const manifest = input?.evidenceManifest ?? null
  const evaluation = input?.evaluationGate ?? null
  if (!manifest && !evaluation) return null

  const terminalEvaluation = evaluation && isTerminalEvaluation(evaluation.status)
  const approved = terminalEvaluation ? evaluation.verdict.approved : null
  const manifestSummary = manifest?.summary
  const evaluationSignals = evaluation?.signals

  return {
    state: resolveSurfaceState(
      manifest?.status ?? null,
      evaluation?.status ?? null,
      approved,
    ),
    manifestStatus: manifest?.status ?? null,
    evaluationStatus: evaluation?.status ?? null,
    approved,
    verdictReason: terminalEvaluation ? evaluation.verdict.reason : null,
    summary: {
      artifacts: manifestSummary?.artifacts ?? 0,
      validationRuns: manifestSummary?.validationRuns ?? evaluationSignals?.validationRuns ?? 0,
      validationFailures:
        manifestSummary?.validationFailures ?? evaluationSignals?.validationFailures ?? 0,
      filesChanged: manifestSummary?.filesChanged ?? evaluationSignals?.fileChanges ?? 0,
      acceptanceCriteria:
        manifestSummary?.acceptanceCriteria ?? evaluationSignals?.acceptanceCriteria ?? 0,
      supportedAcceptanceCriteria:
        manifestSummary?.supportedAcceptanceCriteria ??
        evaluationSignals?.supportedAcceptanceCriteria ??
        0,
      failedAcceptanceCriteria:
        manifestSummary?.failedAcceptanceCriteria ??
        evaluationSignals?.failedAcceptanceCriteria ??
        0,
      blockedAcceptanceCriteria:
        manifestSummary?.blockedAcceptanceCriteria ??
        evaluationSignals?.blockedAcceptanceCriteria ??
        0,
      unverifiedAcceptanceCriteria:
        manifestSummary?.unverifiedAcceptanceCriteria ??
        evaluationSignals?.unverifiedAcceptanceCriteria ??
        0,
    },
    stages: evaluation
      ? {
          mechanical: evaluation.stages.mechanical.status,
          semantic: evaluation.stages.semantic.status,
          consensus: evaluation.stages.consensus.status,
        }
      : null,
    readbacks: (manifest?.artifacts ?? [])
      .filter((artifact) => artifact.kind === 'artifact_readback')
      .map((artifact) => ({
        id: artifact.id,
        label: artifact.label,
        status: artifact.status,
        timestamp: artifact.timestamp,
        ...(artifact.path ? { path: artifact.path } : {}),
        ...(artifact.tool ? { tool: artifact.tool } : {}),
        ...(artifact.hash ? { hash: artifact.hash } : {}),
      })),
  }
}
