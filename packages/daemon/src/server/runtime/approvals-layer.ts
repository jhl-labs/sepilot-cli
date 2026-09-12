import { ApprovalRegistry } from './approvals.js'
import { ApprovalDecisionStore } from './approval-decisions.js'
import { ApprovalCheckpointStore } from './checkpoints.js'
import { RunCheckpointStore } from './runs.js'
import { ToolExecutionStore } from './tool-executions.js'
import type { WatchedSessionStore } from './session-watch.js'
import {
  DEFAULT_RESUME_ARTIFACT_RETENTION_DAYS,
  type SepilotdConfig,
} from '../../config/schema.js'
import { publishApprovalPendingNotification } from '../../notifications/publish.js'

const DAY_MS = 24 * 60 * 60 * 1000

interface ApprovalsLayerConfig {
  daemon?: {
    resumeArtifactRetentionDays?: number
  }
  notifications?: SepilotdConfig['notifications']
}

export interface ApprovalsLayer {
  approvalDecisions: ApprovalDecisionStore
  approvalRegistry: ApprovalRegistry
  approvalCheckpoints: ApprovalCheckpointStore
  runCheckpoints: RunCheckpointStore
  toolExecutions: ToolExecutionStore
}

/**
 * Approval + tool-execution persistence stack. Needs the session
 * store to scope approval requests. Every sub-store owns its own
 * directory under dataDir, so init() is awaited inside the
 * assembler to keep the caller single-shot.
 */
export async function assembleApprovalsLayer(args: {
  dataDir: string
  sessions: WatchedSessionStore
  config?: ApprovalsLayerConfig
}): Promise<ApprovalsLayer> {
  const { dataDir, sessions, config } = args
  const resumeArtifactRetentionMs = (
    config?.daemon?.resumeArtifactRetentionDays
    ?? DEFAULT_RESUME_ARTIFACT_RETENTION_DAYS
  ) * DAY_MS

  const approvalDecisions = new ApprovalDecisionStore({
    persistentPath: `${dataDir}/security/approvals.json`,
  })
  await approvalDecisions.initialize()
  const approvalRegistry = new ApprovalRegistry(sessions, approvalDecisions, {
    onPending: (approval) => {
      publishApprovalPendingNotification(config, { approval, outcome: 'pending' })
    },
    onTimeout: (approval) => {
      publishApprovalPendingNotification(config, { approval, outcome: 'expired' })
    },
  })
  const approvalCheckpoints = new ApprovalCheckpointStore(
    `${dataDir}/approval-checkpoints`,
  )
  await approvalCheckpoints.init()
  const runCheckpoints = new RunCheckpointStore(
    `${dataDir}/run-checkpoints`,
  )
  await runCheckpoints.init()
  const toolExecutions = new ToolExecutionStore(
    `${dataDir}/tool-executions`,
  )
  await toolExecutions.init()
  await Promise.all([
    approvalCheckpoints.pruneOlderThan(resumeArtifactRetentionMs),
    runCheckpoints.pruneOlderThan(resumeArtifactRetentionMs),
    toolExecutions.pruneOlderThan(resumeArtifactRetentionMs),
  ])

  return {
    approvalDecisions,
    approvalRegistry,
    approvalCheckpoints,
    runCheckpoints,
    toolExecutions,
  }
}
