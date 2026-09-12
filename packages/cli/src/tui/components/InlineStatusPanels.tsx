import type {
  ActivityItem,
  DaemonArtifact,
  DaemonDailyUsageSummary,
  DaemonMemoryEntry,
  DaemonMemorySemanticStatus,
  DaemonRagFolder,
  DaemonRagSearchHit,
  DaemonRagSyncResult,
  DaemonRagVectorDbInfo,
  DaemonUsageSummary,
  DebateRoundSummary,
  EditCheckpointSummary,
  PlannerWorkingMemory,
} from '@sepilotd/api-client'
import type { ToolCallState } from '../types.js'
import { ArtifactPanel } from './ArtifactPanel.js'
import { DebatePanel } from './DebatePanel.js'
import { EditRollbackPanel } from './EditRollbackPanel.js'
import { McpStatusPanel, type McpStatusPanelClient } from './McpStatusPanel.js'
import { MemorySearchPanel } from './MemorySearchPanel.js'
import { PlannerProgressPanel } from './PlannerProgressPanel.js'
import { RagPanel } from './RagPanel.js'
import { TodoProgressPanel, type TodoProgressItem } from './TodoProgressPanel.js'
import { ToolActivityPanel } from './ToolActivityPanel.js'
import { UsageDashboard } from './UsageDashboard.js'

interface InlineStatusPanelsProps {
  showToolActivityPanel: boolean
  activities: ActivityItem[]
  toolCalls: ToolCallState[]
  toolActivityHeight: number
  toolActivityMaxVisibleItems?: number
  runActive?: boolean
  showMemorySearchPanel: boolean
  showRagPanel: boolean
  ragQuery: string
  ragSources: DaemonRagFolder[]
  ragHits: DaemonRagSearchHit[]
  ragVectorInfo: DaemonRagVectorDbInfo | null
  ragSyncResult: DaemonRagSyncResult | null
  ragLoading: boolean
  ragError: string | null
  ragSelectedIndex: number
  ragPanelHeight: number
  onRagSelectIndex: (index: number) => void
  onRagSync: () => void
  onRagClose: () => void
  memorySearchQuery: string
  memorySearchResults: DaemonMemoryEntry[]
  memorySearchStatus: DaemonMemorySemanticStatus | null
  memorySearchLoading: boolean
  memorySearchError: string | null
  memorySearchHeight: number
  showUsageDashboardPanel: boolean
  usageSummary: DaemonUsageSummary | null
  usageDaily: DaemonDailyUsageSummary[]
  usageDashboardDays: number
  usageDashboardLoading: boolean
  usageDashboardError: string | null
  usageDashboardHeight: number
  showMcpPanel: boolean
  mcpPanelHeight: number
  mcpClient?: McpStatusPanelClient | null
  showArtifactPanel: boolean
  artifacts: DaemonArtifact[]
  artifactPanelHeight: number
  plannerWorkingMemory?: PlannerWorkingMemory | null
  showPlannerPanel?: boolean
  plannerPanelHeight?: number
  /** Live todo items from the `state_board` stream frame (shown while a run is active). */
  stateBoardTodos?: TodoProgressItem[] | null
  editRollbacks?: EditCheckpointSummary[]
  showEditRollbackPanel?: boolean
  debateRounds?: DebateRoundSummary[]
  showDebatePanel?: boolean
}

export function InlineStatusPanels({
  showToolActivityPanel,
  activities,
  toolCalls,
  toolActivityHeight,
  toolActivityMaxVisibleItems = 3,
  runActive = false,
  showMemorySearchPanel,
  showRagPanel,
  ragQuery,
  ragSources,
  ragHits,
  ragVectorInfo,
  ragSyncResult,
  ragLoading,
  ragError,
  ragSelectedIndex,
  ragPanelHeight,
  onRagSelectIndex,
  onRagSync,
  onRagClose,
  memorySearchQuery,
  memorySearchResults,
  memorySearchStatus,
  memorySearchLoading,
  memorySearchError,
  memorySearchHeight,
  showUsageDashboardPanel,
  usageSummary,
  usageDaily,
  usageDashboardDays,
  usageDashboardLoading,
  usageDashboardError,
  usageDashboardHeight,
  showMcpPanel,
  mcpPanelHeight,
  mcpClient = null,
  showArtifactPanel,
  artifacts,
  artifactPanelHeight,
  plannerWorkingMemory,
  showPlannerPanel = true,
  plannerPanelHeight,
  stateBoardTodos,
  editRollbacks,
  showEditRollbackPanel = true,
  debateRounds,
  showDebatePanel = true,
}: InlineStatusPanelsProps) {
  return (
    <>
      {showToolActivityPanel && toolActivityHeight > 0 && (
        <ToolActivityPanel
          activities={activities}
          toolCalls={toolCalls}
          height={toolActivityHeight}
          maxVisibleItems={toolActivityMaxVisibleItems}
          runActive={runActive}
        />
      )}
      {showRagPanel && ragPanelHeight > 0 && (
        <RagPanel
          query={ragQuery}
          sources={ragSources}
          hits={ragHits}
          vectorInfo={ragVectorInfo}
          syncResult={ragSyncResult}
          loading={ragLoading}
          error={ragError}
          selectedIndex={ragSelectedIndex}
          height={ragPanelHeight}
          onSelectIndex={onRagSelectIndex}
          onSync={onRagSync}
          onClose={onRagClose}
        />
      )}
      {showMemorySearchPanel && (
        <MemorySearchPanel
          query={memorySearchQuery}
          results={memorySearchResults}
          status={memorySearchStatus}
          loading={memorySearchLoading}
          error={memorySearchError}
          height={memorySearchHeight}
        />
      )}
      {showUsageDashboardPanel && (
        <UsageDashboard
          summary={usageSummary}
          daily={usageDaily}
          days={usageDashboardDays}
          loading={usageDashboardLoading}
          error={usageDashboardError}
          height={usageDashboardHeight}
        />
      )}
      {showMcpPanel && mcpPanelHeight > 0 && (
        <McpStatusPanel client={mcpClient} />
      )}
      {showArtifactPanel && (
        <ArtifactPanel
          artifacts={artifacts}
          height={artifactPanelHeight}
        />
      )}
      {showPlannerPanel && plannerWorkingMemory
        ? (
            <PlannerProgressPanel
              workingMemory={plannerWorkingMemory}
              height={plannerPanelHeight}
            />
          )
        : null}
      {runActive && stateBoardTodos && stateBoardTodos.length > 0
        ? <TodoProgressPanel todos={stateBoardTodos} />
        : null}
      {showEditRollbackPanel && editRollbacks && editRollbacks.length > 0
        ? <EditRollbackPanel checkpoints={editRollbacks} />
        : null}
      {showDebatePanel && debateRounds && debateRounds.length > 0
        ? <DebatePanel rounds={debateRounds} />
        : null}
    </>
  )
}
