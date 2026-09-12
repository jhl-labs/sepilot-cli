import { resolve } from 'node:path'
import type { DaemonProject, DaemonUsageSummary } from '@sepilotd/api-client'

export interface ProjectCommandDeps {
  workspaceRoot: string
  sessionId: string | null
  currentProjectId: string | null
  currentProjectName: string | null
  fetchProjects(): Promise<DaemonProject[]>
  attachSession(projectId: string, sessionId: string): Promise<unknown>
  cacheProjects(projects: DaemonProject[]): void
  openPicker(): void
  select(project: DaemonProject | null): void
  setError(message: string): void
  showNotice(message: string): void
}

export async function runProjectCommand(args: string, deps: ProjectCommandDeps): Promise<void> {
  const query = args.trim()
  if (!query) {
    deps.openPicker()
    return
  }
  if (query === 'current' || query === 'info') {
    deps.showNotice(`Project: ${deps.currentProjectName ?? 'none'}${deps.currentProjectId ? ` (${deps.currentProjectId})` : ''}\nWorkspace: ${deps.workspaceRoot}`)
    return
  }
  if (query === 'none') {
    deps.select(null)
    deps.showNotice('Project selection cleared.')
    return
  }
  try {
    const projects = await deps.fetchProjects()
    deps.cacheProjects(projects)
    const normalized = query.toLowerCase()
    const selected = query === 'auto'
      ? projects.find((project) => project.workingDirectory && resolve(project.workingDirectory) === deps.workspaceRoot)
      : projects.find((project) => project.name.toLowerCase() === normalized || project.id.toLowerCase() === normalized)
    if (!selected) {
      deps.setError(query === 'auto' ? `No project matches workspace ${deps.workspaceRoot}.` : `Project not found: ${query}`)
      return
    }
    if (deps.sessionId && !selected.sessionIds.includes(deps.sessionId)) {
      await deps.attachSession(selected.id, deps.sessionId)
    }
    deps.select(selected)
    deps.showNotice(`Project selected: ${selected.name}`)
  } catch (error) {
    deps.setError(error instanceof Error ? error.message : String(error))
  }
}

export interface UsageCommandDeps {
  days: number
  summary: DaemonUsageSummary | null
  dashboardOpen: boolean
  dashboardOnTop: boolean
  setDays(days: number): void
  openDashboard(): void
  closeDashboard(): void
  setError(message: string): void
  showNotice(message: string): void
}

export function runUsageCommand(args: string, deps: UsageCommandDeps): void {
  const query = args.trim().toLowerCase()
  if (query === 'close' || query === 'hide' || query === 'off') {
    if (deps.dashboardOnTop) deps.closeDashboard()
    deps.showNotice('Usage dashboard closed.')
    return
  }
  if (query === 'current' || query === 'info') {
    deps.showNotice(`Usage window: last ${deps.days} day${deps.days === 1 ? '' : 's'}\nDashboard: ${deps.dashboardOpen ? 'open' : 'closed'}${deps.summary ? `\nTokens: ${deps.summary.inputTokens.toLocaleString()} in / ${deps.summary.outputTokens.toLocaleString()} out\nCost: $${deps.summary.costUsd.toFixed(4)}` : '\nTotals: not loaded'}`)
    return
  }
  if (query && !['open', 'show', 'refresh'].includes(query)) {
    const match = query.match(/^(\d+)d?$/)
    const days = match ? Number.parseInt(match[1], 10) : 0
    if (!days || days > 365) {
      deps.setError('Usage: /usage [current|close] or /usage <1-365>')
      return
    }
    deps.setDays(days)
  }
  deps.openDashboard()
}
