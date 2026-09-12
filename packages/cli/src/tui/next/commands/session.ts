import { resolve } from 'node:path'
import { buildSessionExportFilename, parseSessionExportArgs } from '../../utils/session-commands.js'

export interface SessionCommandDeps {
  sessionId: string | null
  messageCount: number
  provider: string
  model: string
  isStreaming: boolean
  cwd: string
  openPicker(): void
  inspect(sessionId: string): Promise<{ messageCount?: number }>
  activate(sessionId: string): Promise<void>
  branch(sessionId: string): Promise<{ branchId: string }>
  compact(sessionId: string): Promise<{ originalTokens: number; compactedTokens: number; savedTokens: number }>
  exportSession(sessionId: string, format: 'markdown' | 'json'): Promise<unknown>
  writeFile(path: string, content: string): Promise<void>
  deleteSession(sessionId: string): Promise<void>
  resetSession(): void
  setError(message: string): void
  showNotice(message: string): void
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export async function runSessionCommand(args: string, deps: SessionCommandDeps): Promise<void> {
  const [action, value] = args.split(/\s+/).filter(Boolean)
  if (!action) {
    deps.openPicker()
    return
  }
  if (action === 'current' || action === 'info') {
    if (!deps.sessionId) {
      deps.showNotice('Context Map\nSession  none\nNext     /resume or send a prompt')
      return
    }
    try {
      const session = await deps.inspect(deps.sessionId)
      deps.showNotice(`Session ${deps.sessionId}\nMessages ${session.messageCount ?? deps.messageCount}\nModel ${deps.provider}/${deps.model}`)
    } catch (error) {
      deps.setError(errorMessage(error))
    }
    return
  }
  if (!deps.sessionId && ['branch', 'compact', 'export', 'delete'].includes(action)) {
    deps.setError(`No active session to ${action}.`)
    return
  }
  try {
    if (action === 'branch') {
      const sourceId = deps.sessionId!
      const branch = await deps.branch(sourceId)
      await deps.activate(branch.branchId)
      deps.showNotice(`Branched session ${sourceId} → ${branch.branchId}.`)
      return
    }
    if (action === 'compact') {
      if (deps.isStreaming) {
        deps.setError('Cancel or finish the active run before compacting.')
        return
      }
      const result = await deps.compact(deps.sessionId!)
      deps.showNotice(`Compacted context: ${result.originalTokens.toLocaleString()} → ${result.compactedTokens.toLocaleString()} tokens (saved ${result.savedTokens.toLocaleString()}).`)
      return
    }
    if (action === 'export') {
      const { format, outputPath } = parseSessionExportArgs(args.split(/\s+/).slice(1))
      const exported = await deps.exportSession(deps.sessionId!, format)
      const targetPath = resolve(deps.cwd, outputPath ?? buildSessionExportFilename(deps.sessionId!, format))
      const serialized = typeof exported === 'string' ? exported : JSON.stringify(exported, null, 2)
      await deps.writeFile(targetPath, serialized)
      deps.showNotice(`Exported session ${deps.sessionId!.slice(0, 8)} to ${targetPath} (${format}).`)
      return
    }
    if (action === 'delete') {
      if (value !== 'confirm') {
        deps.setError('Use /session delete confirm to remove the active session.')
        return
      }
      await deps.deleteSession(deps.sessionId!)
      deps.resetSession()
      deps.showNotice('Deleted the active session.')
      return
    }
    await deps.activate(action)
  } catch (error) {
    deps.setError(errorMessage(error))
  }
}
