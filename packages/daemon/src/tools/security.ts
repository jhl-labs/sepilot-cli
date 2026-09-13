import type { ToolSecurityDescriptor } from '@sepilotd/core'

/**
 * Canonical built-in tool security catalog.
 *
 * Tool effects used to be copied into independent read-only, plan-mode and
 * approval-guard name lists. That made a newly registered bookkeeping or
 * observation tool writable-by-default in one gate and blocked-by-default in
 * another. The registry now attaches this single descriptor to each built-in
 * tool and every policy gate consumes it. Unknown/plugin tools still fail
 * closed unless their registration declares an explicit descriptor.
 */
const descriptors = new Map<string, ToolSecurityDescriptor>()

function register(
  effect: ToolSecurityDescriptor['effect'],
  rationale: string,
  names: readonly string[],
  options: Pick<ToolSecurityDescriptor, 'mutationResultBoundary'> = {},
): void {
  for (const name of names) {
    if (descriptors.has(name)) {
      throw new Error(`Duplicate built-in tool security descriptor: ${name}`)
    }
    descriptors.set(name, { effect, rationale, ...options })
  }
}

register('observe', 'Reads information without a durable user-directed mutation', [
  'fs.read',
  'fs.list',
  'fs.glob',
  'fs.search',
  'git.status',
  'git.diff',
  'git.log',
  'gitea.actions.runs.inspect',
  'web.search',
  'webfetch',
  'jpad.workspaces',
  'jpad.pages.list',
  'jpad.pages.get',
  'market.quote',
  'browser.remote_snapshot',
  'browser.navigate',
  'browser.extract',
  'browser.screenshot',
  'computer.list_windows',
  'computer.list_elements',
  'computer.observe',
  'computer.wait',
  'assistant.status',
  'system.info',
  'self.info',
  'usage.report',
  'process.list',
  'process.sessions',
  'process.read',
  'process.follow',
  'process.wait',
  'service.list',
  'service.status',
  'service.logs',
  'service.native.status',
  'service.native.logs',
  'skill',
  'skillhub.search',
  'lsp',
  'code.diagnostics',
  'code.symbols',
  'code.dependencies',
  'doc.get',
  'doc.outline',
  'notebook.inspect',
  'apps.list',
  'apps.read',
  'apps.search',
  'office.list_open_documents',
  'office.read_active',
  'office.read_selection',
  'office.preview_edit',
  'office.read_slide',
  'office.capture_slide',
  'media.inspect',
  'media.extract_text',
  'media.transcribe',
  'image_gen.providers',
  'image_gen.job',
  'image_gen.file',
  'pages.scan',
  'pages.status',
  'schedule_list',
  'schedule_get',
  'schedule_runs',
  'knowledge.search',
  'knowledge.read',
  'memory.list',
  'memory.search',
  'memory.graph.search',
  'memory.graph.neighbors',
  'memory.graph.page',
  'memory.graph.audit',
  'memory.daily.read',
  'memory.journal.inspect',
  'memory.daily.list',
  'memory.daily.search',
  'memory.documents.search',
  'memory.documents.list',
  'memory.documents.get',
  'memory.documents.preview',
  'memory.audit',
  'memory.reminders.list',
  'memory.export',
  'memory.tag.suggest',
  'memory.diff',
  'memory.usage',
  'memory.context.snapshot',
  'memory.conflicts.find',
  'memory.search.related',
  'memory.search.by_tag',
  'memory.history',
  'memory.tag.list',
  'memory.summarize',
  'memory.pinned.list',
  'monitor.report',
])

register(
  'internal-state',
  'Updates agent-owned progress metadata without producing evidence required by a workspace edit',
  ['todowrite'],
  { mutationResultBoundary: 'commutative' },
)

register('internal-state', 'Mutates bounded agent-owned control metadata only', [
  'question',
  'monitor.evaluate',
  'subagent.dispatch',
  'subagent.job',
  'doc.diff_preview',
  'process.stop',
  'memory.access.hot',
  'memory.access.touch',
  'memory.pin',
  'memory.unpin',
])

register('workspace-write', 'Writes files or the active document in the user workspace', [
  'workspace.prepare',
  'fs.write',
  'fs.append',
  'fs.edit',
  'fs.move',
  'apply_patch',
  'doc.replace_section',
  'doc.replace_range',
  'doc.insert_after_section',
  'doc.append',
  'doc.rewrite',
  'pages.scaffold',
])

register('external-write', 'Changes durable agent, application, browser, or remote-system state', [
  'notification.publish',
  'jpad.pages.create',
  'jpad.pages.update',
  'a2a.send',
  'apps.write',
  'apps.mutate',
  'browser.remote_action',
  'browser.click',
  'browser.fill',
  'browser.evaluate',
  'computer.launch_app',
  'computer.open_url',
  'computer.focus_window',
  'computer.move_mouse',
  'computer.click',
  'computer.drag',
  'computer.type_text',
  'computer.hotkey',
  'computer.scroll',
  'device.delegate',
  'image_gen.create',
  'media.speak',
  'office.open_presentation',
  'office.navigate_slide',
  'office.apply_edit',
  'office.replace_selection',
  'schedule_create',
  'schedule_update',
  'schedule_pause',
  'schedule_resume',
  'schedule_cancel',
  'schedule_cancel_all',
  'schedule_run_now',
  'skillhub.install',
  'memory.daily.append',
  'memory.daily.replace',
  'memory.journal.manage',
  'memory.documents.delete',
  'memory.documents.ingest',
  'memory.documents.update',
  'memory.forget',
  'memory.graph.repair',
  'memory.graph.repair.apply',
  'memory.import',
  'memory.maintenance',
  'memory.merge',
  'knowledge.save',
  'knowledge.edit',
  'knowledge.review',
  'memory.remember',
  'memory.remind_at',
  'memory.reminders.cancel',
  'memory.section.replace',
  'memory.tag.rename',
  'memory.update',
])

register('process-lifecycle', 'Starts, stops, signals, or reconfigures a managed execution', [
  'process.write',
  'external_acp.run',
  'image_gen.cancel',
  'process.signal',
  'service.disable',
  'service.enable',
  'service.healthcheck',
  'service.install',
  'service.remove',
  'service.restart',
  'service.start',
  'service.stop',
  'service.uninstall',
])

register('dynamic', 'Effect is derived from validated invocation arguments and enforced execution capabilities', [
  'terminal.run',
  'process.start',
])

export const UNKNOWN_TOOL_SECURITY: ToolSecurityDescriptor = {
  effect: 'unknown',
  rationale: 'No audited security descriptor is registered; fail closed',
}

export function builtinToolSecurityDescriptor(
  name: string,
): ToolSecurityDescriptor | undefined {
  return descriptors.get(name)
}

export function resolveToolSecurityDescriptor(
  name: string,
  declared?: ToolSecurityDescriptor,
): ToolSecurityDescriptor {
  return declared ?? builtinToolSecurityDescriptor(name) ?? UNKNOWN_TOOL_SECURITY
}

export function listClassifiedBuiltinToolNames(): string[] {
  return [...descriptors.keys()].sort()
}
