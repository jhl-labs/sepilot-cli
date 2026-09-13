import type { DaemonClient } from '../client/http.js'
import { DaemonClient as Client } from '../client/http.js'
import { output } from '../output/formatter.js'

export type TasksClient = Pick<DaemonClient, 'listWork' | 'listSessionInbox' | 'acknowledgeSessionInbox' | 'readManagedProcess' | 'writeManagedProcess' | 'stopManagedProcess' | 'cancelActiveRun' | 'cancelBackgroundChat' | 'backgroundChatStatus' | 'pauseSchedulerJob' | 'stopService' | 'sessionApprovals' | 'jobs'>

export async function inspectTask(client: TasksClient, key: string): Promise<unknown> {
  const split = key.indexOf(':')
  const kind = key.slice(0, split)
  const id = key.slice(split + 1)
  if (split < 1 || !id) throw new Error('Expected <kind>:<id> from tasks list')
  if (kind === 'job') return { job: await client.jobs.get(id), results: await client.jobs.getItems(id, 0) }
  if (kind === 'chat') return client.backgroundChatStatus(id)
  if (kind === 'process') return client.readManagedProcess(id)
  const item = (await client.listWork()).items.find(item => item.key === key)
  if (!item) throw new Error('Task not found')
  if (kind === 'approval' && item.sessionId) return (await client.sessionApprovals(item.sessionId)).find(approval => approval.requestId === id)
  return item
}

export async function cancelTask(client: TasksClient, key: string): Promise<unknown> {
  const split = key.indexOf(':')
  const kind = key.slice(0, split)
  const id = key.slice(split + 1)
  if (split < 1 || !id) throw new Error('Expected <kind>:<id> from tasks list')
  if (kind === 'job') { await client.jobs.cancel(id); return client.jobs.get(id) }
  if (kind === 'run') return client.cancelActiveRun(id)
  if (kind === 'chat') return client.cancelBackgroundChat(id)
  if (kind === 'process') return client.stopManagedProcess(id)
  // Pausing preserves the recurring task and its evidence. It does not claim
  // to cancel an already active run; that run has its own entry.
  if (kind === 'schedule') return client.pauseSchedulerJob(id)
  if (kind === 'service') return client.stopService(id)
  throw new Error('Use the approval dialog to review and deny an approval; task cancellation never grants a decision')
}

export async function runTasksCommand(client: TasksClient, args: string[] = []): Promise<unknown> {
  const action = args[0] ?? 'list'
  if (action === 'send' && args[1] && args[2] !== undefined) return client.writeManagedProcess(args[1], args.slice(2).join(' '))
  if (action === 'eof' && args[1]) return client.writeManagedProcess(args[1], '', true)
  if (action === 'status' && args[1]) return client.jobs.get(args[1])
  if (action === 'items' && args[1]) return client.jobs.getItems(args[1], 0)
  if (action === 'list' && args.length > 1) {
    const flags: Record<string, string | number> = {}
    for (let i = 1; i < args.length; i += 2) {
      const name = args[i]?.replace(/^--/, '')
      const value = args[i + 1]
      if (!name || !['kind', 'status', 'limit', 'offset'].includes(name) || !value) throw new Error('Invalid task list filter')
      flags[name] = name === 'limit' || name === 'offset' ? Number(value) : value
    }
    return client.jobs.list(flags)
  }
  if (action === 'list') return client.listWork()
  if (action === 'inspect' && args[1]) return inspectTask(client, args[1])
  if (action === 'cancel' && args[1]) return cancelTask(client, args[1])
  if (action === 'logs' && args[1]) return client.readManagedProcess(args[1])
  if (action === 'inbox' && args[1]) return client.listSessionInbox(args[1], { unreadOnly: args[2] !== 'all' })
  if (action === 'ack' && args[1] && /^[1-9]\d*$/.test(args[2] ?? '')) return client.acknowledgeSessionInbox(args[1], Number(args[2]))
  throw new Error('Usage: tasks [list|inspect <kind:id>|cancel <kind:id>|logs <process-id>|send <process-id> <text>|eof <process-id>|inbox <session-id> [all]|ack <session-id> <receipt>]')
}

export async function tasksCommand(args: string[], options: { url?: string }): Promise<void> {
  const client = new Client(options.url)
  if (args[0] === 'logs' && args[1] && args.includes('--follow')) {
    let stopped = false
    const stop = () => { stopped = true }
    process.once('SIGINT', stop)
    let stdoutOffset = 0
    let stderrOffset = 0
    try {
      while (!stopped) {
        const result = await client.readManagedProcess(args[1], { stdoutOffset, stderrOffset })
        stdoutOffset = result.nextStdoutOffset
        stderrOffset = result.nextStderrOffset
        if (result.stdout || result.stderr || result.status !== 'running') output(result, value => value.screen ?? `${value.stdout}${value.stderr}`)
        if (result.status !== 'running') break
        await new Promise(resolve => setTimeout(resolve, 500))
      }
    } finally { process.off('SIGINT', stop) }
    return // Leaving observation never stops the managed process.
  }
  const result = await runTasksCommand(client, args)
  output(result, value => {
    if (value && typeof value === 'object' && 'items' in value) {
      const list = value as Awaited<ReturnType<DaemonClient['listWork']>>
      if ('unavailable' in list) return [
        ...list.items.map(item => `${item.key}  ${item.status}  ${item.title}\n  ${item.detail}\n  ${item.action}`),
        ...(list.unavailable.length ? [`Unavailable sources: ${list.unavailable.join(', ')}`] : []),
        ...(list.truncated.length ? [`Bounded sources: ${list.truncated.join(', ')}; use their dedicated lists for older work.`] : []),
      ].join('\n') || 'No tasks.'
    }
    return JSON.stringify(value, null, 2)
  })
}
