import type { ManagedProcessExitEvent, ManagedProcessRegistry } from '../../tools/process.js'
import { SessionInbox } from './session-inbox.js'
import { publishStoredNotification } from '../../notifications/publish.js'
import { createLogger } from '../../logger.js'

const log = createLogger('process-exit-receipts')

function outcome(event: ManagedProcessExitEvent): string {
  if (event.terminationReason === 'ttl') return 'stopped after its lifetime ended'
  if (event.process.signal) return `ended by signal ${event.process.signal}`
  return `exited with code ${event.process.exitCode ?? 'unknown'}`
}

function shortCommand(command: string): string {
  const flat = command.replace(/\s+/g, ' ').trim()
  return flat.length > 80 ? `${flat.slice(0, 77)}…` : flat
}

export function receiptForProcessExit(event: ManagedProcessExitEvent): Parameters<SessionInbox['publish']>[0] | null {
  if (!event.notifyOnExit || !event.ownerSessionId) return null
  if (event.terminationReason === 'requested' || event.terminationReason === 'daemon-shutdown') return null
  const status = outcome(event)
  const tail = [
    event.stdoutTail.trim() ? `stdout (tail):\n${event.stdoutTail.trim()}` : '',
    event.stderrTail.trim() ? `stderr (tail):\n${event.stderrTail.trim()}` : '',
  ].filter(Boolean).join('\n\n')
  return {
    sessionId: event.ownerSessionId,
    source: 'process',
    sourceId: event.process.id,
    eventKey: `${event.process.id}:exit`,
    title: `Process ${shortCommand(event.process.command)} ${status}`,
    body: [
      `Background process ${event.process.id} (${event.process.command}) ${status}.`,
      'Read the complete retained output with process.read before drawing conclusions from this tail.',
      tail,
    ].filter(Boolean).join('\n\n'),
    wake: true,
  }
}

export function observeManagedProcessExits(
  registry: ManagedProcessRegistry,
  inbox: () => SessionInbox = () => new SessionInbox(),
): () => void {
  return registry.onExit((event) => {
    const receipt = receiptForProcessExit(event)
    if (!receipt) return
    inbox().tryPublish(receipt)
    try {
      publishStoredNotification({
        id: `process-exit:${event.process.id}`,
        title: receipt.title,
        body: `Session ${event.ownerSessionId}. Inspect with sepilot tasks logs process:${event.process.id}.`,
        topic: 'process-exit',
        audience: ['cli', 'desktop', 'web'],
      })
    } catch (error) {
      log.warn('process exit notification failed', { processId: event.process.id, error: String(error) })
    }
  })
}
