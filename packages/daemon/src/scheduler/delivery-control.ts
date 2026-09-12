import type { AgentEvent } from '@sepilotd/core'

export type ScheduledDeliveryDisposition = 'deliver' | 'suppress'

function dispositionFromMetadata(
  metadata: Record<string, unknown> | undefined,
): ScheduledDeliveryDisposition | null {
  const value = metadata?.schedulerDelivery
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const control = value as Record<string, unknown>
  if (
    control.version !== 1
    || control.source !== 'monitor.evaluate'
    || (control.disposition !== 'deliver' && control.disposition !== 'suppress')
    || !Number.isInteger(control.stateVersion)
    || Number(control.stateVersion) < 0
  ) {
    return null
  }
  return control.disposition
}

/**
 * Capture a built-in monitor's structured delivery decision from one agent
 * run. Tool-call identity is joined locally so model prose and plugin metadata
 * cannot suppress a scheduled message by imitating the result shape.
 */
export class ScheduledDeliveryTracker {
  private readonly toolNames = new Map<string, string>()
  private disposition: ScheduledDeliveryDisposition | null = null

  consume(event: AgentEvent): void {
    if (event.type === 'tool_call') {
      this.toolNames.set(event.toolCall.id, event.toolCall.name)
      return
    }
    if (
      event.type !== 'tool_result'
      || event.status !== 'success'
      || this.toolNames.get(event.toolCallId) !== 'monitor.evaluate'
    ) {
      return
    }
    const disposition = dispositionFromMetadata(event.metadata)
    if (disposition) this.disposition = disposition
  }

  shouldSuppress(): boolean {
    return this.disposition === 'suppress'
  }
}
