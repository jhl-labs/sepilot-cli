import type { DaemonSessionDetail } from './types.js'

export function resumableRunCopy(
  resumableRun?: DaemonSessionDetail['resumableRun'],
): string {
  if (!resumableRun) {
    return 'No run checkpoint was saved (the turn stopped outside a tool run). The conversation itself is intact — send a new message to continue.'
  }

  const target = formatResumableRunTarget(resumableRun)
  const replaySafeSuffix = hasMultipleReplayTargets(resumableRun)
    ? 'Each call is marked replay-safe.'
    : 'It is marked replay-safe.'

  if (resumableRun.journaledResultAvailable) {
    return `The daemon preserved the completed result from ${target}. Resume continues without replay.`
  }

  if (resumableRun.recoveryProbeAvailable) {
    return `The daemon can verify whether ${target} already completed before replay. If recovery succeeds, the run continues without replay.`
  }

  if (resumableRun.mode === 'replay-risky') {
    return `The daemon disconnected with ${target} in progress. Resuming may replay ${target}, so replay needs explicit confirmation.`
  }

  if (resumableRun.mode === 'replay-safe') {
    return `The daemon disconnected with ${target} in progress. Resuming may replay ${target}. ${replaySafeSuffix}`
  }

  switch (resumableRun.stage) {
    case 'acting':
      return 'The daemon saved a checkpoint at the tool boundary. Resume from there instead of sending a new prompt.'
    case 'observing':
      return 'The daemon saved a checkpoint after tool execution. Resume the interrupted run without rewriting the task.'
    default:
      return 'The daemon saved a checkpoint before the next model step. Resume the interrupted run from there.'
  }
}

export function resumableRunActionLabel(
  resumableRun?: DaemonSessionDetail['resumableRun'],
): string {
  const multipleTargets = hasMultipleReplayTargets(resumableRun)

  switch (resumableRun?.mode) {
    case 'replay-risky':
      if (resumableRun.recoveryProbeAvailable) {
        return 'Verify And Resume'
      }
      return multipleTargets ? 'Replay Calls And Resume' : 'Replay Tool And Resume'
    case 'replay-safe':
      return 'Resume With Replay'
    default:
      return 'Resume Run'
  }
}

export function resumableRunActivityDetail(
  resumableRun?: DaemonSessionDetail['resumableRun'],
): string {
  if (!resumableRun) {
    return 'Recovering from saved checkpoint'
  }

  switch (resumableRun.mode) {
    case 'exact':
      if (resumableRun.journaledResultAvailable) {
        return `Recovering with preserved result from ${formatResumableRunTarget(resumableRun)}`
      }
      return `Recovering from ${resumableRun.stage} checkpoint`
    case 'replay-risky':
      if (resumableRun.recoveryProbeAvailable) {
        return `Attempting recovery for ${formatResumableRunTarget(resumableRun)} before any replay`
      }
      return `Recovering with explicit replay of ${formatResumableRunTarget(resumableRun)}`
    case 'replay-safe':
      return `Recovering with replay of ${formatResumableRunTarget(resumableRun)}`
  }
}

function hasMultipleReplayTargets(
  resumableRun?: DaemonSessionDetail['resumableRun'],
): boolean {
  const toolCountFromNames = resumableRun?.currentTools?.length ?? 0
  const currentToolCount =
    resumableRun?.currentToolCount
    ?? (toolCountFromNames > 0 ? toolCountFromNames : (resumableRun?.currentTool ? 1 : 0))
  return currentToolCount > 1
}

function formatResumableRunTarget(
  resumableRun?: DaemonSessionDetail['resumableRun'],
): string {
  const currentTools = resumableRun?.currentTools
    ?? (resumableRun?.currentTool ? [resumableRun.currentTool] : [])
  const currentToolCount =
    resumableRun?.currentToolCount
    ?? (currentTools.length > 0 ? currentTools.length : (resumableRun?.currentTool ? 1 : 0))

  if (currentToolCount <= 1) {
    return resumableRun?.currentTool ?? currentTools[0] ?? 'the active tool'
  }

  const toolCounts = new Map<string, number>()
  for (const toolName of currentTools) {
    toolCounts.set(toolName, (toolCounts.get(toolName) ?? 0) + 1)
  }

  const summary =
    toolCounts.size > 0
      ? ` (${Array.from(toolCounts.entries())
          .slice(0, 2)
          .map(([toolName, count]) => count > 1 ? `${toolName} x${count}` : toolName)
          .join(', ')}${toolCounts.size > 2 ? `, +${toolCounts.size - 2} more` : ''})`
      : ''

  return `${currentToolCount} tool calls${summary}`
}
