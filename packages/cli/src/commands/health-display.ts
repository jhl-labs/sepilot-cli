import chalk from 'chalk'

const healthyStatuses = new Set([
  'ok',
  'ready',
  'connected',
  'registered',
])

const warningStatuses = new Set([
  'degraded',
  'backfilling',
  'reindex_required',
  'connecting',
])

const neutralStatuses = new Set([
  'disabled',
])

export function formatHealthComponentIcon(status: string): string {
  if (healthyStatuses.has(status)) {
    return chalk.green('✓')
  }
  if (warningStatuses.has(status)) {
    return chalk.yellow('!')
  }
  if (neutralStatuses.has(status)) {
    return chalk.gray('-')
  }
  return chalk.red('✗')
}
