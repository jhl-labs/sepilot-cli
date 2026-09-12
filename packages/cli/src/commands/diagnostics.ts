import chalk from 'chalk'
import type {
  DaemonObservabilityRange,
  DaemonObservabilitySupportBundleInput,
  DaemonObservabilitySupportBundleResult,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { friendlyErrorMessage, printApiError } from '../utils/error-message.js'

export interface DiagnosticsBundleOptions {
  url?: string
  range?: string
  limit?: string
  events?: boolean
  crashes?: boolean
  feedback?: boolean
  health?: boolean
  persist?: boolean
}

function parsePositiveInteger(raw: string | undefined, name: string): number | undefined {
  if (raw === undefined || raw === '') return undefined
  const value = Number(raw)
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive integer`)
  }
  return value
}

function parseRange(raw: string | undefined): DaemonObservabilityRange | undefined {
  if (!raw) return undefined
  if (raw === '24h' || raw === '7d' || raw === '30d') return raw
  throw new Error('--range must be one of: 24h, 7d, 30d')
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`
}

function formatBundle(result: DaemonObservabilitySupportBundleResult): string {
  const lines = [
    chalk.bold('Diagnostic support bundle'),
    `Path: ${result.path ?? '(preview only)'}`,
    `Files: ${result.fileCount} (${formatBytes(result.totalBytes)})`,
    `Events: ${result.eventCount}  Crashes: ${result.crashCount}  Feedback: ${result.feedbackCount}`,
    `Redaction: ${result.redaction.level} (${result.redaction.rules.length} rules)`,
    '',
    'Contents:',
  ]
  for (const file of result.files) {
    lines.push(`  ${file.path.padEnd(32)} ${formatBytes(file.bytes)}`)
  }
  return lines.join('\n')
}

export async function diagnosticsBundleCommand(
  options: DiagnosticsBundleOptions = {},
): Promise<void> {
  try {
    const input: DaemonObservabilitySupportBundleInput = {
      range: parseRange(options.range),
      limit: parsePositiveInteger(options.limit, '--limit'),
      includeEvents: options.events,
      includeCrashes: options.crashes,
      includeFeedback: options.feedback,
      includeHealth: options.health,
      persist: options.persist,
    }
    const client = new DaemonClient(options.url)
    output(await client.exportObservabilitySupportBundle(input), formatBundle)
  } catch (err) {
    if (printApiError(err)) {
      process.exit(1)
    }
    console.error(chalk.red(`Failed to create diagnostic bundle: ${friendlyErrorMessage(err)}`))
    process.exit(1)
  }
}
