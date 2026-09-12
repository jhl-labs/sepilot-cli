import chalk from 'chalk'
import type {
  ExternalAcpAgentName,
  ExternalAcpDispatchInput,
  ExternalAcpDispatchResult,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export interface AcpAgentRunOptions {
  url?: string
  agent?: string
  cwd?: string
  session?: string
  timeoutMs?: string
}

export interface AcpAgentClientLike {
  runExternalAcpAgent(input: ExternalAcpDispatchInput): Promise<ExternalAcpDispatchResult>
}

function parseTimeout(raw?: string): number | undefined {
  if (!raw) return undefined
  const value = Number.parseInt(raw, 10)
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`--timeout-ms must be a positive integer (got ${JSON.stringify(raw)})`)
  }
  return value
}

function parseAgent(raw?: string): ExternalAcpAgentName {
  if (raw === undefined || raw === 'opencode' || raw === 'codex') {
    return raw ?? 'opencode'
  }
  throw new Error(`unsupported external ACP agent: ${raw}`)
}

function emitResult(result: ExternalAcpDispatchResult): void {
  output(result, (r) => {
    const head = `${chalk.cyan(`[external-acp ${r.agent}]`)} session=${r.sessionId} external=${r.externalSessionId || '-'} status=${r.status} stop=${r.stopReason}`
    return r.output ? `${head}\n\n${r.output}` : head
  })
}

export async function runAcpAgent(
  prompt: string,
  options: AcpAgentRunOptions,
  client: AcpAgentClientLike,
): Promise<ExternalAcpDispatchResult> {
  const result = await client.runExternalAcpAgent({
    prompt,
    agent: parseAgent(options.agent),
    cwd: options.cwd,
    sessionId: options.session,
    timeoutMs: parseTimeout(options.timeoutMs),
  })
  emitResult(result)
  return result
}

export async function acpAgentRunCommand(
  prompt: string,
  options: AcpAgentRunOptions,
): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    const result = await runAcpAgent(prompt, options, client)
    if (result.status === 'failed') process.exit(1)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    process.stderr.write(chalk.red(`external ACP agent run failed: ${message}\n`))
    process.exit(1)
  }
}
