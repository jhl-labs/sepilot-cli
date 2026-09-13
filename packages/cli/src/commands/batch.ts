import { appendFile, readFile, writeFile } from 'node:fs/promises'
import chalk from 'chalk'
import type { JobsClient } from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { ensureDaemon } from '../client/ensure-daemon.js'
import { errorMessage } from '../utils/error-message.js'
import { output } from '../output/formatter.js'

interface BatchTask {
  message: string
  id?: string
}

interface BatchResult {
  id: string
  message: string
  response: string
  tokens: number
  status: string
  error?: string
}

interface BatchProgressUpdate {
  idx: number
  total: number
  ok: boolean
  message: string
  tokens: number
  error?: string
}

export interface BatchFacadeInput {
  tasks: BatchTask[]
  concurrency: number
  failureMode: 'continue' | 'abort'
  jobs: Pick<JobsClient, 'submitBatch' | 'get' | 'getItems' | 'cancel'>
  writer: { append(line: string): void | Promise<void> }
  pollMs?: number
  /**
   * Total elapsed-time cap for the polling loop, in milliseconds. When
   * the daemon never transitions to a terminal state (network blip,
   * crash without status update) the CLI would otherwise hang. Default:
   * 60 minutes — overridable via `SEPILOTD_BATCH_TIMEOUT_MS`. Tests
   * pass an explicit small value via this field instead of relying on
   * the env var. `0` disables the watchdog (used by tests with stubbed
   * pollers).
   */
  timeoutMs?: number
  onProgress?(snap: BatchProgressUpdate): void
}

interface BatchFacadeResult {
  jobId: string
  ok: number
  total: number
  tokens: number
  abandoned: boolean
}

interface TextLikeJobResult {
  content?: string
  usage?: { inputTokens?: number; outputTokens?: number }
  status?: string
  error?: string
}

const DEFAULT_BATCH_TIMEOUT_MS = 60 * 60 * 1000

function resolveBatchTimeoutMs(explicit?: number): number {
  if (typeof explicit === 'number') return explicit
  const envRaw = process.env.SEPILOTD_BATCH_TIMEOUT_MS
  if (envRaw && envRaw.length > 0) {
    const parsed = Number.parseInt(envRaw, 10)
    if (Number.isFinite(parsed) && parsed >= 0) return parsed
  }
  return DEFAULT_BATCH_TIMEOUT_MS
}

/**
 * Thrown by `runBatchFacade` when the polling loop exceeds its
 * elapsed-time cap. The verb wrapper catches this and prints a
 * resume-with-`batch:resume` hint, then exits 2.
 */
export class BatchPollingTimeoutError extends Error {
  readonly jobId: string
  readonly elapsedMs: number
  constructor(jobId: string, elapsedMs: number) {
    super(
      `batch polling timed out after ${Math.round(elapsedMs / 1000)}s (jobId ${jobId})`,
    )
    this.name = 'BatchPollingTimeoutError'
    this.jobId = jobId
    this.elapsedMs = elapsedMs
  }
}

// Daemon job items can contain chat envelopes (`{ data: { content, usage } }`),
// flat chat results (`{ content, usage }`), or subagent results
// (`{ output, usage }`). Normalize them before writing resumable JSONL.
function unwrapJobResult(result: unknown): TextLikeJobResult {
  if (!result || typeof result !== 'object') return {}
  const envelope = result as { data?: unknown }
  const source =
    envelope.data && typeof envelope.data === 'object' ? envelope.data : result
  const r = source as {
    content?: unknown
    output?: unknown
    usage?: { inputTokens?: number; outputTokens?: number }
    status?: unknown
    error?: unknown
  }
  const content =
    typeof r.content === 'string'
      ? r.content
      : typeof r.output === 'string'
        ? r.output
        : undefined
  return {
    content,
    usage: r.usage,
    status: typeof r.status === 'string' ? r.status : undefined,
    error: typeof r.error === 'string' ? r.error : undefined,
  }
}

function isErrorJobItem(
  item: { status: string; error: string | null },
  result: TextLikeJobResult,
): boolean {
  return item.status !== 'succeeded' || result.status === 'failed'
}

export async function runBatchFacade(
  input: BatchFacadeInput,
): Promise<BatchFacadeResult> {
  const {
    tasks,
    concurrency,
    failureMode,
    jobs,
    writer,
    pollMs = 500,
    onProgress,
  } = input
  const timeoutMs = resolveBatchTimeoutMs(input.timeoutMs)

  const submitted = await jobs.submitBatch({
    items: tasks.map((t) => ({ message: t.message, id: t.id })),
    concurrency,
    failureMode,
    preserveOrder: true,
  })

  let nextIdx = 0
  const writtenIndices = new Set<number>()
  let ok = 0
  let totalTokens = 0
  let abandoned = false
  const startedAt = Date.now()

  const onSigint = (): void => {
    abandoned = true
    process.stderr.write(
      `\nctrl+c — daemon job continues. resume: sepilot batch:resume ${submitted.jobId}\n`,
    )
    console.log(submitted.jobId)
    process.exit(130)
  }
  process.once('SIGINT', onSigint)

  try {
    while (!abandoned) {
      if (pollMs > 0) {
        await new Promise((r) => setTimeout(r, pollMs))
      }
      const snap = await jobs.get(submitted.jobId)
      const fetched = await jobs.getItems(submitted.jobId, nextIdx)
      for (const it of fetched.items) {
        // Defensive: even though getItems is `since=nextIdx`, the
        // daemon may return idx<nextIdx if a previous result was
        // re-fetched on cursor reset. Skip already-written items.
        if (it.idx < nextIdx || writtenIndices.has(it.idx)) continue
        const t = tasks[it.idx]
        if (!t) continue
        const r = unwrapJobResult(it.result)
        const tokens =
          (r.usage?.inputTokens ?? 0) + (r.usage?.outputTokens ?? 0)
        const failed = isErrorJobItem(it, r)
        const error = it.error ?? r.error
        const result: BatchResult = {
          id: t.id ?? String(it.idx + 1),
          message: t.message,
          response: r.content ?? '',
          tokens,
          status: failed ? 'error' : 'ok',
          ...(error ? { error } : {}),
        }
        await writer.append(JSON.stringify(result))
        onProgress?.({
          idx: it.idx + 1,
          total: tasks.length,
          ok: !failed,
          message: t.message,
          tokens,
          error: error ?? undefined,
        })
        if (!failed) ok++
        totalTokens += tokens
        writtenIndices.add(it.idx)
        // Advance only across a contiguous prefix. A fast item 3 must not
        // hide the still-running item 1 on the next poll.
        while (writtenIndices.delete(nextIdx)) nextIdx++
      }
      if (
        snap.status === 'completed' ||
        snap.status === 'failed' ||
        snap.status === 'canceled'
      ) {
        break
      }
      // Watchdog: if the daemon never reaches a terminal state (network
      // blip, crash without status update), fail loudly with a resume
      // hint instead of hanging forever. `batch:resume <jobId>` re-tails
      // the same daemon-side run.
      if (timeoutMs > 0 && Date.now() - startedAt >= timeoutMs) {
        throw new BatchPollingTimeoutError(
          submitted.jobId,
          Date.now() - startedAt,
        )
      }
    }
  } finally {
    process.removeListener('SIGINT', onSigint)
  }

  return {
    jobId: submitted.jobId,
    ok,
    total: tasks.length,
    tokens: totalTokens,
    abandoned,
  }
}

export async function batchCommand(
  inputFile: string,
  options: {
    url?: string
    output?: string
    model?: string
    concurrency?: string
    detach?: boolean
    strict?: boolean
  },
): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    await ensureDaemon(client, { url: options.url })
  } catch (err) {
    console.error(chalk.red(err instanceof Error ? err.message : String(err)))
    process.exit(1)
  }

  let lines: string[]
  try {
    const content = await readFile(inputFile, 'utf-8')
    lines = content.trim().split('\n').filter(Boolean)
  } catch {
    console.error(chalk.red(`Cannot read file: ${inputFile}`))
    process.exit(1)
  }

  const tasks: BatchTask[] = []
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (!line) continue
    try {
      tasks.push(JSON.parse(line) as BatchTask)
    } catch {
      console.error(chalk.yellow(`Line ${i + 1}: invalid JSON, skipping`))
    }
  }

  const concurrency = parseInt(options.concurrency ?? '1', 10) || 1
  const failureMode: 'continue' | 'abort' = 'continue'

  if (options.detach) {
    if (tasks.length === 0) {
      console.error(chalk.red('No tasks to submit (input file is empty).'))
      process.exit(1)
    }
    const sub = await client.jobs.submitBatch({
      items: tasks.map((t) => ({ message: t.message, id: t.id })),
      concurrency,
      failureMode,
      preserveOrder: true,
    })
    console.log(sub.jobId)
    return
  }

  console.log(chalk.gray(`Processing ${tasks.length} tasks from ${inputFile}...\n`))

  if (tasks.length === 0) {
    if (options.output) {
      try {
        await writeFile(options.output, '', 'utf-8')
      } catch (err) {
        handleWriteError(err, options.output)
        process.exit(1)
      }
    }
    console.log(chalk.gray(`\nDone: 0/0 succeeded, 0 total tokens`))
    if (options.output) {
      console.log(chalk.green(`Results written to ${options.output}`))
    }
    return
  }

  // Streaming append writer (or stdout when --output not provided).
  const outputPath = options.output
  if (outputPath) {
    try {
      await writeFile(outputPath, '', 'utf-8')
    } catch (err) {
      handleWriteError(err, outputPath)
      process.exit(1)
    }
  }
  const writer = outputPath
    ? {
        append: async (line: string): Promise<void> => {
          await appendFile(outputPath, line + '\n', 'utf-8')
        },
      }
    : {
        append: (line: string): void => {
          console.log(line)
        },
      }

  const onProgress = (s: BatchProgressUpdate): void => {
    const tag = s.ok ? chalk.green('✓') : chalk.red('✗')
    const detail = s.ok ? `(${s.tokens} tokens)` : `(${s.error ?? 'error'})`
    console.log(
      `  ${tag} [${s.idx}/${s.total}] ${s.message.slice(0, 50)}... ${detail}`,
    )
  }

  let out: BatchFacadeResult
  try {
    out = await runBatchFacade({
      tasks,
      concurrency,
      failureMode,
      jobs: client.jobs,
      writer,
      onProgress,
    })
  } catch (err) {
    if (err instanceof BatchPollingTimeoutError) {
      const seconds = Math.round(err.elapsedMs / 1000)
      console.error(
        chalk.red(
          `batch polling timed out after ${seconds}s (jobId ${err.jobId}) — daemon may still be running. resume with: sepilot batch:resume ${err.jobId}`,
        ),
      )
      process.exit(2)
    }
    throw err
  }

  console.log(
    chalk.gray(
      `\nDone: ${out.ok}/${out.total} succeeded, ${out.tokens} total tokens`,
    ),
  )
  if (outputPath) {
    console.log(chalk.green(`Results written to ${outputPath}`))
  }

  if (options.strict && out.ok < out.total) process.exit(2)
}

export async function batchResumeCommand(
  jobId: string,
  options: { url?: string; output?: string },
): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    await ensureDaemon(client, { url: options.url })
  } catch (err) {
    console.error(chalk.red(err instanceof Error ? err.message : String(err)))
    process.exit(1)
  }

  const outputPath = options.output
  if (outputPath) {
    try {
      await writeFile(outputPath, '', 'utf-8')
    } catch (err) {
      handleWriteError(err, outputPath)
      process.exit(1)
    }
  }

  const writer = outputPath
    ? {
        append: async (line: string): Promise<void> => {
          await appendFile(outputPath, line + '\n', 'utf-8')
        },
      }
    : {
        append: (line: string): void => {
          console.log(line)
        },
      }

  let nextIdx = 0
  const writtenIndices = new Set<number>()
  let abandoned = false
  const onSigint = (): void => {
    abandoned = true
    console.log(jobId)
    process.exit(130)
  }
  process.once('SIGINT', onSigint)

  try {
    while (!abandoned) {
      await new Promise((r) => setTimeout(r, 500))
      const snap = await client.jobs.get(jobId)
      const fetched = await client.jobs.getItems(jobId, nextIdx)
      for (const it of fetched.items) {
        if (it.idx < nextIdx || writtenIndices.has(it.idx)) continue
        const r = unwrapJobResult(it.result)
        const tokens =
          (r.usage?.inputTokens ?? 0) + (r.usage?.outputTokens ?? 0)
        const failed = isErrorJobItem(it, r)
        const error = it.error ?? r.error
        await writer.append(
          JSON.stringify({
            id: String(it.idx + 1),
            message: '',
            response: r.content ?? '',
            tokens,
            status: failed ? 'error' : 'ok',
            ...(error ? { error } : {}),
          }),
        )
        writtenIndices.add(it.idx)
        while (writtenIndices.delete(nextIdx)) nextIdx++
      }
      if (
        snap.status === 'completed' ||
        snap.status === 'failed' ||
        snap.status === 'canceled'
      ) {
        break
      }
    }
  } finally {
    process.removeListener('SIGINT', onSigint)
  }
  console.log(chalk.gray(`Resume complete: ${jobId}`))
}

export async function batchCancelCommand(
  jobId: string,
  options: { url?: string },
): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    await client.jobs.cancel(jobId)
    const snapshot = await client.jobs.get(jobId)
    output({ jobId, status: snapshot.status }, () => chalk.gray(`${snapshot.status}: ${jobId}`))
  } catch (err) {
    console.error(chalk.red(`cancel failed: ${errorMessage(err)}`))
    process.exit(1)
  }
}

export async function batchStatusCommand(
  jobId: string,
  options: { url?: string },
): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    const snap = await client.jobs.get(jobId)
    const kind = snap.kind ? `${snap.kind} ` : ''
    let detail = ''
    if (snap.failed > 0) {
      const items = await client.jobs.getItems(jobId, 0)
      const firstError = items.items.find((item) => item.error)?.error
      if (firstError) detail = `\nerror: ${firstError}`
    }
    const jobError = snap.error ? `\nerror: ${snap.error}` : ''
    const activity = (snap.activity ?? []).map((item) =>
      `\n  [${item.idx + 1}] ${item.status}: ${item.phase}${item.toolName ? ` ${item.toolName}` : ''} · session=${item.sessionId} · ${Math.max(0, Math.floor((Date.now() - item.updatedAt) / 1000))}s ago${item.status === 'running' && item.approvalRequestId ? `\n    approval: sepilot approve ${item.approvalRequestId}` : ''}`,
    ).join('')
    output(snap, () => `${kind}${snap.status} ${snap.succeeded}/${snap.total} succeeded, ${snap.failed} failed${snap.canceled ? `, ${snap.canceled} canceled` : ''}${detail || jobError}${activity}`)
  } catch (err) {
    console.error(chalk.red(`status failed: ${errorMessage(err)}`))
    process.exit(1)
  }
}

export async function jobsListCommand(options: { url?: string; status?: string; kind?: string; limit?: string; offset?: string }): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    const page = await client.jobs.list({ status: options.status, kind: options.kind, limit: options.limit === undefined ? undefined : Number(options.limit), offset: options.offset === undefined ? undefined : Number(options.offset) })
    output(page, (result) => result.jobs.length === 0 ? 'No background jobs.' : [
      ...result.jobs.map((job) => `${job.id}  ${job.kind ?? 'job'}  ${job.status}  ${job.succeeded}/${job.total}${job.activity?.some((item) => item.status === 'running' && item.approvalRequestId) ? '  needs approval' : ''}`),
      'Inspect: sepilot jobs status <id> · Collect: sepilot jobs resume <id> · Stop: sepilot jobs cancel <id>',
      ...(result.nextOffset === null ? [] : [`More: sepilot jobs list --offset ${result.nextOffset}`]),
    ].join('\n'))
  } catch (err) {
    console.error(chalk.red(`jobs list failed: ${errorMessage(err)}`))
    process.exit(1)
  }
}

function handleWriteError(err: unknown, output: string): void {
  const code = (err as { code?: string }).code
  if (code === 'EISDIR') {
    console.error(chalk.red(`--output path is a directory: ${output}`))
  } else if (code === 'EACCES') {
    console.error(
      chalk.red(`--output not writable: ${output} (permission denied)`),
    )
  } else if (code === 'ENOENT') {
    console.error(chalk.red(`--output parent directory missing: ${output}`))
  } else {
    console.error(chalk.red(`Failed to write ${output}: ${errorMessage(err)}`))
  }
}
