import chalk from 'chalk'
import {
  DaemonClient,
  type CliFileMemorySnapshot,
  type CliMemoryAuditEntry,
  type CliMemoryLifecycleStatus,
  type CliMemoryMaintenanceInput,
  type CliMemoryScopeTransferInput,
  type CliMemoryScopeTransferResult,
  type CliMemorySemanticStatus,
  type CliMemorySecurityAuditResult,
  type CliMemoryScopes,
} from '../client/http.js'
import { getOutputFormat, output } from '../output/formatter.js'

type MemorySearchType = 'semantic' | 'keyword' | 'hybrid'

const OPEN_LOOP_QUEUE_SECTION = 'Open Loop Queue'
const DAILY_BACKLOG_SECTION = 'Backlog'
const DAILY_REFLECTION_SECTION = 'Reflection Ledger'

export async function memorySearchCommand(
  query: string,
  options: {
    url?: string
    type?: MemorySearchType
    limit?: string
  },
) {
  const client = new DaemonClient(options.url)
  const type = normalizeSearchType(options.type)
  if (options.type && !type) {
    console.error(chalk.red(`Invalid search type: ${options.type}`))
    process.exit(1)
  }
  const limit = parseOptionalLimit(options.limit)
  const data = await client.searchMemory(query, {
    type,
    limit,
  })
  if (getOutputFormat() === 'json') {
    output(data)
    return
  }
  if (!data?.length) { console.log('No results.'); return }
  for (const m of data) {
    console.log(`  [${m.source}] ${m.content.slice(0, 100)}`)
    if (m.tags?.length) console.log(`    tags: ${m.tags.join(', ')}`)
  }
}

export async function memoryDocumentSearchCommand(
  query: string,
  options: {
    url?: string
    type?: MemorySearchType
    limit?: string
    documentId?: string
  },
) {
  const client = new DaemonClient(options.url)
  const type = normalizeSearchType(options.type)
  if (options.type && !type) {
    console.error(chalk.red(`Invalid search type: ${options.type}`))
    process.exit(1)
  }
  const limit = parseOptionalLimit(options.limit)
  const data = await client.searchMemoryDocuments(query, {
    type,
    limit,
    documentId: options.documentId?.trim() || undefined,
  })

  if (getOutputFormat() === 'json') {
    output(data)
    return
  }
  if (!data?.length) {
    console.log('No document results.')
    return
  }

  for (const chunk of data) {
    console.log(chalk.cyan(chunk.citationLabel ?? chunk.documentTitle))
    console.log(`  ${chunk.snippet ?? chunk.content.slice(0, 180)}`)
    const detailParts = [
      `chunk ${chunk.chunkIndex + 1}/${chunk.chunkCount}`,
      chunk.score != null ? `score ${chunk.score.toFixed(2)}` : null,
      chunk.documentPath ? `path ${chunk.documentPath}` : null,
    ].filter((part): part is string => Boolean(part))
    if (detailParts.length > 0) {
      console.log(chalk.gray(`  ${detailParts.join(' • ')}`))
    }
  }
}

export async function memoryDocumentListCommand(
  options: {
    url?: string
    query?: string
    limit?: string
  },
) {
  const client = new DaemonClient(options.url)
  const limit = parseOptionalLimit(options.limit)
  const data = await client.listMemoryDocuments({
    query: options.query?.trim() || undefined,
    limit,
  })

  if (getOutputFormat() === 'json') {
    output(data)
    return
  }
  if (!data?.length) {
    console.log('No indexed documents.')
    return
  }

  for (const document of data) {
    console.log(chalk.cyan(document.title))
    const detailParts = [
      document.id,
      document.path ? `path ${document.path}` : null,
      document.chunkCount > 0
        ? `${document.chunkCount} chunk${document.chunkCount === 1 ? '' : 's'}`
        : null,
      document.tags.length > 0 ? `tags ${document.tags.join(', ')}` : null,
    ].filter((part): part is string => Boolean(part))
    if (detailParts.length > 0) {
      console.log(chalk.gray(`  ${detailParts.join(' • ')}`))
    }
  }
}

export async function memoryDocumentAddCommand(
  options: {
    url?: string
    id?: string
    title?: string
    content?: string
    path?: string
    mimeType?: string
    sourceFileId?: string
    tags?: string
  },
) {
  if (!options.title?.trim() || !options.content?.trim()) {
    console.error(chalk.red('`--title` and `--content` are required.'))
    process.exit(1)
  }

  const client = new DaemonClient(options.url)
  const data = await client.ingestMemoryDocument({
    id: options.id?.trim() || undefined,
    title: options.title.trim(),
    content: options.content.trim(),
    path: options.path?.trim() || undefined,
    mimeType: options.mimeType?.trim() || undefined,
    sourceFileId: options.sourceFileId?.trim() || undefined,
    tags: splitTags(options.tags),
  })

  if (getOutputFormat() === 'json') {
    output({ ok: true, ...data })
    return
  }
  console.log(chalk.green(`Indexed document: ${data.title}`))
  console.log(chalk.gray(`ID: ${data.id}`))
  console.log(chalk.gray(`Chunks: ${data.chunkCount}`))
  if (data.path) {
    console.log(chalk.gray(`Path: ${data.path}`))
  }
}

export async function memoryDocumentShowCommand(
  id: string,
  options: { url?: string },
) {
  const client = new DaemonClient(options.url)
  const data = await client.memoryDocument(id)

  if (getOutputFormat() === 'json') {
    output(data)
    return
  }

  console.log(chalk.cyan(data.title))
  console.log(chalk.gray(`ID: ${data.id}`))
  console.log(chalk.gray(`Chunks: ${data.chunkCount}`))
  if (data.path) console.log(chalk.gray(`Path: ${data.path}`))
  if (data.mimeType) console.log(chalk.gray(`MIME: ${data.mimeType}`))
  if (data.sourceFileId) console.log(chalk.gray(`Source file: ${data.sourceFileId}`))
  if (data.tags.length > 0) console.log(chalk.gray(`Tags: ${data.tags.join(', ')}`))
}

export async function memoryDocumentDeleteCommand(
  id: string,
  options: { url?: string },
) {
  const client = new DaemonClient(options.url)
  await client.deleteMemoryDocument(id)

  if (getOutputFormat() === 'json') {
    output({ ok: true, id, deleted: true })
    return
  }
  console.log(chalk.green(`Deleted indexed document: ${id}`))
}

export async function memoryAddCommand(content: string, options: { url?: string; tags?: string }) {
  const client = new DaemonClient(options.url)
  const tags = splitTags(options.tags)
  const data = await client.addMemory(content, tags)
  if (getOutputFormat() === 'json') {
    output({ ok: true, ...data })
    return
  }
  console.log(`Memory added: ${data.id}`)
}

export async function memoryFileShowCommand(
  section: string | undefined,
  options: { url?: string; daily?: 'today' | 'yesterday' },
) {
  const client = new DaemonClient(options.url)
  const data = await client.fileMemory()

  if (getOutputFormat() === 'json') {
    output(data)
    return
  }

  if (options.daily) {
    const note = options.daily === 'yesterday' ? data.yesterdayNote : data.todayNote
    const path = options.daily === 'yesterday' ? data.yesterdayNotePath : data.todayNotePath
    console.log(chalk.green(`${options.daily} daily note`))
    console.log(chalk.gray(path))
    console.log(note.trim() || '(empty)')
    return
  }

  if (section?.trim()) {
    const found = data.sections.find((entry) => entry.title === section.trim())
    if (!found) {
      console.error(chalk.red(`Memory section not found: ${section}`))
      process.exit(1)
    }
    console.log(chalk.green(`## ${found.title}`))
    console.log(found.content || '(empty)')
    return
  }

  console.log(chalk.green('Markdown memory'))
  console.log(chalk.gray(data.memoryPath))
  if (data.sections.length === 0) {
    console.log('No sections.')
    return
  }
  for (const item of data.sections) {
    const lineCount = item.content ? item.content.split('\n').length : 0
    console.log(`  ${item.title} ${chalk.gray(`(${lineCount} line${lineCount === 1 ? '' : 's'})`)}`)
  }
  console.log(chalk.gray('Use `sepilot memory file show <section>` to inspect a section.'))
}

export async function memoryFileSetCommand(
  section: string,
  content: string | string[],
  options: { url?: string },
) {
  const client = new DaemonClient(options.url)
  const nextContent = Array.isArray(content) ? content.join(' ') : content
  const data = await client.updateFileMemorySection(section, nextContent)

  if (getOutputFormat() === 'json') {
    output({ ok: true, ...data })
    return
  }
  if (data.deleted) {
    console.log(chalk.yellow(`Cleared memory section: ${section}`))
  } else {
    console.log(chalk.green(`Updated memory section: ${data.title}`))
  }
}

export async function memoryFileDeleteCommand(
  section: string,
  options: { url?: string },
) {
  const client = new DaemonClient(options.url)
  const data = await client.deleteFileMemorySection(section)

  if (getOutputFormat() === 'json') {
    output({ ok: true, section, ...data })
    return
  }
  console.log(data.deleted
    ? chalk.green(`Deleted memory section: ${section}`)
    : chalk.yellow(`Memory section was already absent: ${section}`))
}

export async function memoryBacklogListCommand(
  options: {
    url?: string
    daily?: 'today' | 'yesterday' | 'both'
    limit?: string
    reflections?: boolean
  },
) {
  const client = new DaemonClient(options.url)
  const data = await client.fileMemory()
  const limit = parseOptionalLimit(options.limit)
  const snapshot = buildBacklogSnapshot(data, {
    daily: normalizeBacklogDaily(options.daily),
    limit,
    includeReflections: Boolean(options.reflections),
  })

  if (getOutputFormat() === 'json') {
    output(snapshot)
    return
  }

  console.log(chalk.green('Memory backlog'))
  console.log(chalk.gray(data.memoryPath))

  if (snapshot.openLoopQueue.length > 0) {
    console.log(chalk.cyan('\nOpen loop queue'))
    for (const item of snapshot.openLoopQueue) {
      console.log(`  ${item}`)
    }
  } else {
    console.log(chalk.gray('\nOpen loop queue: empty'))
  }

  for (const daily of snapshot.daily) {
    console.log(chalk.cyan(`\n${daily.label} backlog`))
    if (daily.backlog.length === 0) {
      console.log(chalk.gray('  empty'))
    } else {
      for (const item of daily.backlog) {
        console.log(`  ${item}`)
      }
    }
    if (options.reflections) {
      console.log(chalk.cyan(`${daily.label} reflection ledger`))
      if (daily.reflections.length === 0) {
        console.log(chalk.gray('  empty'))
      } else {
        for (const item of daily.reflections) {
          console.log(`  ${item}`)
        }
      }
    }
  }

  console.log(chalk.gray('\nUse `sepilot memory backlog add <text>` to add an item, or `sepilot memory backlog done <text>` to clear matching open-loop items.'))
}

export async function memoryBacklogAddCommand(
  content: string | string[],
  options: { url?: string },
) {
  const text = normalizeVariadicText(content)
  if (!text) {
    console.error(chalk.red('Backlog item text is required.'))
    process.exit(1)
  }

  const client = new DaemonClient(options.url)
  const data = await client.fileMemory()
  const current = findMemorySection(data, OPEN_LOOP_QUEUE_SECTION)
  const line = formatManualBacklogLine(text)
  const next = appendUniqueLine(current, line)
  await client.updateFileMemorySection(OPEN_LOOP_QUEUE_SECTION, next)

  if (getOutputFormat() === 'json') {
    output({ ok: true, section: OPEN_LOOP_QUEUE_SECTION, item: line })
    return
  }

  console.log(chalk.green('Added backlog item'))
  console.log(`  ${line}`)
}

export async function memoryBacklogDoneCommand(
  query: string | string[],
  options: { url?: string },
) {
  const text = normalizeVariadicText(query).toLowerCase()
  if (!text) {
    console.error(chalk.red('A query is required to match backlog items.'))
    process.exit(1)
  }

  const client = new DaemonClient(options.url)
  const data = await client.fileMemory()
  const current = findMemorySection(data, OPEN_LOOP_QUEUE_SECTION)
  const result = removeMatchingLines(current, text)
  if (result.removed.length === 0) {
    console.error(chalk.yellow(`No open-loop backlog items matched: ${text}`))
    process.exit(1)
  }

  await client.updateFileMemorySection(OPEN_LOOP_QUEUE_SECTION, result.remaining)

  if (getOutputFormat() === 'json') {
    output({
      ok: true,
      section: OPEN_LOOP_QUEUE_SECTION,
      removed: result.removed,
      remainingCount: countNonEmptyLines(result.remaining),
    })
    return
  }

  console.log(chalk.green(`Resolved ${result.removed.length} backlog item${result.removed.length === 1 ? '' : 's'}`))
  for (const item of result.removed) {
    console.log(`  ${item}`)
  }
}

export async function memoryReindexCommand(options: { url?: string }) {
  const client = new DaemonClient(options.url)
  try {
    const data = await client.reindexMemory()

    if (getOutputFormat() === 'json') {
      output({ ok: true, ...data })
      return
    }

    const semantic = data.status
    const detailParts = formatSemanticDetailParts(semantic)

    console.log(chalk.green('Semantic memory reindex started'))
    console.log(chalk.gray(`Status: ${semantic.status}`))
    if (detailParts.length > 0) {
      console.log(chalk.gray(`Details: ${detailParts.join(', ')}`))
    }
    if (semantic.lastError) {
      console.log(chalk.yellow(`Last error: ${semantic.lastError}`))
    }
    console.log(chalk.gray('Use `sepilot memory status` to watch progress.'))
  } catch (error) {
    const message = formatCliError(error)
    console.error(chalk.red(message))
    process.exit(1)
  }
}

export async function memoryStatusCommand(options: { url?: string }) {
  const client = new DaemonClient(options.url)

  try {
    const data = await client.memoryStatus()

    if (getOutputFormat() === 'json') {
      output(data)
      return
    }

    console.log(chalk.green('Semantic memory'))
    console.log(chalk.gray(`Status: ${data.status}`))

    const detailParts = formatSemanticDetailParts(data)
    if (detailParts.length > 0) {
      console.log(chalk.gray(`Details: ${detailParts.join(', ')}`))
    }
    // Only surface the indexed model when it differs from the configured
    // model — otherwise it's just noise duplicating the Details line.
    if (
      data.indexedProviderId
      && data.indexedModel
      && (
        data.indexedProviderId !== data.configuredProviderId
        || data.indexedModel !== data.configuredModel
      )
    ) {
      console.log(chalk.gray(`Indexed: ${data.indexedProviderId}/${data.indexedModel} (drift)`))
    }
    if (data.lastError) {
      console.log(chalk.yellow(`Last error: ${data.lastError}`))
    }
  } catch (error) {
    const message = formatCliError(error)
    console.error(chalk.red(message))
    process.exit(1)
  }
}

export async function memoryLifecycleCommand(
  options: {
    url?: string
    staleAfterDays?: string
    lowImportance?: string
  },
) {
  const client = new DaemonClient(options.url)
  const staleAfterDays = parseOptionalPositiveInt(options.staleAfterDays)
  const lowImportance = parseOptionalFraction(options.lowImportance)

  try {
    const data = await client.memoryLifecycle({
      staleAfterDays,
      lowImportance,
    })

    if (getOutputFormat() === 'json') {
      output(data)
      return
    }

    printMemoryLifecycle(data)
  } catch (error) {
    const message = formatCliError(error)
    console.error(chalk.red(message))
    process.exit(1)
  }
}

export async function memoryAuditCommand(
  options: {
    url?: string
    memoryId?: string
    limit?: string
  },
) {
  const client = new DaemonClient(options.url)

  try {
    const data = await client.memoryAudit({
      memoryId: options.memoryId?.trim() || undefined,
      limit: parseOptionalLimit(options.limit),
    })

    if (getOutputFormat() === 'json') {
      output(data)
      return
    }

    if (data.length === 0) {
      console.log(chalk.gray('No memory audit entries.'))
      return
    }

    console.log(chalk.green('Memory audit trail'))
    for (const entry of data) {
      printMemoryAuditEntry(entry)
    }
  } catch (error) {
    const message = formatCliError(error)
    console.error(chalk.red(message))
    process.exit(1)
  }
}

export async function memorySecurityAuditCommand(
  options: {
    url?: string
    limit?: string
    since?: string
    actor?: string
    authKind?: string
    route?: string
  },
) {
  const client = new DaemonClient(options.url)

  try {
    const data = await client.memorySecurityAudit({
      limit: parseOptionalLimit(options.limit),
      since: options.since?.trim() || undefined,
      actor: options.actor?.trim() || undefined,
      authKind: options.authKind?.trim() || undefined,
      route: options.route?.trim() || undefined,
    })

    if (getOutputFormat() === 'json') {
      output(data)
      return
    }

    printMemorySecurityAudit(data)
  } catch (error) {
    const message = formatCliError(error)
    console.error(chalk.red(message))
    process.exit(1)
  }
}

export async function memoryScopesCommand(options: { url?: string }) {
  const client = new DaemonClient(options.url)

  try {
    const data = await client.memoryScopes()

    if (getOutputFormat() === 'json') {
      output(data)
      return
    }

    printMemoryScopes(data)
  } catch (error) {
    const message = formatCliError(error)
    console.error(chalk.red(message))
    process.exit(1)
  }
}

export async function memoryScopeTransferCommand(
  source: string,
  target: string,
  options: {
    url?: string
    apply?: boolean
    confirmGlobal?: boolean
    ids?: string
    limit?: string
    includeFile?: boolean
    reminders?: boolean
    reason?: string
  },
) {
  const sourceScope = source.trim()
  const targetScope = target.trim()
  if (!sourceScope || !targetScope) {
    console.error(chalk.red('Source and target scopes are required.'))
    process.exit(1)
  }

  const client = new DaemonClient(options.url)
  const input: CliMemoryScopeTransferInput = {
    target: targetScope,
    dryRun: !options.apply,
    confirmGlobal: Boolean(options.confirmGlobal),
    includeFile: Boolean(options.includeFile),
    includeReminders: options.reminders !== false,
    reason: options.reason?.trim() || (
      options.apply
        ? `Transfer memory scope ${sourceScope} to ${targetScope} from CLI`
        : `Preview memory scope transfer ${sourceScope} to ${targetScope} from CLI`
    ),
    ids: splitTags(options.ids),
    limit: parseOptionalLimit(options.limit),
  }

  try {
    const data = await client.transferMemoryScope(sourceScope, input)

    if (getOutputFormat() === 'json') {
      output({ ok: true, ...data })
      return
    }

    printMemoryScopeTransfer(data)
  } catch (error) {
    const message = formatCliError(error)
    console.error(chalk.red(message))
    process.exit(1)
  }
}

export async function memoryMaintenanceCommand(
  options: {
    url?: string
    apply?: boolean
    maxAgeDays?: string
    maxImportance?: string
    reason?: string
  },
) {
  const client = new DaemonClient(options.url)
  const input: CliMemoryMaintenanceInput = {
    dryRun: !options.apply,
    maxAgeDays: parseOptionalPositiveInt(options.maxAgeDays),
    maxImportance: parseOptionalFraction(options.maxImportance),
    reason: options.reason?.trim() || (
      options.apply
        ? 'Run memory lifecycle maintenance from CLI'
        : 'Preview memory lifecycle maintenance from CLI'
    ),
  }

  try {
    const data = await client.runMemoryMaintenance(input)

    if (getOutputFormat() === 'json') {
      output({ ok: true, ...data })
      return
    }

    console.log(chalk.green(data.dryRun ? 'Memory maintenance preview' : 'Memory maintenance completed'))
    console.log(chalk.gray(`Would prune: ${data.wouldPrune}`))
    console.log(chalk.gray(`Pruned: ${data.pruned}`))
    if (data.importanceUpdated > 0) {
      console.log(chalk.gray(`Importance updated: ${data.importanceUpdated}`))
    }
    printMemoryLifecycle(data.status)
    if (data.dryRun) {
      console.log(chalk.gray('Use `sepilot memory maintenance --apply` to prune matching entries.'))
    }
  } catch (error) {
    const message = formatCliError(error)
    console.error(chalk.red(message))
    process.exit(1)
  }
}

function formatSemanticDetailParts(semantic: CliMemorySemanticStatus): string[] {
  const detailParts = []
  if (semantic.configuredProviderId && semantic.configuredModel) {
    detailParts.push(`${semantic.configuredProviderId}/${semantic.configuredModel}`)
  }
  if (semantic.dimensions) {
    detailParts.push(`${semantic.dimensions}d`)
  }
  if (semantic.pendingCount > 0) {
    detailParts.push(`${semantic.pendingCount} pending`)
  }
  if (semantic.failedCount > 0) {
    detailParts.push(`${semantic.failedCount} failed`)
  }
  if (!semantic.vecAvailable) {
    detailParts.push('sqlite-vec unavailable')
  }
  return detailParts
}

function printMemoryLifecycle(data: CliMemoryLifecycleStatus) {
  console.log(chalk.green('Memory lifecycle'))
  console.log(chalk.gray(`Total memories: ${data.totalMemories}`))
  console.log(chalk.gray(`Cleanup candidates: ${data.pruneCandidateMemories}`))
  console.log(chalk.gray(
    `Embeddings: ${data.pendingEmbeddings} pending, ${data.failedEmbeddings} failed`,
  ))
  console.log(chalk.gray(
    [
      `conversation ${data.conversationMemories}`,
      `documents ${data.documentMemories}`,
      `skills ${data.skillMemories}`,
      `user ${data.userMemories}`,
    ].join(' • '),
  ))
  if (data.staleConversationMemories > 0 || data.lowImportanceConversationMemories > 0) {
    console.log(chalk.yellow(
      `Conversation cleanup signals: ${data.staleConversationMemories} stale, ${data.lowImportanceConversationMemories} low-importance`,
    ))
  }
  console.log(chalk.gray(`Last audit: ${formatCliDateTime(data.lastAuditAt, 'never')}`))
}

function printMemoryAuditEntry(entry: CliMemoryAuditEntry) {
  console.log(chalk.cyan(`\n${entry.action} ${entry.memoryId}`))
  console.log(chalk.gray(`  actor: ${entry.actor} • at: ${formatCliDateTime(entry.createdAt, 'unknown')}`))
  if (entry.reason) {
    console.log(chalk.gray(`  reason: ${entry.reason}`))
  }
  const after = entry.after?.content ?? entry.before?.content
  if (after) {
    console.log(`  ${clipText(after, 160)}`)
  }
}

function printMemorySecurityAudit(result: CliMemorySecurityAuditResult) {
  if (result.data.length === 0) {
    console.log(chalk.gray('No memory security audit events.'))
    return
  }

  console.log(chalk.green('Memory security audit'))
  console.log(chalk.gray(`Returned ${result.meta.returned}/${result.meta.limit}`))
  for (const event of result.data) {
    const title = `${event.event} ${event.actor ?? 'unknown-actor'}`
    console.log(chalk.cyan(`\n${title}`))
    console.log(chalk.gray(`  at: ${formatCliDateTime(event.timestamp, 'unknown')}`))
    const route = [event.method, event.route].filter(Boolean).join(' ')
    if (route) console.log(chalk.gray(`  route: ${route}`))
    if (event.authKind) console.log(chalk.gray(`  auth: ${event.authKind}`))
    if (event.requested) console.log(chalk.gray(`  requested: ${event.requested}`))
    if (event.reason) console.log(chalk.gray(`  reason: ${event.reason}`))
    if (event.scopeTags?.length) console.log(chalk.gray(`  scope: ${event.scopeTags.join(', ')}`))
    if (event.tokenId) console.log(chalk.gray(`  token: ${event.tokenId}${event.label ? ` (${event.label})` : ''}`))
  }
}

function printMemoryScopes(data: CliMemoryScopes) {
  console.log(chalk.green('Memory scopes'))
  if (data.fileScopes.length > 0) {
    console.log(chalk.cyan('\nFile buckets'))
    for (const scope of data.fileScopes) console.log(`  ${scope}`)
  } else {
    console.log(chalk.gray('\nFile buckets: none'))
  }

  if (data.semanticScopes.length > 0) {
    console.log(chalk.cyan('\nSemantic scopes'))
    for (const item of data.semanticScopes) {
      console.log(`  ${item.scope} ${chalk.gray(`(${item.count})`)}`)
    }
  } else {
    console.log(chalk.gray('\nSemantic scopes: none'))
  }

  console.log(chalk.gray(`\nLegacy global semantic entries: ${data.untaggedSemanticEntries}`))
  if (data.pendingReminders.length > 0) {
    console.log(chalk.cyan('\nPending reminder scopes'))
    for (const item of data.pendingReminders) {
      console.log(`  ${item.scope} ${chalk.gray(`(${item.count})`)}`)
    }
  }
}

function printMemoryScopeTransfer(data: CliMemoryScopeTransferResult) {
  console.log(chalk.green(data.dryRun ? 'Memory scope transfer preview' : 'Memory scope transfer completed'))
  console.log(chalk.gray(`${data.fromScope} -> ${data.toScope}`))
  if (data.globalSource) {
    console.log(chalk.yellow('Source is legacy global memory. Apply requires --confirm-global.'))
  }

  if (data.dryRun) {
    console.log(chalk.gray(`Matched semantic memories: ${data.matchedSemanticMemories ?? data.wouldRetagSemanticMemories ?? 0}`))
    console.log(chalk.gray(`Would retag semantic memories: ${data.wouldRetagSemanticMemories ?? 0}`))
    console.log(chalk.gray(`Would update reminders: ${data.wouldUpdateReminders ?? 0}`))
    console.log(chalk.gray(`Would move file bucket: ${data.wouldMoveFileBucket ?? 'no'}`))
    if (data.limited) console.log(chalk.yellow('Result was limited; rerun without --limit or with a larger limit to include all matches.'))
    if (data.sampleSemanticIds?.length) {
      console.log(chalk.gray(`Sample semantic ids: ${data.sampleSemanticIds.join(', ')}`))
    }
    const confirm = data.globalSource ? ' --confirm-global' : ''
    console.log(chalk.gray(`Use --apply${confirm} to execute this transfer.`))
    return
  }

  console.log(chalk.gray(`Matched semantic memories: ${data.matchedSemanticMemories ?? data.retaggedSemanticMemories ?? 0}`))
  console.log(chalk.gray(`Retagged semantic memories: ${data.retaggedSemanticMemories ?? 0}`))
  console.log(chalk.gray(`Retagged reminders: ${data.retaggedReminders ?? 0}`))
  console.log(chalk.gray(`File bucket moved: ${data.fileBucketMoved ? 'yes' : 'no'}`))
}

function parseOptionalLimit(limit: string | undefined): number | undefined {
  if (!limit) return undefined
  const parsed = Number.parseInt(limit, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined
}

function parseOptionalPositiveInt(value: string | undefined): number | undefined {
  if (!value) return undefined
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined
}

function parseOptionalFraction(value: string | undefined): number | undefined {
  if (!value) return undefined
  const parsed = Number.parseFloat(value)
  return Number.isFinite(parsed) && parsed >= 0 && parsed <= 1 ? parsed : undefined
}

function formatCliDateTime(value: string | undefined, fallback: string): string {
  if (!value) return fallback
  const parsed = Date.parse(value)
  if (!Number.isFinite(parsed)) return fallback
  return new Date(parsed).toISOString()
}

function clipText(value: string, maxLength: number): string {
  const normalized = value.replace(/\s+/g, ' ').trim()
  return normalized.length > maxLength ? `${normalized.slice(0, maxLength - 1)}…` : normalized
}

export function buildBacklogSnapshot(
  data: CliFileMemorySnapshot,
  options: {
    daily?: 'today' | 'yesterday' | 'both'
    limit?: number
    includeReflections?: boolean
  } = {},
) {
  const dailyMode = options.daily ?? 'today'
  const dailySources = dailyMode === 'both'
    ? [
        { label: 'today' as const, note: data.todayNote, path: data.todayNotePath },
        { label: 'yesterday' as const, note: data.yesterdayNote, path: data.yesterdayNotePath },
      ]
    : [
        dailyMode === 'yesterday'
          ? { label: 'yesterday' as const, note: data.yesterdayNote, path: data.yesterdayNotePath }
          : { label: 'today' as const, note: data.todayNote, path: data.todayNotePath },
      ]

  return {
    memoryPath: data.memoryPath,
    openLoopQueue: limitLines(splitDisplayLines(findMemorySection(data, OPEN_LOOP_QUEUE_SECTION)), options.limit),
    daily: dailySources.map((source) => ({
      ...source,
      backlog: limitLines(splitDisplayLines(extractMarkdownSection(source.note, DAILY_BACKLOG_SECTION)), options.limit),
      reflections: options.includeReflections
        ? limitLines(splitDisplayLines(extractMarkdownSection(source.note, DAILY_REFLECTION_SECTION)), options.limit)
        : [],
    })),
  }
}

export function extractMarkdownSection(markdown: string, title: string): string {
  const lines = markdown.split(/\r?\n/)
  const start = lines.findIndex((line) => line.trim() === `## ${title}`)
  if (start < 0) return ''
  const collected: string[] = []
  for (const line of lines.slice(start + 1)) {
    if (/^##\s+\S/.test(line)) break
    collected.push(line)
  }
  return collected.join('\n').trim()
}

export function removeMatchingLines(content: string, query: string): { removed: string[]; remaining: string } {
  const removed: string[] = []
  const remaining: string[] = []
  for (const line of content.split(/\r?\n/)) {
    const normalized = line.trim()
    if (!normalized) continue
    if (normalized.toLowerCase().includes(query)) {
      removed.push(normalized)
    } else {
      remaining.push(normalized)
    }
  }
  return { removed, remaining: remaining.join('\n') }
}

function normalizeBacklogDaily(raw: string | undefined): 'today' | 'yesterday' | 'both' {
  if (raw === 'yesterday' || raw === 'both') return raw
  return 'today'
}

function findMemorySection(data: CliFileMemorySnapshot, title: string): string {
  return data.sections.find((section) => section.title === title)?.content ?? ''
}

function formatManualBacklogLine(content: string): string {
  return `- ${new Date().toISOString()} source=manual: ${content} | status=needs_follow_up | next=Resume this backlog item when relevant.`
}

function appendUniqueLine(content: string, line: string): string {
  const lines = splitDisplayLines(content)
  if (lines.some((existing) => existing === line)) return lines.join('\n')
  return [...lines, line].join('\n')
}

function splitDisplayLines(content: string): string[] {
  return content.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
}

function limitLines(lines: string[], limit: number | undefined): string[] {
  return limit ? lines.slice(0, limit) : lines
}

function normalizeVariadicText(value: string | string[]): string {
  return (Array.isArray(value) ? value.join(' ') : value).trim()
}

function countNonEmptyLines(value: string): number {
  return splitDisplayLines(value).length
}

function splitTags(tags: string | undefined): string[] {
  return tags?.split(',').map((tag) => tag.trim()).filter(Boolean) ?? []
}

function normalizeSearchType(
  type: string | undefined,
): MemorySearchType | undefined {
  if (!type) return undefined
  if (type === 'semantic' || type === 'keyword' || type === 'hybrid') return type
  return undefined
}

function formatCliError(error: unknown): string {
  const cause = (error as { cause?: unknown }).cause
  const causeCode = cause && typeof cause === 'object' && 'code' in cause
    ? String((cause as { code: unknown }).code)
    : undefined
  const looksLikeFetchFailure = error instanceof TypeError
    && /fetch failed/i.test(error.message)
  if (
    causeCode === 'ECONNREFUSED'
    || causeCode === 'ECONNRESET'
    || causeCode === 'ENOTFOUND'
    || looksLikeFetchFailure
  ) {
    return 'Cannot connect to sepilotd. Is the daemon running? Start with: sepilot start'
  }
  const raw = error instanceof Error ? error.message : String(error)
  const jsonMatch = raw.match(/\{[\s\S]+\}$/)
  if (!jsonMatch) return raw

  try {
    const parsed = JSON.parse(jsonMatch[0]) as { error?: { message?: string } }
    return parsed.error?.message ?? raw
  } catch {
    return raw
  }
}
