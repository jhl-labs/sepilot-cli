import { realpathSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { Command, Option } from 'commander'
import chalk from 'chalk'
import { setOutputFormat, getOutputFormat } from './output/formatter.js'
import { friendlyErrorMessage, printApiError } from './utils/error-message.js'
import { mergeActionOptionsWithGlobals } from './command-options.js'
import { resolveDaemonBaseUrl } from './client/token.js'
import { DaemonClient } from './client/http.js'
import { ensureDaemon, type EnsureDaemonResult } from './client/ensure-daemon.js'
import { getStandaloneDaemon } from './client/standalone-daemon.js'
import {
  loadCliVersion,
  readOptionValue,
  shouldLaunchTui,
  shouldRunRootPrompt,
} from './entrypoint.js'
import { acpCommand } from './commands/acp.js'
import { acpAgentRunCommand } from './commands/acp-agent.js'
import { chatCommand } from './commands/chat.js'
import { statusCommand } from './commands/status.js'
import { assistantStatusCommand } from './commands/assistant.js'
import {
  assistantNotificationReadCommand,
  assistantNotificationShowCommand,
  assistantNotificationsCommand,
  assistantNotificationsReadAllCommand,
} from './commands/assistant-notifications.js'
import { statsCommand } from './commands/stats.js'
import { stateCommand } from './commands/state.js'
import { diagnosticsBundleCommand } from './commands/diagnostics.js'
import {
  sessionsCommand,
  sessionDetailCommand,
  sessionRunbookCommand,
  sessionResumeCommand,
  sessionDeleteCommand,
  sessionCompactCommand,
  sessionExportCommand,
  sessionBranchCommand,
} from './commands/sessions.js'
import {
  scheduleListCommand,
  scheduleShowCommand,
  scheduleAddCommand,
  scheduleEditCommand,
  scheduleRescheduleCommand,
  scheduleRemoveCommand,
  scheduleRunCommand,
  scheduleRunsCommand,
  schedulePauseCommand,
  scheduleResumeCommand,
  scheduleUnattendedCommand,
  scheduleRouteCommand,
} from './commands/schedule.js'
import {
  skillsCommand,
  skillSearchCommand,
  skillEnableCommand,
  skillDisableCommand,
} from './commands/skills.js'
import { pagesInitCommand, pagesScanCommand, pagesStatusCommand } from './commands/pages.js'
import {
  containersLsCommand,
  containersPruneCommand,
  containersRmCommand,
} from './commands/containers.js'
import {
  nativeServiceDisableCommand,
  nativeServiceEnableCommand,
  nativeServiceInstallCommand,
  nativeServiceLogsCommand,
  nativeServiceStatusCommand,
  nativeServiceUninstallCommand,
  serviceHealthcheckCommand,
  serviceListCommand,
  serviceLogsCommand,
  serviceRemoveCommand,
  serviceRestartCommand,
  serviceStartCommand,
  serviceStatusCommand,
  serviceStopCommand,
} from './commands/services.js'
import {
  collectPlanOption,
  plansCreateCommand,
  plansListCommand,
  plansShowCommand,
  plansStartCommand,
} from './commands/plans.js'
import { agentsListCommand, agentsCreateCommand, agentsDeleteCommand } from './commands/agents.js'
import {
  commandsListCommand,
  commandsCreateCommand,
  commandsDeleteCommand,
} from './commands/commands.js'
import { runCommand } from './commands/run.js'
import { autonomyCommand } from './commands/autonomy.js'
import { configCommand, providersCommand } from './commands/config.js'
import { policyCommand } from './commands/policy.js'
import { approveCommand, denyCommand } from './commands/approval.js'
import { answerCommand } from './commands/answer.js'
import { steerCommand } from './commands/steer.js'
import {
  decisionsListCommand,
  decisionsClearCommand,
  decisionsShowCommand,
} from './commands/decisions.js'
import { devicesCommand } from './commands/devices.js'
import {
  memorySearchCommand,
  memoryDocumentAddCommand,
  memoryDocumentDeleteCommand,
  memoryDocumentListCommand,
  memoryDocumentSearchCommand,
  memoryDocumentShowCommand,
  memoryAddCommand,
  memoryAuditCommand,
  memoryBacklogAddCommand,
  memoryBacklogDoneCommand,
  memoryBacklogListCommand,
  memoryFileDeleteCommand,
  memoryFileSetCommand,
  memoryFileShowCommand,
  memoryLifecycleCommand,
  memoryMaintenanceCommand,
  memoryReindexCommand,
  memoryScopeTransferCommand,
  memoryScopesCommand,
  memorySecurityAuditCommand,
  memoryStatusCommand,
} from './commands/memory.js'
import {
  ragDocumentsListCommand,
  ragSearchCommand,
  ragSourceAddCommand,
  ragSourcesListCommand,
  ragStatusCommand,
  ragSyncCommand,
} from './commands/rag.js'
import { usageCommand, usageDailyCommand } from './commands/usage.js'
import { initCommand } from './commands/init.js'
import { doctorCommand } from './commands/doctor.js'
import { logsCommand } from './commands/logs.js'
import { traceCommand } from './commands/trace.js'
import {
  channelAddCommand,
  channelDisableCommand,
  channelEnableCommand,
  channelListCommand,
  channelPairCommand,
  channelRemoveCommand,
  channelUnpairCommand,
  channelUsersCommand,
} from './commands/channel.js'
import { versionCommand } from './commands/version.js'
import { upgradeCommand } from './commands/upgrade.js'
import { backupCommand, restoreCommand } from './commands/backup.js'
import { completionsCommand } from './commands/completions.js'
import { startCommand, stopCommand, restartCommand } from './commands/daemon.js'
import {
  installServiceCommand,
  uninstallServiceCommand,
  showServiceStatus,
} from './commands/daemon-service.js'
import { askCommand } from './commands/ask.js'
import { historyCommand } from './commands/history.js'
import { configEditCommand } from './commands/config-edit.js'
import { testConnectionCommand } from './commands/test-connection.js'
import { skillCreateCommand } from './commands/skill-create.js'
import { skillInstallCommand } from './commands/skill-install.js'
import { skillUninstallCommand } from './commands/skill-uninstall.js'
import { skillUpdateCommand } from './commands/skill-update.js'
import {
  marketplaceListCommand,
  marketplaceAddCommand,
  marketplaceRemoveCommand,
} from './commands/marketplace.js'
import { configSetCommand } from './commands/config-set.js'
import {
  batchCancelCommand,
  batchCommand,
  batchResumeCommand,
  batchStatusCommand,
  jobsListCommand,
} from './commands/batch.js'
import { extensionCreateCommand } from './commands/extension-create.js'
import { extensionInstallCommand } from './commands/extension-install.js'
import {
  migrateCancelCommand,
  migrateFromSepilotDesktopCommand,
  migrateStatusCommand,
} from './commands/migrate.js'
import { tokenIssueCommand, tokenListCommand, tokenRevokeCommand } from './commands/tokens.js'
import {
  hooksAckCommand,
  hooksAddCommand,
  hooksDeadLettersCommand,
  hooksDeliveriesCommand,
  hooksDisableCommand,
  hooksEnableCommand,
  hooksListCommand,
  hooksCommandsCommand,
  hooksReplayCommand,
  hooksReplayFailedCommand,
  hooksRemoveCommand,
} from './commands/hooks.js'
import {
  mcpAddCommand,
  mcpDisableCommand,
  mcpEnableCommand,
  mcpListCommand,
  mcpRemoveCommand,
  mcpTrustManifestCommand,
} from './commands/mcp.js'
import { mcpMetricsCommand } from './commands/mcp-metrics.js'
import { mcpCompletionCommand } from './commands/mcp-completion.js'
import { mcpDoctorCommand } from './commands/mcp-doctor.js'
import { mcpLoggingLogsCommand, mcpLoggingSetLevelCommand } from './commands/mcp-logging.js'
import { mcpSearchCommand } from './commands/mcp-search.js'
import { mcpInstallCommand } from './commands/mcp-install.js'
import { mcpPlaywrightCommand } from './commands/mcp-playwright.js'
import {
  mcpToolsListCommand,
  mcpToolsDisableCommand,
  mcpToolsEnableCommand,
  mcpToolsCallCommand,
} from './commands/mcp-tools.js'
import {
  mcpMarketplaceListCommand as mcpMpListCommand,
  mcpMarketplaceAddCommand as mcpMpAddCommand,
  mcpMarketplaceRemoveCommand as mcpMpRemoveCommand,
} from './commands/mcp-marketplace.js'
import { mcpPromptsListCommand, mcpPromptsGetCommand } from './commands/mcp-prompts.js'
import {
  mcpResourcesListCommand,
  mcpResourcesReadCommand,
  mcpResourcesSubscribeCommand,
  mcpResourcesSubscriptionsCommand,
  mcpResourcesTemplatesCommand,
  mcpResourcesUnsubscribeCommand,
  mcpResourcesUpdatesCommand,
} from './commands/mcp-resources.js'
import {
  webhooksAddCommand,
  webhooksDisableCommand,
  webhooksEnableCommand,
  webhooksListCommand,
  webhooksRemoveCommand,
} from './commands/webhooks.js'
import { secretsSetCommand, secretsListCommand, secretsRemoveCommand } from './commands/secrets.js'
import { authLoginCommand, authListCommand, authLogoutCommand } from './commands/auth.js'
import { subagentCategoriesCommand, subagentDispatchCommand } from './commands/subagent.js'
import { tasksCommand } from './commands/tasks.js'
import { rewindCommand } from './commands/rewind.js'
import {
  swarmRunCommand,
  swarmListCommand,
  swarmStatusCommand,
  swarmKillCommand,
  swarmAgentsCommand,
  swarmLogsCommand,
  swarmAttachCommand,
  swarmDriveCommand,
} from './commands/swarm.js'
import type { SwarmRunStatus } from '@sepilotd/api-client'
import {
  createFunctionKeyAwareStdin,
  createMouseAwareStdin,
  type TuiInputSource,
} from './tui/utils/mouse-input.js'

// No commander default here: when --url is absent, leave options.url
// undefined so resolveDaemonBaseUrl can fall through to SEPILOTD_URL
// env or the library default. A baked-in default would always mask
// the env.
const urlOption = [
  '--url <url>',
  'Daemon URL (defaults to SEPILOTD_URL env or http://127.0.0.1:17600)',
] as const
const perTurnAutonomyDescription =
  'Per-turn autonomy selection (readonly|accept-edits|workspace-write|supervised|autonomous); channel ACL and tool policy still apply'

const program = new Command()
program
  .name('sepilot')
  .description('CLI for sepilotd AI agent daemon')
  .version(loadCliVersion())
  .option(...urlOption)
  .option('-p, --prompt <text>', 'Ask a one-shot question without launching the TUI')
  .option('--model <model>', 'LLM model to use in TUI')
  .option('--provider <provider>', 'LLM provider to use in TUI')
  .option('--max-tokens <n>', 'Output token cap for -p/--prompt one-shot asks')
  .option('--interactive', 'Force rich agent progress for -p/--prompt one-shot asks')
  .option('--session <id>', 'Continue an existing session in TUI or one-shot prompt mode')
  .option('--resume', 'Resume the most recent TUI session')
program.option('--json', 'Output in JSON format')
program.option(
  '--output-format <format>',
  'Output format: text, json, or stream-json (NDJSON agent event stream for automation)',
)
program.addHelpText('after', '\nRunning `sepilot` with no command launches the interactive TUI.')
program.hook('preAction', (_thisCommand, actionCommand) => {
  mergeActionOptionsWithGlobals(actionCommand)
})

const TUI_DAEMON_STOP_WAIT_ATTEMPTS = 10
const TUI_DAEMON_STOP_WAIT_INTERVAL_MS = 300

async function waitForDaemonUnreachable(client: DaemonClient): Promise<void> {
  for (let i = 0; i < TUI_DAEMON_STOP_WAIT_ATTEMPTS; i++) {
    try {
      await client.health({ signal: AbortSignal.timeout(500) })
    } catch {
      return
    }
    await new Promise((resolve) => setTimeout(resolve, TUI_DAEMON_STOP_WAIT_INTERVAL_MS))
  }
}

async function stopOwnedDaemon(
  daemon: EnsureDaemonResult | null,
  client: DaemonClient,
): Promise<void> {
  if (!daemon?.started || typeof daemon.stop !== 'function') return
  daemon.stop()
  await waitForDaemonUnreachable(client)
}

program
  .command('chat')
  .description('Interactive AI chat')
  .option(...urlOption)
  .option('--model <model>', 'LLM model to use')
  .option('--provider <provider>', 'LLM provider to use')
  .option('--max-tokens <n>', 'Output token cap for chat replies')
  .option('--session <id>', 'Resume existing session')
  .action(chatCommand)
program
  .command('acp')
  .description('Run an ACP (Agent Client Protocol) stdio server backed by the local daemon')
  .option(...urlOption)
  .action(acpCommand)
const acpAgent = program
  .command('acp-agent')
  .description('Run external ACP agents through the daemon')
acpAgent
  .command('run <prompt>')
  .description('Run an external ACP agent such as opencode or Codex')
  .option(...urlOption)
  .option('--agent <name>', 'External ACP agent preset (opencode, codex)', 'opencode')
  .option('--cwd <path>', 'Working directory for the external ACP session')
  .option('--session <id>', 'Attach the result to an existing sepilotd session')
  .option('--timeout-ms <ms>', 'ACP request timeout in milliseconds')
  .action(acpAgentRunCommand)
program
  .command('status')
  .description('Check daemon status')
  .option(...urlOption)
  .option('--report', 'Print a detailed health report')
  .option('--output <file>', 'Write the health report to file')
  .action(statusCommand)

const assistant = program
  .command('assistant')
  .description('Show personal-assistant readiness and operating state')
  .option(...urlOption)
  .option('--limit <n>', 'Maximum schedules, background jobs, and notifications to show', '5')
  .action(assistantStatusCommand)
assistant
  .command('status')
  .description('Show personal-assistant readiness and operating state')
  .option(...urlOption)
  .option('--limit <n>', 'Maximum schedules, background jobs, and notifications to show', '5')
  .action(assistantStatusCommand)
assistant
  .command('run [request]')
  .description('Start personal-assistant work as a detached background job')
  .option(...urlOption)
  .option('--model <model>', 'Model')
  .option('--provider <provider>', 'Provider')
  .option('--max-tokens <n>', 'Output token cap for this answer')
  .option('--persona <persona>', 'Persona')
  .option(
    '--mode <mode>',
    'Agent mode or registered graph id (defaults to react)',
  )
  .option(
    '--autonomy <level>',
    perTurnAutonomyDescription,
  )
  .option('--thinking-level <level>', 'Per-turn thinking level: auto, off, low, medium, high, or max')
  .option('--skill <id>', 'Select a daemon skill for this run (repeatable, maximum 8)', collectRepeatedOption, [])
  .option('--panel-preset <id>', 'Persona panel preset id')
  .option('--session <id>', 'Continue an existing session')
  .option('--wait', 'Poll until the background job reaches a terminal state')
  .option('--poll-ms <n>', 'Polling interval for --wait', '1500')
  .option('--output <file>', 'Write a waited-for or foreground answer to file')
  .option('--strip-fences', 'Strip the first Markdown code fence when writing --output')
  .option('--foreground', 'Run as a foreground one-shot request instead of detaching')
  .option('--interactive', 'Show rich progress for a foreground request')
  .action((request, options) => {
    return askCommand(request, {
      url: options.url,
      model: options.model,
      provider: options.provider,
      maxTokens: options.maxTokens,
      persona: options.persona,
      mode: options.mode,
      autonomy: options.autonomy,
      thinkingLevel: options.thinkingLevel,
      skill: options.skill,
      panelPreset: options.panelPreset,
      session: options.session,
      wait: options.wait,
      pollMs: options.pollMs,
      output: options.output,
      stripFences: options.stripFences,
      interactive: options.interactive,
      background: options.foreground !== true,
      backgroundCommandSurface: 'assistant',
    })
  })
assistant
  .command('jobs')
  .description('List personal-assistant background jobs')
  .option(...urlOption)
  .option('--status <status>', 'Only show running, completed, failed, or cancelled jobs')
  .option('--needs-action', 'Only show running jobs waiting for approval or an answer')
  .option('--limit <n>', 'Maximum number of matching jobs to show')
  .action((options) => askCommand(undefined, {
    url: options.url,
    backgroundList: true,
    backgroundListStatus: options.status,
    backgroundListNeedsAction: options.needsAction,
    backgroundListLimit: options.limit,
    backgroundCommandSurface: 'assistant',
  }))
assistant
  .command('job <jobId>')
  .description('Show or wait for a personal-assistant background job')
  .option(...urlOption)
  .option('--wait', 'Poll until the job reaches a terminal state')
  .option('--poll-ms <n>', 'Polling interval for --wait', '1500')
  .option('--output <file>', 'Write a completed answer to file')
  .option('--strip-fences', 'Strip the first Markdown code fence when writing --output')
  .action((jobId, options) => askCommand(undefined, {
    url: options.url,
    wait: options.wait,
    pollMs: options.pollMs,
    output: options.output,
    stripFences: options.stripFences,
    backgroundStatus: jobId,
    backgroundCommandSurface: 'assistant',
  }))
assistant
  .command('cancel <jobId>')
  .description('Cancel a personal-assistant background job')
  .option(...urlOption)
  .action((jobId, options) => askCommand(undefined, {
    url: options.url,
    backgroundCancel: jobId,
    backgroundCommandSurface: 'assistant',
  }))

const assistantNotifications = assistant
  .command('notifications')
  .description('List a bounded personal-assistant notification inbox')
  .option(...urlOption)
  .option('--unread', 'Only show unread notifications')
  .option('--limit <n>', 'Maximum notifications to show (1-200)', '20')
  .action(assistantNotificationsCommand)
assistantNotifications
  .command('show <id>')
  .description('Show one notification, including its body')
  .option(...urlOption)
  .action(assistantNotificationShowCommand)
assistantNotifications
  .command('read <id>')
  .description('Mark one notification as read')
  .option(...urlOption)
  .action(assistantNotificationReadCommand)
assistantNotifications
  .command('read-all')
  .description('Mark all CLI-visible notifications as read')
  .option(...urlOption)
  .action(assistantNotificationsReadAllCommand)

program
  .command('stats')
  .description('Compact daemon snapshot — session/approval/decision counts in one view')
  .option(...urlOption)
  .action(statsCommand)

program
  .command('state [sessionId]')
  .description('Show the agent state board for a run (goal, criteria, evidence, failed attempts, open questions)')
  .option(...urlOption)
  .option('--json', 'Print the raw board JSON')
  .action(stateCommand)

const diagnostics = program.command('diagnostics').description('Export local diagnostic data')
diagnostics
  .command('bundle')
  .description('Create a redacted observability support bundle')
  .option(...urlOption)
  .option('--range <range>', 'Range to include: 24h | 7d | 30d')
  .option('--limit <n>', 'Maximum events, crashes, and feedback rows per file')
  .option('--no-events', 'Exclude general observability events')
  .option('--no-crashes', 'Exclude crash-class events')
  .option('--no-feedback', 'Exclude feedback rows')
  .option('--no-health', 'Exclude health/readiness snapshot')
  .option('--no-persist', 'Preview bundle metadata without writing files')
  .action(diagnosticsBundleCommand)

const sessions = program.command('sessions').description('Manage sessions')
sessions
  .command('list')
  .description('List sessions')
  .option(...urlOption)
  .option('--query <text>', 'Search sessions')
  .option('--limit <n>', 'Limit displayed sessions (default 20, --limit 0 for all)', '20')
  .option('--status <status>', 'Narrow to active | completed | abandoned')
  .action(sessionsCommand)
sessions
  .command('show <id>')
  .description('Show session detail')
  .option(...urlOption)
  .option('--tail <n>', 'How many trailing events to print (default 25; --json prints all)')
  .action(sessionDetailCommand)
sessions
  .command('runbook <id>')
  .description('Build a diagnostic runbook for a session')
  .option(...urlOption)
  .action(sessionRunbookCommand)
sessions
  .command('resume <id>')
  .description('Resume a saved run checkpoint for a session')
  .option('--force', 'Resume even if the previous run looks stale or uncertain')
  .option(...urlOption)
  .action(sessionResumeCommand)
sessions
  .command('delete <id>')
  .description('Delete session')
  .option(...urlOption)
  .action(sessionDeleteCommand)
sessions
  .command('branch <id>')
  .description('Branch a session into a new fork')
  .option(
    '--from-event-index <n>',
    'Copy first N events (0-based count, e.g. 5 copies events 0..4). Omit to copy whole tail.',
  )
  .option(...urlOption)
  .action(sessionBranchCommand)
sessions
  .command('compact <id>')
  .description('Compact session context')
  .option(...urlOption)
  .action(sessionCompactCommand)
sessions
  .command('export <id>')
  .description('Export session')
  .option('--format <format>', 'Export format: markdown | json', 'markdown')
  .option('--output <file>', 'Write export to file')
  .option('--sanitize', 'Mask home path, emails and tokens in the export')
  .option(...urlOption)
  .action(sessionExportCommand)
sessions.action(sessionsCommand)

const schedule = program
  .command('schedule')
  .description('Manage daemon scheduled tasks (cron jobs)')
schedule
  .command('list')
  .description('List scheduled tasks')
  .option('--all', 'Include completed/cancelled/failed jobs')
  .option(...urlOption)
  .action(scheduleListCommand)
schedule
  .command('add <when> <instruction>')
  .description(
    'Schedule a task. <when> = cron ("0 9 * * *"), @daily/@hourly, @every 30s, or natural language ("매일 오전 9시", "in 2 minutes")',
  )
  .option('--name <name>', 'Job name (defaults to a slice of the instruction)')
  .option(
    '--timezone <tz>',
    'IANA timezone used to interpret cron (e.g. Asia/Seoul); blank = daemon default',
  )
  .option('--retries <n>', 'Max attempts per fire (1 = no retry)')
  .option('--backoff <ms>', 'Base retry backoff in ms (exponential, default 30000)')
  .option(
    '--skill <id>',
    'Stable skill id to load at every execution (repeatable, maximum 8 unique ids)',
    (value: string, previous: string[] | undefined) => [...(previous ?? []), value],
  )
  .option(
    '--unattended',
    'Explicitly authorize this job\'s future runs to execute approval-gated tools without a person present',
  )
  .option('--channel-type <type>', 'Deliver results through this configured channel type')
  .option('--channel-target <target>', 'Channel session key or raw target; requires --channel-type')
  .option('--reply-to <message-id>', 'Optional thread/message reply id; requires a channel route')
  .option(...urlOption)
  .action(scheduleAddCommand)
schedule
  .command('show <id>')
  .description('Show a scheduled task')
  .option(...urlOption)
  .action(scheduleShowCommand)
schedule
  .command('edit <id> <when> [instruction]')
  .description(
    'Edit a scheduled task without losing its run history. The new <when> must keep the existing one-shot/recurring kind.',
  )
  .option('--name <name>', 'Job name')
  .option('--timezone <tz>', 'IANA timezone used to interpret cron')
  .option('--retries <n>', 'Max attempts per fire (1 = no retry)')
  .option('--backoff <ms>', 'Base retry backoff in ms')
  .option(
    '--skill <id>',
    'Replace the scheduled skill selection (repeatable, maximum 8 unique ids)',
    (value: string, previous: string[] | undefined) => [...(previous ?? []), value],
  )
  .option('--clear-skills', 'Clear all selected skills while preserving other metadata')
  .option('--pause', 'Pause the task after editing')
  .option('--resume', 'Resume the task after editing')
  .option(...urlOption)
  .action(scheduleEditCommand)
schedule
  .command('reschedule <id> <when>')
  .description('Change only the schedule time/expression without losing run history')
  .option('--timezone <tz>', 'IANA timezone used to interpret cron')
  .option(...urlOption)
  .action(scheduleRescheduleCommand)
schedule
  .command('rm <id>')
  .alias('remove')
  .description('Cancel a scheduled task')
  .option(...urlOption)
  .action(scheduleRemoveCommand)
schedule
  .command('run <id>')
  .description('Trigger a scheduled task now (recorded in run history)')
  .option(
    '--no-delivery',
    'Suppress scheduler-owned notification, parent-chat, and configured channel delivery; agent tools still execute',
  )
  .option(...urlOption)
  .action(scheduleRunCommand)
schedule
  .command('runs <id>')
  .description('Show execution history, or one exact persisted run, for a scheduled task')
  .option('--limit <n>', 'Max entries (default 20)')
  .option('--run <run-id>', 'Show only this exact run under the scheduled task')
  .option(...urlOption)
  .action(scheduleRunsCommand)
schedule
  .command('pause <id>')
  .description('Pause a scheduled task')
  .option(...urlOption)
  .action(schedulePauseCommand)
schedule
  .command('resume <id>')
  .description('Resume a paused scheduled task')
  .option(...urlOption)
  .action(scheduleResumeCommand)
schedule
  .command('unattended <id> <state>')
  .description('Set durable headless authority for a job; <state> must be on or off')
  .option(...urlOption)
  .action(scheduleUnattendedCommand)
schedule
  .command('route <id> <channel-type> [channel-target]')
  .description('Set a paired channel delivery route, or pass channel-type "clear" to remove it')
  .option('--reply-to <message-id>', 'Optional thread/message reply id for this route')
  .option(...urlOption)
  .action(scheduleRouteCommand)
schedule.action(scheduleListCommand)

const agents = program.command('agents').description('Manage user-defined sub-agents')
agents
  .command('list')
  .description('List registered agents')
  .option(...urlOption)
  .action(agentsListCommand)
agents
  .command('create <id>')
  .description('Create or replace a user agent (.md file)')
  .requiredOption('--description <text>', 'One-line description')
  .option('--base <id>', 'Base preset (default: enhanced)')
  .option('--model <id>', 'Override model id')
  .option('--temperature <n>', 'Sampling temperature (0..2)')
  .option('--max-iterations <n>', 'Cap agent steps')
  .option('--prompt <text>', 'Inline system prompt body')
  .option('--prompt-file <path>', 'Read system prompt body from file')
  .option(...urlOption)
  .action(agentsCreateCommand)
agents
  .command('delete <id>')
  .description('Remove a user agent')
  .option(...urlOption)
  .action(agentsDeleteCommand)
agents.action(agentsListCommand)

const commands = program.command('commands').description('Manage user-defined slash commands')
commands
  .command('list')
  .description('List custom commands')
  .option(...urlOption)
  .action(commandsListCommand)
commands
  .command('create <id>')
  .description('Create or replace a custom command (.md file)')
  .requiredOption('--description <text>', 'One-line description')
  .option('--args <mode>', 'Argument mode: none | optional | required', 'optional')
  .option('--agent <id>', 'Override agent for this command')
  .option('--model <id>', 'Override model for this command')
  .option('--body <text>', 'Inline command body (prompt template)')
  .option('--body-file <path>', 'Read command body from file')
  .option(...urlOption)
  .action(commandsCreateCommand)
commands
  .command('delete <id>')
  .description('Remove a custom command')
  .option(...urlOption)
  .action(commandsDeleteCommand)
commands.action(commandsListCommand)

const skills = program.command('skills').description('Manage skills')
skills
  .command('list')
  .description('List skills')
  .option('--include-disabled', 'Include disabled skills')
  .option(...urlOption)
  .action(skillsCommand)
skills
  .command('installed')
  .description('List installed skills')
  .option('--include-disabled', 'Include disabled skills')
  .option(...urlOption)
  .action(skillsCommand)
skills
  .command('enable <id>')
  .description('Enable a skill')
  .option(...urlOption)
  .action(skillEnableCommand)
skills
  .command('disable <id>')
  .description('Disable a skill')
  .option(...urlOption)
  .action(skillDisableCommand)
skills
  .command('search <query>')
  .description('Search installed skills or registered marketplaces')
  .option('--remote', 'Search registered skill marketplaces')
  .option('--marketplace <name>', 'Search one marketplace')
  .option('--limit <n>', 'Maximum marketplace results', '20')
  .option(...urlOption)
  .action(skillSearchCommand)
skills
  .command('create <name>')
  .description('Create a new skill')
  .option('--description <text>', 'Skill description')
  .option('--tools <list>', 'Comma-separated tools', 'terminal.run')
  .option('--force', 'Overwrite existing skill and ignore validation warnings/errors')
  .option(...urlOption)
  .action(skillCreateCommand)
skills
  .command('install <source>')
  .description('Install a skill from marketplace, git URL, or SKILL.md URL')
  .option('--force', 'Ignore validation warnings and errors')
  .option(...urlOption)
  .action(skillInstallCommand)
skills
  .command('uninstall <name>')
  .description('Remove an installed skill')
  .option(...urlOption)
  .action(skillUninstallCommand)
skills
  .command('update [name]')
  .description('Update an installed skill')
  .option('--all', 'Update every installed skill')
  .option(...urlOption)
  .action(skillUpdateCommand)
skills.action(skillsCommand)

const pages = program.command('pages').description('Manage GitHub Pages Studio sites')
pages
  .command('init')
  .description('Scaffold an Astro + MDX static site for GitHub Pages')
  .option(...urlOption)
  .option('--repo-path <path>', 'Repository root (defaults to the current directory)')
  .option('--site-path <path>', 'Static site directory inside the repository (default: site)')
  .option('--name <name>', 'Human-readable site name')
  .option('--branch <branch>', 'GitHub Pages deploy branch (default: main)')
  .option('--no-workflow', 'Skip the GitHub Actions Pages workflow')
  .option('--overwrite', 'Overwrite existing scaffold files after review')
  .action(pagesInitCommand)
pages
  .command('scan [path]')
  .description('Scan a static site tree for publish blockers')
  .option(...urlOption)
  .option('--max-files <n>', 'Maximum files to inspect')
  .option('--max-bytes-per-file <n>', 'Maximum bytes to read per file')
  .action(pagesScanCommand)
pages
  .command('status')
  .description('Show GitHub Pages and latest Pages Studio workflow status')
  .option(...urlOption)
  .option('--repo <owner/name>', 'GitHub repository; overrides local git remote inference')
  .option('--repo-path <path>', 'Repository root used to infer owner/repo')
  .option('--workflow-file <file>', 'Pages workflow filename (default: pages-studio.yml)')
  .option('--branch <branch>', 'Only inspect workflow runs for this branch')
  .action(pagesStatusCommand)
pages.action(pagesStatusCommand)

const containers = program
  .command('containers')
  .description('Inspect / clean up sepilot-managed Docker containers')
containers.command('ls').description('List sepilot-managed containers').action(containersLsCommand)
containers
  .command('prune')
  .description('Remove stopped sepilot-managed containers')
  .option('--all', 'Remove all sepilot-managed containers, including running ones')
  .action(containersPruneCommand)
containers
  .command('rm <name|id>')
  .description('Remove one sepilot-managed container')
  .action(containersRmCommand)
containers.action(containersLsCommand)

const service = program.command('service').description('Manage durable daemon services')
service
  .command('list')
  .description('List durable services supervised by the daemon')
  .option(...urlOption)
  .action(serviceListCommand)
service
  .command('start [id]')
  .description('Start a durable process or container service')
  .option('--name <name>', 'Human-readable service name')
  .option('--backend <backend>', 'Backend: process | container')
  .option('--executable <path>', 'Process executable for backend=process')
  .option('--arg <value>', 'Process argument', collectRepeatedOption, [])
  .option('--cwd <path>', 'Working directory')
  .option('--env <pair>', 'Environment variable entry (KEY=VALUE)', collectRepeatedOption, [])
  .option('--image <image>', 'Container image for backend=container')
  .option('--runtime <name>', 'Container runtime command, such as docker or podman')
  .option('--container-name <name>', 'Container runtime name')
  .option('--command <value>', 'Container command argument', collectRepeatedOption, [])
  .option('--port <mapping>', 'Container port publish HOST:CONTAINER[/tcp|udp]', collectRepeatedOption, [])
  .option('--volume <mount>', 'Container volume mount SOURCE:TARGET[:ro]', collectRepeatedOption, [])
  .option('--restart <mode>', 'Restart mode: never | on-failure | always')
  .option('--max-restarts <n>', 'Max restarts for restart policy')
  .option('--backoff-ms <ms>', 'Restart backoff in milliseconds')
  .option('--health-process', 'Use process-liveness health checks')
  .option('--health-http <url>', 'Use HTTP health checks')
  .option('--health-tcp <host:port>', 'Use TCP health checks')
  .option('--health-interval-ms <ms>', 'Health check interval')
  .option('--health-timeout-ms <ms>', 'Health check timeout')
  .option('--health-grace-ms <ms>', 'Initial health check grace period')
  .option('--expected-status <code>', 'Expected HTTP status for --health-http')
  .option(...urlOption)
  .action(serviceStartCommand)
service
  .command('status <id>')
  .description('Show one durable service')
  .option(...urlOption)
  .action(serviceStatusCommand)
service
  .command('logs <id>')
  .description('Read service stdout and stderr logs')
  .option('--stdout-offset <bytes>', 'Previously consumed stdout offset')
  .option('--stderr-offset <bytes>', 'Previously consumed stderr offset')
  .option('--limit-bytes <bytes>', 'Maximum bytes to read per stream')
  .option('--tail-bytes <bytes>', 'Read the last N bytes when no offset is supplied')
  .option('--follow-ms <ms>', 'Long-poll for new bytes up to this duration')
  .option('--poll-interval-ms <ms>', 'Polling interval used with --follow-ms')
  .option(...urlOption)
  .action(serviceLogsCommand)
service
  .command('healthcheck <id>')
  .description('Run one service health check now')
  .option(...urlOption)
  .action(serviceHealthcheckCommand)
service
  .command('stop <id>')
  .description('Stop a durable service')
  .option('--signal <signal>', 'Signal such as SIGTERM or SIGKILL')
  .option('--timeout-ms <ms>', 'How long to wait for process exit')
  .option(...urlOption)
  .action(serviceStopCommand)
service
  .command('restart <id>')
  .description('Restart a durable service')
  .option(...urlOption)
  .action(serviceRestartCommand)
service
  .command('remove <id>')
  .alias('rm')
  .description('Remove a durable service record')
  .option('--force', 'Stop the service first if it is still running')
  .option('--delete-logs', 'Delete service log files')
  .option(...urlOption)
  .action(serviceRemoveCommand)
const nativeService = service.command('native').description('Manage native OS user services')
nativeService
  .command('install <id>')
  .description('Install a native user service through the daemon')
  .requiredOption('--executable <path>', 'Executable path to run')
  .option('--name <name>', 'Human-readable service name')
  .option('--description <text>', 'Native service description')
  .option('--provider <provider>', 'Provider: systemd-user | launchd-user')
  .option('--arg <value>', 'Command argument', collectRepeatedOption, [])
  .option('--cwd <path>', 'Working directory')
  .option('--env <pair>', 'Environment variable entry (KEY=VALUE)', collectRepeatedOption, [])
  .option('--restart <mode>', 'Restart mode: no | on-failure | always')
  .option('--enable', 'Enable the service after install')
  .option('--start', 'Start the service after install')
  .option('--skip-reload', 'Skip daemon/provider reload')
  .option(...urlOption)
  .action(nativeServiceInstallCommand)
nativeService
  .command('status <id>')
  .description('Show native service status')
  .option(...urlOption)
  .action(nativeServiceStatusCommand)
nativeService
  .command('logs <id>')
  .description('Read native service stdout and stderr logs')
  .option('--stdout-offset <bytes>', 'Previously consumed stdout offset')
  .option('--stderr-offset <bytes>', 'Previously consumed stderr offset')
  .option('--limit-bytes <bytes>', 'Maximum bytes to read per stream')
  .option('--tail-bytes <bytes>', 'Read the last N bytes when no offset is supplied')
  .option('--follow-ms <ms>', 'Long-poll for new bytes up to this duration')
  .option('--poll-interval-ms <ms>', 'Polling interval used with --follow-ms')
  .option(...urlOption)
  .action(nativeServiceLogsCommand)
nativeService
  .command('enable <id>')
  .description('Enable a native service')
  .option('--start', 'Start the service after enabling')
  .option('--skip-reload', 'Skip daemon/provider reload')
  .option(...urlOption)
  .action(nativeServiceEnableCommand)
nativeService
  .command('disable <id>')
  .description('Disable a native service')
  .option('--stop', 'Stop the service before disabling')
  .option('--skip-reload', 'Skip daemon/provider reload')
  .option(...urlOption)
  .action(nativeServiceDisableCommand)
nativeService
  .command('uninstall <id>')
  .description('Uninstall a native service and remove its unit file')
  .option('--stop', 'Stop the service before uninstalling')
  .option('--skip-reload', 'Skip daemon/provider reload')
  .option(...urlOption)
  .action(nativeServiceUninstallCommand)
service.action(serviceListCommand)

const marketplace = program.command('marketplace').description('Manage skill marketplaces')
marketplace
  .command('list')
  .description('List registered marketplaces')
  .option(...urlOption)
  .action(marketplaceListCommand)
marketplace
  .command('add <name> <url>')
  .description('Add a marketplace')
  .option(...urlOption)
  .action(marketplaceAddCommand)
marketplace
  .command('remove <name>')
  .description('Remove a marketplace')
  .option(...urlOption)
  .action(marketplaceRemoveCommand)

program
  .command('extension-create <name>')
  .description('Scaffold a local extension starter')
  .option('--dir <path>', 'Target directory')
  .option('--description <text>', 'Project description')
  .option('--template <name>', 'Starter template: minimal | full', 'minimal')
  .option('--workspace', 'Create the starter under the current pnpm workspace packages/ directory')
  .action(extensionCreateCommand)
program
  .command('extension-install <path>')
  .description('Issue a scoped token for a local extension manifest')
  .option('--expires-at <timestamp>', 'RFC3339 expiration timestamp for the generated token')
  .option(...urlOption)
  .action(extensionInstallCommand)

// `migrate` is a thin facade over /api/v1/migration/* — POSTs the
// run, polls until terminal, then prints the daemon-side report.
program
  .command('migrate')
  .description('Import sepilot_desktop user assets via the daemon migration runner')
  .option(...urlOption)
  .option('--source <path>', 'Source sepilot-desktop userData directory (default: OS convention)')
  .option('--steps <list>', 'Comma-separated step names to include')
  .option('--exclude <list>', 'Comma-separated step names to exclude')
  .option('--conflict <policy>', 'Conflict policy: skip | overwrite (default: skip)', 'skip')
  .option('--dry-run', 'Preview changes without writing', false)
  .option('--report-path <path>', 'Write the full migration report JSON to <path> after completion')
  .action(migrateFromSepilotDesktopCommand)
program
  .command('migrate:status <id>')
  .description('Print the snapshot for a daemon-resident migration by id')
  .option(...urlOption)
  .action(migrateStatusCommand)
program
  .command('migrate:cancel <id>')
  .description('Cancel a running migration by id')
  .option(...urlOption)
  .action(migrateCancelCommand)

program
  .command('run <skill>')
  .description('Execute a skill')
  .option(...urlOption)
  .option('--input <text>', 'Input for the skill')
  .option('--model <model>', 'Model to use')
  .option('--max-tokens <n>', 'Output token cap for each model response in this skill run')
  .option('--background', 'Start the skill run as a detached daemon background chat job')
  .option('--wait', 'With --background, poll until the background skill run completes')
  .option('--poll-ms <n>', 'Polling interval for --wait background skill runs', '1500')
  .option(
    '--max-iters <n>',
    "Agent loop iteration cap for this run (1–500; defaults to the daemon's SEPILOTD_CHAT_MAX_ITERATIONS or 50). Bump for multi-pass skills like software-architect on a non-trivial codebase.",
  )
  .action(runCommand)
program
  .command('config')
  .description('Show config')
  .option(...urlOption)
  .action(configCommand)
program
  .command('providers')
  .description('List LLM providers')
  .option(...urlOption)
  .action(providersCommand)
program
  .command('policy')
  .description('Show the active tool policy (modes, deny lists)')
  .option(...urlOption)
  .action(policyCommand)
program
  .command('autonomy [level]')
  .description('Show or set the agent autonomy level (readonly|accept-edits|workspace-write|supervised|autonomous)')
  .option(...urlOption)
  .action(autonomyCommand)
program
  .command('approve <requestId>')
  .description("Approve a pending tool-use request (mirrors the chat shell's /approve)")
  .option(...urlOption)
  .option('--session <id>', 'Session id (forwarded so the daemon can route the decision)')
  .option('--scope <scope>', 'once | session | always | run | session-all (default once)')
  .option('--reason <text>', 'Optional note forwarded to the agent')
  .action(approveCommand)
program
  .command('deny <requestId>')
  .description(
    'Deny a pending tool-use request (one-off; persistent denials live in the policy file)',
  )
  .option(...urlOption)
  .option('--session <id>', 'Session id (forwarded so the daemon can route the decision)')
  .option('--reason <text>', 'Optional note forwarded to the agent so it can pick a different path')
  .action(denyCommand)
program
  .command('answer <sessionId> <questionId> <answer...>')
  .description('Answer a pending agent question for a live session')
  .option(...urlOption)
  .action((sessionId, questionId, answerParts, options) =>
    answerCommand(sessionId, questionId, answerParts, options),
  )
program
  .command('steer <sessionId> <message...>')
  .description('Queue a mid-run steering note (instruction/question) onto a live, in-flight run')
  .option(...urlOption)
  .option('--kind <kind>', 'instruction | question (default instruction)')
  .action((sessionId, messageParts, options) => steerCommand(sessionId, messageParts, options))
const decisions = program
  .command('decisions')
  .description('Inspect / clear remembered approval decisions (--scope session|always rules)')
decisions
  .command('list', { isDefault: true })
  .description('List remembered approval decisions')
  .option(...urlOption)
  .option('--limit <n>', 'Limit displayed rules (default 20, --limit 0 for all)', '20')
  .option('--scope <scope>', 'session | always (omit for all)')
  .option('--tool <name>', 'Narrow to a single tool name (exact match, e.g. terminal.run)')
  .option('--stale', 'Show only rules with zero hits and createdAt ≥7d old (cleanup candidates)')
  .action(decisionsListCommand)
decisions
  .command('show <tool>')
  .description('Show verbose detail for every rule matching a tool name')
  .option(...urlOption)
  .action(decisionsShowCommand)
decisions
  .command('clear')
  .description('Clear remembered approval decisions, optionally narrowed by scope or session')
  .option(...urlOption)
  .option('--scope <scope>', 'session | always (omit to clear all scopes)')
  .option('--session <id>', 'Only clear decisions tied to this session id')
  .option(
    '--tool <name>',
    'Only clear rules with this exact tool name (pairs with `decisions list --tool`)',
  )
  .option(
    '--stale',
    'Only remove rules with zero hits and createdAt ≥7d old (combinable with --scope)',
  )
  .action(decisionsClearCommand)
program
  .command('devices')
  .description('List devices')
  .option(...urlOption)
  .action(devicesCommand)

const memory = program.command('memory').description('Manage memory')
memory
  .command('search <query>')
  .description('Search memory')
  .option(...urlOption)
  .option('--type <type>', 'Search type: semantic | keyword | hybrid')
  .option('--limit <n>', 'Maximum results')
  .action(memorySearchCommand)
const memoryDocuments = memory.command('documents').description('Manage indexed documents')
memoryDocuments
  .command('list')
  .description('List indexed documents')
  .option(...urlOption)
  .option('--query <text>', 'Filter documents by title, path, or tag')
  .option('--limit <n>', 'Maximum results')
  .action(memoryDocumentListCommand)
memoryDocuments
  .command('add')
  .description('Index a text document')
  .option(...urlOption)
  .requiredOption('--title <title>', 'Document title')
  .requiredOption('--content <content>', 'Document content')
  .option('--id <id>', 'Stable document id')
  .option('--path <path>', 'Document path or source path')
  .option('--mime-type <mimeType>', 'Document MIME type')
  .option('--source-file-id <id>', 'Source uploaded file id')
  .option('--tags <tags>', 'Comma-separated tags')
  .action(memoryDocumentAddCommand)
memoryDocuments
  .command('search <query>')
  .description('Search indexed documents')
  .option(...urlOption)
  .option('--type <type>', 'Search type: semantic | keyword | hybrid')
  .option('--limit <n>', 'Maximum results')
  .option('--document-id <id>', 'Filter to a single indexed document')
  .action(memoryDocumentSearchCommand)
memoryDocuments
  .command('show <id>')
  .description('Show indexed document metadata')
  .option(...urlOption)
  .action(memoryDocumentShowCommand)
memoryDocuments
  .command('delete <id>')
  .description('Delete indexed document')
  .option(...urlOption)
  .action(memoryDocumentDeleteCommand)
const memoryFile = memory.command('file').description('Manage markdown MEMORY.md and daily notes')
memoryFile
  .command('show [section]')
  .description('Show markdown memory sections or a single section')
  .option(...urlOption)
  .option('--daily <day>', 'Show daily note: today | yesterday')
  .action(memoryFileShowCommand)
memoryFile
  .command('set <section> <content...>')
  .description('Replace a markdown memory section')
  .option(...urlOption)
  .action(memoryFileSetCommand)
memoryFile
  .command('delete <section>')
  .description('Delete a markdown memory section')
  .option(...urlOption)
  .action(memoryFileDeleteCommand)
const memoryBacklog = memory
  .command('backlog')
  .description('Manage durable open-loop backlog items')
memoryBacklog
  .command('list', { isDefault: true })
  .description('Show open-loop queue and daily backlog')
  .option(...urlOption)
  .option('--daily <day>', 'Daily backlog to include: today | yesterday | both')
  .option('--limit <n>', 'Maximum items per section')
  .option('--reflections', 'Include reflection ledger entries')
  .action(memoryBacklogListCommand)
memoryBacklog
  .command('add <content...>')
  .description('Add an open-loop backlog item')
  .option(...urlOption)
  .action(memoryBacklogAddCommand)
memoryBacklog
  .command('done <query...>')
  .description('Resolve open-loop backlog items matching text')
  .option(...urlOption)
  .action(memoryBacklogDoneCommand)
memory
  .command('add <content>')
  .description('Add memory entry')
  .option(...urlOption)
  .option('--tags <tags>', 'Comma-separated tags')
  .action(memoryAddCommand)
memory
  .command('status')
  .description('Show semantic memory status')
  .option(...urlOption)
  .action(memoryStatusCommand)
memory
  .command('lifecycle')
  .description('Show semantic memory lifecycle status')
  .option(...urlOption)
  .option('--stale-after-days <n>', 'Age threshold for stale conversation memories')
  .option('--low-importance <n>', 'Importance threshold between 0 and 1')
  .action(memoryLifecycleCommand)
memory
  .command('audit')
  .description('List semantic memory audit entries')
  .option(...urlOption)
  .option('--memory-id <id>', 'Filter audit entries to one memory id')
  .option('--limit <n>', 'Maximum audit entries')
  .action(memoryAuditCommand)
memory
  .command('security-audit')
  .description('List denied memory scope bypass audit events')
  .option(...urlOption)
  .option('--limit <n>', 'Maximum audit events')
  .option('--since <timestamp>', 'Only include events at or after this timestamp')
  .option('--actor <actor>', 'Filter by actor')
  .option('--auth-kind <kind>', 'Filter by auth kind')
  .option('--route <route>', 'Filter by route')
  .action(memorySecurityAuditCommand)
memory
  .command('scopes')
  .description('List active memory scopes and legacy global entries')
  .option(...urlOption)
  .action(memoryScopesCommand)
memory
  .command('transfer <source> <target>')
  .description('Preview or run a memory scope transfer')
  .option(...urlOption)
  .option('--apply', 'Execute the transfer; default is a dry-run preview')
  .option('--confirm-global', 'Required when applying a legacy global memory transfer')
  .option('--ids <ids>', 'Comma-separated semantic memory ids to transfer')
  .option('--limit <n>', 'Maximum semantic memories to transfer')
  .option('--include-file', 'Also move the scoped markdown file bucket when possible')
  .option('--no-reminders', 'Do not retag pending reminders')
  .option('--reason <text>', 'Audit reason')
  .action(memoryScopeTransferCommand)
memory
  .command('maintenance')
  .description('Preview or run semantic memory lifecycle maintenance')
  .option(...urlOption)
  .option('--apply', 'Actually prune entries; default is a dry-run preview')
  .option('--max-age-days <n>', 'Maximum age for prune candidates')
  .option('--max-importance <n>', 'Maximum importance between 0 and 1')
  .option('--reason <text>', 'Audit reason')
  .action(memoryMaintenanceCommand)
memory
  .command('reindex')
  .description('Rebuild semantic memory index')
  .option(...urlOption)
  .action(memoryReindexCommand)

const rag = program
  .command('rag')
  .description('Manage local RAG sources')
  .option(...urlOption)
rag
  .command('sources')
  .description('List local RAG sources')
  .option(...urlOption)
  .action(ragSourcesListCommand)
rag
  .command('add <path>')
  .description('Add a local git/worktree folder as a RAG source')
  .option(...urlOption)
  .option('--id <id>', 'Stable source id')
  .option('--name <name>', 'Source display name')
  .option('--include <patterns>', 'Comma-separated glob patterns to include')
  .option('--exclude <patterns>', 'Comma-separated glob patterns to exclude')
  .option('--no-sync', 'Register the source without syncing immediately')
  .action(ragSourceAddCommand)
rag
  .command('sync')
  .description('Sync local RAG sources into the sqlite-vec document index')
  .option(...urlOption)
  .action(ragSyncCommand)
rag
  .command('search <query>')
  .description('Search local RAG documents')
  .option(...urlOption)
  .option('--limit <n>', 'Maximum results')
  .action(ragSearchCommand)
rag
  .command('documents <folderId>')
  .description('List documents indexed for one RAG source')
  .option(...urlOption)
  .action(ragDocumentsListCommand)
rag
  .command('status')
  .description('Show local RAG vector backend status')
  .option(...urlOption)
  .action(ragStatusCommand)
rag.action(ragSourcesListCommand)

const usage = program.command('usage').description('Show usage stats')
usage
  .command('summary')
  .description('Total usage summary')
  .option(...urlOption)
  .action(usageCommand)
usage
  .command('daily')
  .description('Daily breakdown')
  .option(...urlOption)
  .option('--days <days>', 'Number of days')
  .action(usageDailyCommand)
usage.action(usageCommand)

program
  .command('init')
  .description('Initialize sepilotd')
  .option('--name <name>', 'Device name')
  .option('--role <role>', 'Device role')
  .option('--no-wizard', 'Skip interactive provider/model setup')
  .action(initCommand)
program.command('doctor').description('Security health check').action(doctorCommand)

program
  .command('logs')
  .description('View logs')
  .option('--lines <n>', 'Number of lines', '50')
  .option('--type <type>', 'Log type: audit | daemon', 'audit')
  .action(logsCommand)

program
  .command('trace')
  .description('Inspect SEPILOT_DEBUG=1 agent diagnostics (agent-trace.jsonl)')
  .option('--last <n>', 'Show the last N matching entries', '20')
  .option('--session <id>', 'Filter by session id (prefix match allowed)')
  .option('--event <type>', 'Filter by event type (llm.call, graph.node.end, tool.execution.end, ...)')
  .option('--since <iso>', 'Only entries at or after this ISO timestamp')
  .option('--until <iso>', 'Only entries at or before this ISO timestamp')
  .option('--grep <text>', 'Only entries whose JSON contains this substring (case-insensitive)')
  .option('--full', 'Expand request/response messages instead of a one-line summary')
  .option(
    '--stats',
    'Print aggregate stats (per event, model, session, errors, tokens) instead of entries',
  )
  .option('--file <path>', 'Read a trace file from this path instead of the default')
  .action(traceCommand)

const channel = program.command('channel').description('Manage channels')
channel
  .command('list')
  .description('List channels')
  .option(...urlOption)
  .action(channelListCommand)
channel
  .command('add <type>')
  .description('Add or replace a provider-specific channel')
  .option('--bot-token <token>', 'Telegram bot token (prefer a secure environment or prompt wrapper)')
  .option('--from-env', 'For Mattermost, read credentials from MATTERMOST_BOT_TOKEN and MATTERMOST_WEBHOOK_TOKEN')
  .option('--server-url <url>', 'Mattermost server URL (or set MATTERMOST_SERVER_URL)')
  .option('--team <id>', 'Mattermost team id allowed to invoke the agent', collectRepeatedOption, [])
  .option('--channel <id>', 'Mattermost channel id allowed to invoke the agent', collectRepeatedOption, [])
  .option('--user <id>', 'Allowed Telegram user or Mattermost user id', collectRepeatedOption, [])
  .option('--no-pairing-required', 'Allow all Telegram users without pairing')
  .option('--pairing-code-ttl <seconds>', 'Pairing code TTL in seconds')
  .option('--rate-limit-per-minute <n>', 'Per-user Telegram message rate limit')
  .option('--disabled', 'Create the channel in disabled state')
  .option(...urlOption)
  .action(channelAddCommand)
channel
  .command('remove <type>')
  .description('Remove a provider-specific channel')
  .option(...urlOption)
  .action(channelRemoveCommand)
channel
  .command('enable <type>')
  .description('Enable a provider-specific channel')
  .option(...urlOption)
  .action(channelEnableCommand)
channel
  .command('disable <type>')
  .description('Disable a provider-specific channel')
  .option(...urlOption)
  .action(channelDisableCommand)
channel
  .command('pair <type>')
  .description('Pair a channel')
  .option(...urlOption)
  .action(channelPairCommand)
channel
  .command('users <type>')
  .description('List paired users for a provider-specific channel')
  .option(...urlOption)
  .action(channelUsersCommand)
channel
  .command('unpair <type> <userId>')
  .description('Revoke a paired user from a provider-specific channel')
  .option(...urlOption)
  .action(channelUnpairCommand)
channel.action(channelListCommand)

const tokens = program.command('tokens').description('Manage daemon-issued extension tokens')
tokens
  .command('list')
  .description('List daemon-issued extension tokens')
  .option(...urlOption)
  .action(tokenListCommand)
tokens
  .command('issue <label>')
  .description('Issue a scoped daemon extension token')
  .requiredOption('--scope <scope>', 'Granted scope', collectRepeatedOption, [])
  .option('--expires-at <timestamp>', 'RFC3339 expiration timestamp')
  .option(...urlOption)
  .action(tokenIssueCommand)
tokens
  .command('revoke <id>')
  .description('Revoke a daemon-issued extension token')
  .option(...urlOption)
  .action(tokenRevokeCommand)
tokens.action(tokenListCommand)

program
  .command('version')
  .description('Show version and check updates')
  .option('--check', 'Check for updates')
  .option(...urlOption)
  .action(versionCommand)
program
  .command('upgrade')
  .description('Update the standalone sepilot binary to the latest release')
  .option('--check', 'Only check whether an update is available')
  .option('--force', 'Re-download even if already up to date')
  .action(upgradeCommand)
program
  .command('backup')
  .description('Backup sepilotd data')
  .option('--output <path>', 'Output file path')
  .action(backupCommand)
program
  .command('restore <file>')
  .description('Restore from backup')
  .option('--target <dir>', 'Parent directory to extract into (default: $HOME)')
  .option('--yes', 'Skip the 3-second confirmation countdown (use in scripts)')
  .action(restoreCommand)
program
  .command('completions')
  .description('Generate shell completions')
  .option('--shell <shell>', 'Shell type: bash | zsh | fish')
  .action(completionsCommand)

program
  .command('start')
  .description('Start the daemon')
  .option(...urlOption)
  .option('--foreground', 'Run in foreground')
  .option(
    '--forward-env <name>',
    'Set the saved daemon forwarding list to one named environment variable without putting its value in argv (repeatable)',
    collectRepeatedOption,
    [],
  )
  .action(startCommand)
program
  .command('stop')
  .description('Stop the daemon')
  .option(...urlOption)
  .action(stopCommand)
program
  .command('restart')
  .description('Restart the daemon')
  .option(...urlOption)
  .option(
    '--forward-env <name>',
    'Set the saved daemon forwarding list to one named environment variable without putting its value in argv (repeatable)',
    collectRepeatedOption,
    [],
  )
  .option(
    '--clear-forward-env',
    'Intentionally remove all saved daemon environment forwarding',
  )
  .action(restartCommand)

const daemonGroup = program
  .command('daemon')
  .description('Daemon lifecycle and service helpers')
daemonGroup
  .command('start')
  .description('Start the daemon')
  .option(...urlOption)
  .option('--foreground', 'Run in foreground')
  .option(
    '--forward-env <name>',
    'Set the saved daemon forwarding list to one named environment variable without putting its value in argv (repeatable)',
    collectRepeatedOption,
    [],
  )
  .action(startCommand)
daemonGroup
  .command('stop')
  .description('Stop the daemon')
  .option(...urlOption)
  .action(stopCommand)
daemonGroup
  .command('restart')
  .description('Restart the daemon')
  .option(...urlOption)
  .option(
    '--forward-env <name>',
    'Set the saved daemon forwarding list to one named environment variable without putting its value in argv (repeatable)',
    collectRepeatedOption,
    [],
  )
  .option(
    '--clear-forward-env',
    'Intentionally remove all saved daemon environment forwarding',
  )
  .action(restartCommand)
daemonGroup
  .command('status')
  .description('Check daemon status')
  .option(...urlOption)
  .option('--report', 'Print a detailed health report')
  .option('--output <file>', 'Write the health report to file')
  .action(statusCommand)
daemonGroup
  .command('install-service')
  .description('Install a systemd user unit so sepilotd starts on login (Linux)')
  .option('--no-enable', 'Write the unit file without running systemctl enable')
  .action((opts: { enable?: boolean }) =>
    installServiceCommand({ noEnable: opts.enable === false }),
  )
daemonGroup
  .command('uninstall-service')
  .description('Remove the systemd user unit and disable auto-start (Linux)')
  .action(() => uninstallServiceCommand())
daemonGroup
  .command('service-status')
  .description('Show the systemd user unit status (Linux)')
  .action(() => showServiceStatus())

program
  .command('ask [question]')
  .description('Ask a one-shot question (supports stdin pipe)')
  .option(...urlOption)
  .option('--model <model>', 'Model')
  .option('--provider <provider>', 'Provider')
  .option('--max-tokens <n>', 'Output token cap for this answer')
  .option('--persona <persona>', 'Persona')
  .option(
    '--mode <mode>',
    'Agent mode or registered graph id (defaults to react). react = general-purpose tool execution; instant = chat and read-only knowledge',
  )
  .option('--interactive', 'Force rich agent progress instead of answer-only one-shot output')
  .option(
    '--autonomy <level>',
    perTurnAutonomyDescription,
  )
  .option('--thinking-level <level>', 'Per-turn thinking level: auto, off, low, medium, high, or max')
  .option('--skill <id>', 'Select a daemon skill for this ask (repeatable, maximum 8)', collectRepeatedOption, [])
  .option('--panel-preset <id>', 'Persona panel preset id (e.g. architecture-qaw-atam)')
  .option(
    '--background',
    'Start the request as a detached daemon background chat job',
  )
  .option(
    '--wait',
    'With --background or --background-status, poll until the job reaches a terminal state',
  )
  .option(
    '--background-status <jobId>',
    'Show a detached background chat job status',
  )
  .option(
    '--background-cancel <jobId>',
    'Cancel a detached background chat job',
  )
  .option('--background-list', 'List detached background chat jobs')
  .option('--poll-ms <n>', 'Polling interval for --wait background jobs', '1500')
  .option(
    '--session <id>',
    'Continue an existing session so the daemon keeps the conversation history',
  )
  .option('--output <file>', 'Write output to file')
  .option(
    '--strip-fences',
    'When writing --output, strip the first markdown ``` code fence so the file is directly executable',
  )
  .action(askCommand)

program
  .command('history <query>')
  .description('Search conversation history')
  .option(...urlOption)
  .option('--limit <n>', 'Max results', '20')
  .action(historyCommand)

program.command('config-edit').description('Edit config.yaml in editor').action(configEditCommand)

program
  .command('config-set <key> <value>')
  .description('Set a config value (e.g. agent.autonomy supervised)')
  .option(...urlOption)
  .action((key: string, value: string, options: { url?: string }) =>
    configSetCommand(key, value, options),
  )

function collectRepeatedOption(value: string, previous: string[] = []): string[] {
  return [...previous, value]
}

const mcp = program.command('mcp').description('Manage MCP servers')
mcp
  .command('list')
  .description('List configured MCP servers')
  .option(...urlOption)
  .action(mcpListCommand)
mcp
  .command('add <name>')
  .description('Add or replace an MCP server')
  .option(
    '--transport <transport>',
    'MCP transport: stdio, http (Streamable HTTP), or sse (legacy)',
  )
  .option('--command <command>', 'Command used to start a stdio MCP server')
  .option('--arg <value>', 'Additional command argument', collectRepeatedOption, [])
  .option('--env <pair>', 'Environment variable entry (KEY=VALUE)', collectRepeatedOption, [])
  .option('--server-url <url>', 'Remote MCP endpoint URL for http/sse transports')
  .option(
    '--header <pair>',
    'HTTP header entry for remote MCP transports (KEY=VALUE)',
    collectRepeatedOption,
    [],
  )
  .option('--disabled', 'Create the server in disabled state')
  .option(...urlOption)
  .action(mcpAddCommand)
mcp
  .command('remove <name>')
  .description('Remove an MCP server')
  .option(...urlOption)
  .action(mcpRemoveCommand)
mcp
  .command('enable <name>')
  .description('Enable an MCP server')
  .option(...urlOption)
  .action(mcpEnableCommand)
mcp
  .command('disable <name>')
  .description('Disable an MCP server')
  .option(...urlOption)
  .action(mcpDisableCommand)
mcp
  .command('trust-manifest <name>')
  .description('Trust a quarantined MCP server tool manifest after review')
  .option(...urlOption)
  .action(mcpTrustManifestCommand)
const mcpPrompts = mcp.command('prompts').description('Inspect MCP prompts')
mcpPrompts
  .command('list [server]')
  .description('List MCP prompts')
  .option(...urlOption)
  .action(mcpPromptsListCommand)
mcpPrompts
  .command('get <ref>')
  .description('Get an MCP prompt (server/prompt form)')
  .option('--arg <entries...>', 'Prompt argument key=value (repeatable)')
  .option(...urlOption)
  .action(mcpPromptsGetCommand)
const mcpResources = mcp.command('resources').description('Inspect MCP resources')
mcpResources
  .command('list [server]')
  .description('List MCP resources')
  .option(...urlOption)
  .action(mcpResourcesListCommand)
mcpResources
  .command('templates [server]')
  .description('List MCP resource URI templates')
  .option(...urlOption)
  .action(mcpResourcesTemplatesCommand)
mcpResources
  .command('read <server> <uri>')
  .description('Read an MCP resource by URI')
  .option(...urlOption)
  .action(mcpResourcesReadCommand)
mcpResources
  .command('subscribe <server> <uri>')
  .description('Subscribe to MCP resource updates')
  .option(...urlOption)
  .action(mcpResourcesSubscribeCommand)
mcpResources
  .command('unsubscribe <server> <uri>')
  .description('Unsubscribe from MCP resource updates')
  .option(...urlOption)
  .action(mcpResourcesUnsubscribeCommand)
mcpResources
  .command('subscriptions <server>')
  .description('List MCP resource subscriptions')
  .option(...urlOption)
  .action(mcpResourcesSubscriptionsCommand)
mcpResources
  .command('updates <server>')
  .description('List received MCP resource update notifications')
  .option('--limit <n>', 'Maximum updates to show')
  .option(...urlOption)
  .action(mcpResourcesUpdatesCommand)
mcp
  .command('complete <server>')
  .description('Request MCP completion suggestions')
  .option('--prompt <name>', 'Complete a prompt argument')
  .option('--resource <uri>', 'Complete a resource template argument')
  .requiredOption('--arg <key=value>', 'Argument name and current value')
  .option('--context <key=value>', 'Additional argument context', collectRepeatedOption, [])
  .option(...urlOption)
  .action(mcpCompletionCommand)
const mcpLogging = mcp.command('logging').description('Inspect MCP logging')
mcpLogging
  .command('set-level <server> <level>')
  .description('Set MCP server logging level')
  .option(...urlOption)
  .action(mcpLoggingSetLevelCommand)
mcpLogging
  .command('logs <server>')
  .description('Show received MCP log messages')
  .option(...urlOption)
  .action(mcpLoggingLogsCommand)
mcp
  .command('metrics [server]')
  .description('Show MCP tool call metrics')
  .option(...urlOption)
  .action(mcpMetricsCommand)
mcp
  .command('doctor [server]')
  .description('Check MCP connector readiness')
  .option(...urlOption)
  .action(mcpDoctorCommand)
mcp
  .command('search <query>')
  .description('Search MCP server marketplaces')
  .option(...urlOption)
  .action(mcpSearchCommand)
mcp
  .command('install <name>')
  .description('Install an MCP server from a marketplace')
  .option('--marketplace <name>', 'Specific marketplace to install from')
  .option(
    '--var <pair>',
    'Template variable entry for install-time placeholders (KEY=VALUE)',
    collectRepeatedOption,
    [],
  )
  .option(...urlOption)
  .action(mcpInstallCommand)
mcp
  .command('playwright')
  .description('Configure Playwright MCP browser automation')
  .option('--name <name>', 'MCP server name', 'playwright')
  .option('--mode <mode>', 'Browser mode: auto, visible, or headless', 'auto')
  .option('--browser <browser>', 'Browser/channel: chromium, chrome, msedge, firefox, or webkit')
  .option('--user-data-dir <path>', 'Persistent browser profile directory')
  .option('--caps <list>', 'Comma-separated Playwright MCP capabilities: vision,pdf,devtools')
  .option('--mcp-package <spec>', 'Playwright MCP npm package spec', '@playwright/mcp@latest')
  .option('--isolated', 'Use an isolated in-memory browser profile')
  .option('--storage-state <path>', 'Initial storage state JSON for isolated sessions')
  .option('--output-dir <path>', 'Directory for Playwright MCP output files')
  .option('--viewport-size <size>', 'Viewport size, for example 1280x720')
  .option('--user-agent <value>', 'Browser user agent string')
  .option('--device <name>', 'Playwright device to emulate, for example "iPhone 15"')
  .option(
    '--proxy-server <url>',
    'Proxy server, for example http://host:3128 or socks5://host:1080',
  )
  .option('--proxy-bypass <domains>', 'Comma-separated proxy bypass domains')
  .option('--timeout-action <ms>', 'Playwright MCP action timeout in milliseconds')
  .option('--timeout-navigation <ms>', 'Playwright MCP navigation timeout in milliseconds')
  .option('--image-responses <mode>', 'Image responses: allow or omit')
  .option('--allowed-origins <origins>', 'Semicolon-separated request origin allowlist')
  .option('--blocked-origins <origins>', 'Semicolon-separated request origin blocklist')
  .option('--no-sandbox', 'Pass --no-sandbox to Playwright MCP')
  .option('--ignore-https-errors', 'Ignore HTTPS certificate errors')
  .option('--block-service-workers', 'Block service workers')
  .option('--save-session', 'Save Playwright MCP session into the output directory')
  .option('--disabled', 'Create the server in disabled state')
  .option(...urlOption)
  .action(mcpPlaywrightCommand)
mcp
  .command('call <server> <tool>')
  .description('Call a tool on an MCP server')
  .option('--input <json>', 'JSON object to pass as tool arguments')
  .option('--input-file <path>', 'Read tool arguments JSON object from a file, or - for stdin')
  .option(...urlOption)
  .action(mcpToolsCallCommand)

const mcpTools = mcp.command('tools').description('Manage per-tool enable/disable')
mcpTools
  .command('list [server]')
  .description('List tools per MCP server (enabled/disabled)')
  .option(...urlOption)
  .action(mcpToolsListCommand)
mcpTools
  .command('disable <server> <tool>')
  .description('Disable a specific tool on an MCP server')
  .option(...urlOption)
  .action(mcpToolsDisableCommand)
mcpTools
  .command('enable <server> <tool>')
  .description('Enable a specific tool on an MCP server')
  .option(...urlOption)
  .action(mcpToolsEnableCommand)

const mcpMp = mcp.command('marketplace').description('Manage MCP server marketplaces')
mcpMp
  .command('list')
  .description('List MCP marketplaces')
  .option(...urlOption)
  .action(mcpMpListCommand)
mcpMp
  .command('add <name> <url>')
  .description('Add an MCP marketplace')
  .option(...urlOption)
  .action(mcpMpAddCommand)
mcpMp
  .command('remove <name>')
  .description('Remove an MCP marketplace')
  .option(...urlOption)
  .action(mcpMpRemoveCommand)
mcp.action(mcpListCommand)

const hooks = program.command('hooks').description('Manage outbound webhooks')
hooks.command('commands').description('Inspect loaded command hooks without exposing shell command secrets').option(...urlOption).action(hooksCommandsCommand)
hooks
  .command('list')
  .description('List configured outbound webhooks')
  .option(...urlOption)
  .action(hooksListCommand)
hooks
  .command('add <url>')
  .description('Add or replace an outbound webhook')
  .requiredOption('--event <event>', 'Hook event name', collectRepeatedOption, [])
  .option('--header <pair>', 'HTTP header entry (KEY=VALUE)', collectRepeatedOption, [])
  .option('--secret <value>', 'Shared secret used for X-Signature generation')
  .option(
    '--retry-attempts <n>',
    'Maximum delivery attempts before marking the webhook delivery failed',
  )
  .option('--retry-backoff-ms <ms>', 'Base retry backoff in milliseconds')
  .option('--disabled', 'Create the webhook in disabled state')
  .option(...urlOption)
  .action(hooksAddCommand)
hooks
  .command('deliveries [id]')
  .description('List recent outbound webhook delivery events')
  .option('--limit <n>', 'Maximum number of delivery events to show')
  .option('--status <status>', 'Filter by delivery status (success|error)')
  .option('--cursor <cursor>', 'Continue listing from a previous next cursor')
  .option(...urlOption)
  .action(hooksDeliveriesCommand)
hooks
  .command('dead-letters [id]')
  .description('List unresolved outbound webhook dead letters')
  .option('--limit <n>', 'Maximum number of dead-letter chains to show')
  .option('--state <state>', 'Filter by dead-letter state (open|acknowledged|all)')
  .option('--cursor <cursor>', 'Continue listing from a previous next cursor')
  .option(...urlOption)
  .action(hooksDeadLettersCommand)
hooks
  .command('ack <rootDeliveryId>')
  .description('Acknowledge an outbound webhook dead letter')
  .option('--note <text>', 'Optional acknowledgment note')
  .option(...urlOption)
  .action(hooksAckCommand)
hooks
  .command('replay <deliveryId>')
  .description('Replay a recorded outbound webhook delivery')
  .option('--force', 'Replay even if the original delivery succeeded or was already replayed')
  .option(...urlOption)
  .action(hooksReplayCommand)
hooks
  .command('replay-failed [id]')
  .description('Replay unresolved outbound webhook dead letters')
  .option('--limit <n>', 'Maximum number of dead-letter chains to replay')
  .option('--cursor <cursor>', 'Continue selecting dead letters from a previous next cursor')
  .option('--force', 'Force replay even when the selected delivery already has replay descendants')
  .option(...urlOption)
  .action(hooksReplayFailedCommand)
hooks
  .command('remove <id>')
  .description('Remove an outbound webhook')
  .option(...urlOption)
  .action(hooksRemoveCommand)
hooks
  .command('enable <id>')
  .description('Enable an outbound webhook')
  .option(...urlOption)
  .action(hooksEnableCommand)
hooks
  .command('disable <id>')
  .description('Disable an outbound webhook')
  .option(...urlOption)
  .action(hooksDisableCommand)
hooks.action(hooksListCommand)

const webhooks = program.command('webhooks').description('Manage generic inbound webhook triggers')
webhooks
  .command('list')
  .description('List configured generic inbound webhook endpoints')
  .option(...urlOption)
  .action(webhooksListCommand)
webhooks
  .command('add <path>')
  .description('Add or replace a generic inbound webhook endpoint')
  .requiredOption('--header <header>', 'HTTP header used for the HMAC signature')
  .requiredOption('--secret <value>', 'Shared secret used to verify the webhook payload')
  .option('--event <event>', 'Allowed event name', collectRepeatedOption, [])
  .option('--ip <cidr>', 'Allowed IP address or CIDR entry', collectRepeatedOption, [])
  .option('--disabled', 'Create the webhook endpoint in disabled state')
  .option(...urlOption)
  .action(webhooksAddCommand)
webhooks
  .command('remove <id>')
  .description('Remove a generic inbound webhook endpoint')
  .option(...urlOption)
  .action(webhooksRemoveCommand)
webhooks
  .command('enable <id>')
  .description('Enable a generic inbound webhook endpoint')
  .option(...urlOption)
  .action(webhooksEnableCommand)
webhooks
  .command('disable <id>')
  .description('Disable a generic inbound webhook endpoint')
  .option(...urlOption)
  .action(webhooksDisableCommand)
webhooks.action(webhooksListCommand)

const secrets = program.command('secrets').description('Manage encrypted secrets')
secrets
  .command('set <key> [value]')
  .description('Set a secret')
  .option('--stdin', 'Read value from stdin')
  .option(...urlOption)
  .action(secretsSetCommand)
secrets
  .command('list')
  .description('List secret key names')
  .option(...urlOption)
  .action(secretsListCommand)
secrets
  .command('remove <key>')
  .description('Remove a secret')
  .option(...urlOption)
  .action(secretsRemoveCommand)

const auth = program.command('auth').description('Manage provider authentication')
auth
  .command('login')
  .description('Interactively configure and validate a provider')
  .option(...urlOption)
  .action((options) => authLoginCommand(options))
auth
  .command('list')
  .description('List providers with health status and masked key source')
  .option(...urlOption)
  .action(authListCommand)
auth
  .command('logout <provider>')
  .description('Clear stored auth for a provider (keeps the provider entry)')
  .option(...urlOption)
  .action(authLogoutCommand)

program
  .command('test')
  .description('Test all connections')
  .option(...urlOption)
  .action(testConnectionCommand)

program
  .command('batch <file>')
  .description('Process tasks from JSONL file')
  .option(...urlOption)
  .option('--output <file>', 'Output JSONL file')
  .option('--model <model>', 'Model to use')
  .option('--concurrency <n>', 'Concurrent tasks', '1')
  .option('--detach', 'Submit the job and print the jobId without waiting')
  .option('--strict', 'Exit non-zero (2) when any task fails')
  .action(batchCommand)

program
  .command('batch:resume <jobId>')
  .description('Resume polling an in-flight or completed batch job')
  .option(...urlOption)
  .option('--output <file>', 'Output JSONL file')
  .action(batchResumeCommand)

program
  .command('batch:cancel <jobId>')
  .description('Cancel a running batch job by id')
  .option(...urlOption)
  .action(batchCancelCommand)

program
  .command('batch:status <jobId>')
  .description('Print the status snapshot for a batch job')
  .option(...urlOption)
  .action(batchStatusCommand)

const jobs = program.command('jobs').description('Manage daemon background jobs')
program.command('rewind <sessionId> [checkpointId]').description('Preview or apply safe file/conversation rewind; original conversations remain recoverable')
  .option(...urlOption)
  .addOption(new Option('--scope <scope>', 'Rewind scope').choices(['files', 'conversation', 'both']).default('files'))
  .option('--apply', 'Apply the reviewed rewind; without this flag, only preview')
  .option('--turns <n>', 'Conversation user turns to rewind', '1')
  .action(rewindCommand)
program.command('tasks [args...]').description('Unified jobs, agents, processes, schedules, services, approvals and inbox')
  .allowUnknownOption(true).option(...urlOption).action(tasksCommand)
jobs.command('list', { isDefault: true })
  .description('List background jobs, including work needing approval')
  .option(...urlOption)
  .option('--status <status>', 'Filter by lifecycle state')
  .option('--kind <kind>', 'Filter by job kind')
  .option('--limit <n>', 'Page size (1-100)', '30')
  .option('--offset <n>', 'Page offset', '0')
  .action(jobsListCommand)
jobs
  .command('resume <jobId>')
  .description('Resume polling an in-flight or completed daemon background job')
  .option(...urlOption)
  .option('--output <file>', 'Output JSONL file')
  .action(batchResumeCommand)
jobs
  .command('cancel <jobId>')
  .description('Cancel a running daemon background job by id')
  .option(...urlOption)
  .action(batchCancelCommand)
jobs
  .command('status <jobId>')
  .description('Print the status snapshot for a daemon background job')
  .option(...urlOption)
  .action(batchStatusCommand)

const subagent = program.command('subagent').description('Dispatch ad-hoc isolated subagents')
subagent
  .command('categories')
  .description('List daemon-supported subagent delegation categories')
  .option(...urlOption)
  .action(subagentCategoriesCommand)
subagent
  .command('dispatch <prompt>')
  .description('Run an isolated subagent with the given prompt and return its final output')
  .option('--system <text>', 'Override system prompt')
  .option('--system-file <path>', 'Read system prompt from file')
  .option('--category <id>', 'Use a daemon delegation category preset')
  .option('--agent <id>', 'Use a registered user-defined agent')
  .option('--max-iterations <n>', 'Max turns (default 20, hard cap 50)')
  .option('--tools <a,b,c>', 'Comma-separated subset of parent allowed tools')
  .option('--model <id>', 'Provider model override')
  .option('--parent-session <id>', 'Tag the subagent session with this parent id')
  .option('--background', 'Queue the subagent as a daemon background job')
  .addOption(new Option('--isolation <kind>', 'Use a clean HEAD checkout; retain changes for explicit merge').choices(['worktree']))
  .option(...urlOption)
  .action(subagentDispatchCommand)

const plans = program.command('plans').description('Manage durable daemon work plans')
plans
  .command('list', { isDefault: true })
  .description('List durable work plans')
  .option(...urlOption)
  .action(plansListCommand)
plans
  .command('create <goal...>')
  .description('Create a durable work plan')
  .option('--title <title>', 'Plan title')
  .option('--step <title>', 'Add a pending plan step', collectPlanOption, [])
  .option('--acceptance <text>', 'Add acceptance criteria', collectPlanOption, [])
  .option('--risk <text>', 'Add a known risk', collectPlanOption, [])
  .option('--decision <text>', 'Add a recorded decision', collectPlanOption, [])
  .option('--source-session <id>', 'Source session id')
  .option('--ready', 'Create the plan with ready status')
  .option(...urlOption)
  .action(plansCreateCommand)
plans
  .command('show <id>')
  .description('Show a durable work plan')
  .option(...urlOption)
  .action(plansShowCommand)
plans
  .command('start <id>')
  .description('Start a plan as a background subagent job')
  .option('--category <id>', 'Subagent category preset', 'implementation')
  .option('--agent <id>', 'Use a registered user-defined agent')
  .option('--model <id>', 'Provider model override')
  .option('--tools <a,b,c>', 'Comma-separated subset of parent allowed tools')
  .option('--max-iterations <n>', 'Max turns for the execution subagent')
  .option(...urlOption)
  .action(plansStartCommand)

const swarm = program
  .command('swarm')
  .description('Orchestrate external CLI agents (claude/codex/gemini/opencode)')
swarm
  .command('run <goal>')
  .description('Start a new swarm run')
  .option('--agent <names>', 'comma-separated warm pool (claude,codex,gemini,opencode)', 'claude')
  .option('--cwd <path>', 'working directory')
  .option('--worktree <name>', 'git worktree name')
  .option('--autonomy <level>', 'readonly|accept-edits|workspace-write|supervised|autonomous')
  .option('--no-auto-approve-agents', 'do NOT pass auto-approve flags to agents')
  .option('--no-supervisor', 'spawn agents and keep the run for manual attach control')
  .option('--detach', 'return runId and exit (default)')
  .option('--attach', 'attach to the run after starting')
  .option(...urlOption)
  .action(async (goal: string, options) => {
    try {
      await swarmRunCommand(goal, options)
    } catch (err) {
      if (!printApiError(err)) {
        const message = err instanceof Error ? err.message : String(err)
        process.stderr.write(chalk.red(`swarm run failed: ${message}\n`))
      }
      process.exit(1)
    }
  })
swarm
  .command('list')
  .description('List swarm runs')
  .option('--status <s>', 'Filter by status (pending|running|done|error|cancelled|interrupted)')
  .option(...urlOption)
  .action(async (options: { url?: string; status?: SwarmRunStatus }) => {
    try {
      await swarmListCommand(options)
    } catch (err) {
      if (!printApiError(err)) {
        const message = err instanceof Error ? err.message : String(err)
        process.stderr.write(chalk.red(`swarm list failed: ${message}\n`))
      }
      process.exit(1)
    }
  })
swarm
  .command('status <runId>')
  .description('Show snapshot of a swarm run')
  .option(...urlOption)
  .action(async (runId: string, options: { url?: string }) => {
    try {
      await swarmStatusCommand(runId, options)
    } catch (err) {
      if (!printApiError(err)) {
        const message = err instanceof Error ? err.message : String(err)
        process.stderr.write(chalk.red(`swarm status failed: ${message}\n`))
      }
      process.exit(1)
    }
  })
swarm
  .command('kill <runId>')
  .description('Cancel a running swarm run')
  .option(...urlOption)
  .action(async (runId: string, options: { url?: string }) => {
    try {
      await swarmKillCommand(runId, options)
    } catch (err) {
      if (!printApiError(err)) {
        const message = err instanceof Error ? err.message : String(err)
        process.stderr.write(chalk.red(`swarm kill failed: ${message}\n`))
      }
      process.exit(1)
    }
  })
swarm
  .command('agents <runId>')
  .description('List agents inside a swarm run')
  .option(...urlOption)
  .action(async (runId: string, options: { url?: string }) => {
    try {
      await swarmAgentsCommand(runId, options)
    } catch (err) {
      if (!printApiError(err)) {
        const message = err instanceof Error ? err.message : String(err)
        process.stderr.write(chalk.red(`swarm agents failed: ${message}\n`))
      }
      process.exit(1)
    }
  })
swarm
  .command('logs <runId>')
  .description('Print swarm run events (use --follow to stream live)')
  .option('--follow', 'follow new events via SSE')
  .option('--since <ts>', 'only events after this epoch ms')
  .option('--filter <kind>', 'tool|pane|agent')
  .option('--json', 'emit raw jsonl')
  .option(...urlOption)
  .action(
    async (
      runId: string,
      options: { url?: string; follow?: boolean; since?: string; filter?: string; json?: boolean },
    ) => {
      try {
        await swarmLogsCommand(runId, options)
      } catch (err) {
        if (!printApiError(err)) {
          const message = err instanceof Error ? err.message : String(err)
          process.stderr.write(chalk.red(`swarm logs failed: ${message}\n`))
        }
        process.exit(1)
      }
    },
  )
swarm
  .command('attach <runId>')
  .description('Attach to a swarm run (raw tmux mirror + key passthrough)')
  .option(
    '--agent <handleOrRole>',
    'specific agent handle or role to attach to (defaults to active)',
  )
  .option(...urlOption)
  .action(async (runId: string, options: { url?: string; agent?: string }) => {
    try {
      await swarmAttachCommand(runId, options)
    } catch (err) {
      if (!printApiError(err)) {
        const message = err instanceof Error ? err.message : String(err)
        process.stderr.write(chalk.red(`swarm attach failed: ${message}\n`))
      }
      process.exit(1)
    }
  })
swarm
  .command('drive <runId> <handle> <prompt...>')
  .description('Drive one swarm agent through send/wait/observe turns')
  .option('--max-turns <n>', 'maximum turns to drive', '8')
  .option('--timeout-sec <n>', 'per-turn wait timeout in seconds')
  .option('--continue-prompt <text>', 'prompt to send after each non-final idle turn')
  .option('--followup <text>', 'explicit follow-up prompt; repeatable', collectRepeatedOption, [])
  .option('--stop-pattern <regex>', 'regex that marks output as done; repeatable', collectRepeatedOption, [])
  .option('--watch', 'stream swarm drive progress events while waiting')
  .option(...urlOption)
  .action(async (
    runId: string,
    handle: string,
    promptParts: string[],
    options: {
      url?: string
      maxTurns?: string
      timeoutSec?: string
      continuePrompt?: string
      followup?: string[]
      stopPattern?: string[]
      watch?: boolean
    },
  ) => {
    try {
      await swarmDriveCommand(runId, handle, promptParts.join(' '), options)
    } catch (err) {
      if (!printApiError(err)) {
        const message = err instanceof Error ? err.message : String(err)
        process.stderr.write(chalk.red(`swarm drive failed: ${message}\n`))
      }
      process.exit(1)
    }
  })

export async function runCli(argv: string[] = process.argv.slice(2)): Promise<void> {
  // Hidden subcommand: in a single-file bundle this re-enters the same
  // executable as the foreground daemon. Resolved before commander sees argv.
  if (argv[0] === '__daemon') {
    const standalone = getStandaloneDaemon()
    if (!standalone) {
      throw new Error('sepilot __daemon is only available in the standalone single-file build.')
    }
    await standalone.foregroundMain(argv.slice(1))
    return
  }

  // Set JSON output format early before any command runs
  if (argv.includes('--json')) setOutputFormat('json')
  const outputFormatFlag = readOptionValue(argv, '--output-format')
  if (outputFormatFlag === 'json' || outputFormatFlag === 'stream-json') {
    setOutputFormat(outputFormatFlag)
  }

  // resolveDaemonBaseUrl falls through: explicit --url → SEPILOTD_URL
  // env → library default. We coalesce undefined back to the loopback
  // string so downstream options objects stay string-typed.
  const rootUrl = resolveDaemonBaseUrl(readOptionValue(argv, '--url')) ?? 'http://127.0.0.1:17600'

  if (shouldRunRootPrompt(argv)) {
    await askCommand(readOptionValue(argv, '--prompt') ?? readOptionValue(argv, '-p'), {
      url: rootUrl,
      model: readOptionValue(argv, '--model'),
      provider: readOptionValue(argv, '--provider'),
      maxTokens: readOptionValue(argv, '--max-tokens'),
      interactive: argv.includes('--interactive'),
      session: readOptionValue(argv, '--session'),
    })
  } else if (shouldLaunchTui(argv)) {
    const url = rootUrl
    const model = readOptionValue(argv, '--model')
    const provider = readOptionValue(argv, '--provider')
    const sessionId = readOptionValue(argv, '--session')
    const resume = argv.includes('--resume')

    // Ink crashes with 'Raw mode is not supported on the current
    // process.stdin' when stdin isn't a TTY (piped/redirected). Refuse
    // up front with a hint to the non-TUI surface (`ask --session`).
    if (!process.stdin.isTTY) {
      console.error(chalk.red('The TUI requires an interactive terminal (TTY).'))
      console.error(
        chalk.gray(
          sessionId
            ? `For piped/non-interactive use, run: sepilot ask --session ${sessionId} '<question>'`
            : "For piped/non-interactive use, run: sepilot ask '<question>'",
        ),
      )
      process.exit(1)
    }

    const startupAbort = new AbortController()
    const daemonClient = new DaemonClient(url)
    let tuiDaemon: EnsureDaemonResult | null = null
    const abortStartup = (): void => startupAbort.abort()
    const abortStartupInput = (chunk: Buffer | string): void => {
      const text = Buffer.isBuffer(chunk) ? chunk.toString('utf8') : chunk
      if (text.includes('\u0003')) abortStartup()
    }
    process.once('SIGINT', abortStartup)
    process.stdin.on('data', abortStartupInput)
    process.stdin.resume()
    try {
      tuiDaemon = await ensureDaemon(daemonClient, { url, signal: startupAbort.signal })
    } catch (err) {
      if (startupAbort.signal.aborted) {
        process.exit(130)
      }
      console.error(chalk.red(err instanceof Error ? err.message : String(err)))
      process.exit(1)
    } finally {
      process.off('SIGINT', abortStartup)
      process.stdin.off('data', abortStartupInput)
    }

    const { render } = await import('ink')
    const React = await import('react')
    const { selectShell, shellTerminalPolicy } = await import(
      './tui/next/runtime/shell-selection.js'
    )
    const shell = selectShell(process.env)
    const App = shell === 'legacy'
      ? (await import('./tui/App.js')).App
      : (await import('./tui/next/App.js')).NextApp
    const terminalSequences = shell === 'legacy'
      ? await import('./tui/utils/terminal-modes.js').then((module) => ({
          enter: module.TUI_ENTER_SEQUENCE,
          exit: module.TUI_EXIT_SEQUENCE,
        }))
      : await import('./tui/next/runtime/terminal-modes.js').then((module) => ({
          enter: module.INLINE_ENTER_SEQUENCE,
          exit: module.INLINE_EXIT_SEQUENCE,
        }))
    const terminalPolicy = shellTerminalPolicy(shell, {
      isTty: Boolean(process.stdout.isTTY),
      mouseProxyEnabled: process.env.SEPILOT_TUI_MOUSE_PROXY !== '0',
    })
    const imeCursorOutput = shell === 'next'
      ? (await import('./tui/next/runtime/ime-cursor.js')).createImeCursorOutput(process.stdout)
      : null
    const tuiStdout = imeCursorOutput?.stdout ?? process.stdout

    // Both shells enable bracketed paste. Only legacy's sequence also enters
    // alt-screen and mouse reporting; inline mode preserves host scrollback.
    const useTerminalModes = Boolean(process.stdout.isTTY)
    let terminalModesRestored = false
    const restoreTerminalModes = (): void => {
      if (!useTerminalModes || terminalModesRestored) return
      terminalModesRestored = true
      try {
        process.stdout.write(terminalSequences.exit)
      } catch {
        // Best-effort cleanup; the terminal may already be torn down.
      }
    }
    if (useTerminalModes) {
      process.stdout.write(terminalSequences.enter)
      process.once('exit', restoreTerminalModes)
    }

    // Legacy's enter sequence enables mouse reporting, so it needs the full
    // mouse-aware proxy. Inline mode still uses the lighter function-key proxy
    // to coalesce fragmented F11/F12 escape sequences before Ink sees them;
    // mouse reporting remains disabled so normal terminal selection is intact.
    const useMouseProxy = terminalPolicy.useMouseProxy
    const stdinWasRaw = Boolean((process.stdin as { isRaw?: boolean }).isRaw)
    let preInkRawModeApplied = false
    const restorePreInkRawMode = (): void => {
      if (!preInkRawModeApplied || typeof process.stdin.setRawMode !== 'function') return
      preInkRawModeApplied = false
      try {
        process.stdin.setRawMode(stdinWasRaw)
      } catch {
        // Best-effort cleanup; Ink may have already restored or released stdin.
      }
    }

    // Ink enables raw mode from its first input-hook layout effect. Keeping it
    // enabled through the render handoff prevents terminals from echoing an
    // IME preedit at the physical cursor below the inline composer before Ink
    // has mounted its input handler. This matters for the default inline
    // shell too; the legacy-only mouse proxy is unrelated to terminal echo.
    if (process.stdin.isTTY && typeof process.stdin.setRawMode === 'function') {
      try {
        process.stdin.setRawMode(true)
        preInkRawModeApplied = true
      } catch {
        // Ink will still report a clear raw-mode error if the terminal cannot support it.
      }
    }

    const tuiStdin = process.stdin.isTTY
      ? useMouseProxy
        ? createMouseAwareStdin(process.stdin as unknown as TuiInputSource)
        : createFunctionKeyAwareStdin(process.stdin as unknown as TuiInputSource)
      : process.stdin

    let instance: ReturnType<typeof render>
    try {
      instance = render(
        React.createElement(App, {
          config: {
            url: url ?? 'http://127.0.0.1:17600',
            model,
            provider,
            sessionId,
            resume,
          },
          runtime: shell === 'next'
            ? { hardwareCursor: imeCursorOutput?.enabled === true }
            : undefined,
        }),
        {
          stdin: tuiStdin as NodeJS.ReadStream,
          stdout: tuiStdout,
          exitOnCtrlC: false,
        },
      )
    } catch (err) {
      imeCursorOutput?.dispose()
      restorePreInkRawMode()
      if ('dispose' in tuiStdin && typeof tuiStdin.dispose === 'function') {
        tuiStdin.dispose()
      }
      restoreTerminalModes()
      throw err
    }

    const swallowSigint = (): void => {
      // Raw-mode terminals normally deliver Ctrl+C through Ink's useInput,
      // where the TUI can apply its staged interrupt rules. Some terminals
      // still emit SIGINT during the render/raw-mode handoff; treat that as an
      // explicit exit instead of swallowing it and leaving the app alive.
      try {
        instance.unmount()
      } catch {
        // Best-effort fallback; the normal teardown path still runs below.
      }
    }
    process.on('SIGINT', swallowSigint)

    let tuiTornDown = false
    const teardownTui = async (): Promise<void> => {
      if (tuiTornDown) return
      tuiTornDown = true
      process.off('SIGINT', swallowSigint)
      if (useTerminalModes) {
        process.off('exit', restoreTerminalModes)
      }

      imeCursorOutput?.dispose()
      restoreTerminalModes()

      try {
        instance.cleanup()
      } catch {
        // Best-effort cleanup; Ink may already have disposed the instance.
      }
      restorePreInkRawMode()

      if ('dispose' in tuiStdin && typeof tuiStdin.dispose === 'function') {
        tuiStdin.dispose()
      }

      try {
        process.stdin.pause()
      } catch {
        // Some stdin implementations cannot be paused after teardown.
      }

      try {
        process.stdin.removeAllListeners('data')
        process.stdin.removeAllListeners('readable')
        process.stdin.unref()
      } catch {
        // Best-effort stdin cleanup so no lingering TTY handle keeps Node alive.
      }

      await stopOwnedDaemon(tuiDaemon, daemonClient)
    }

    try {
      await instance.waitUntilExit()
    } finally {
      await teardownTui()
    }
    process.exit(0)
  } else {
    try {
      await program.parseAsync(['node', 'sepilot', ...argv])
    } catch (err) {
      reportCliError(err)
      process.exit(1)
    }
  }
}

function reportCliError(err: unknown): void {
  const jsonMode = getOutputFormat() === 'json'
  const cause = (err as { cause?: unknown }).cause
  const causeCode = isObject(cause) && typeof cause.code === 'string' ? cause.code : undefined
  const looksLikeFetchFailure = err instanceof TypeError && /fetch failed/i.test(err.message)
  if (
    causeCode === 'ECONNREFUSED' ||
    causeCode === 'ECONNRESET' ||
    causeCode === 'ENOTFOUND' ||
    looksLikeFetchFailure
  ) {
    if (jsonMode) {
      console.log(JSON.stringify({ ok: false, error: 'daemon-unreachable' }, null, 2))
    } else {
      console.error(chalk.red('Cannot connect to sepilotd.'))
      console.error(chalk.gray('Is the daemon running? Start with: sepilot start'))
    }
    if (process.env.SEPILOTD_DEBUG === '1' && err instanceof Error && err.stack) {
      console.error(chalk.gray(err.stack))
    }
    return
  }
  const friendly = err instanceof Error ? friendlyErrorMessage(err) : String(err)
  if (jsonMode) {
    console.log(JSON.stringify({ ok: false, error: friendly }, null, 2))
  } else if (!printApiError(err)) {
    // Status-aware copy already emitted by printApiError when it matched
    // an http error. Otherwise fall through to the friendlyErrorMessage
    // single-line render — covers connection drops, parse errors, etc.
    console.error(chalk.red(friendly))
  }
  if (process.env.SEPILOTD_DEBUG === '1' && err instanceof Error && err.stack) {
    console.error(chalk.gray(err.stack))
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isMainModule(): boolean {
  const entry = process.argv[1]
  if (!entry) return false
  try {
    return realpathSync(entry) === realpathSync(fileURLToPath(import.meta.url))
  } catch {
    return import.meta.url.endsWith('/index.js')
  }
}

if (isMainModule()) {
  await runCli(process.argv.slice(2))
}
