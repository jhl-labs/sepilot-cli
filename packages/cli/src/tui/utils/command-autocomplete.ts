import type {
  DaemonAgentDescriptor,
  DaemonProviderInfo,
  DaemonProject,
  DaemonSkill,
} from '@sepilotd/api-client'
import { formatProviderModelBadges } from '../../utils/provider-display.js'
import { PROVIDER_WIZARD_PRESETS } from '../../utils/provider-presets.js'
import { DAEMON_ADMIN_SLASH_COMMANDS } from './daemon-admin-slash.js'
import { themeOptions, type ThemeId } from '../theme.js'

export interface SlashCommand {
  name: string
  description: string
  args?: string
}

export interface CommandPaletteItem {
  label: string
  description: string
  insertValue: string
}

interface BuildCommandPaletteItemsOptions {
  input: string
  agentModes?: DaemonAgentDescriptor[]
  providers?: DaemonProviderInfo[]
  projects?: DaemonProject[]
  skills?: DaemonSkill[]
  currentMode?: string
  currentProvider?: string
  currentModel?: string
  defaultProvider?: string
  defaultModel?: string
  currentProjectName?: string | null
  currentSessionId?: string | null
  currentThemeId?: ThemeId
  contextPercent?: number | null
  hasPendingApproval?: boolean
  isStreaming?: boolean
  lastSwarmRunId?: string | null
}

export interface RecommendedCommandContext {
  currentProjectName?: string | null
  currentSessionId?: string | null
  providerCount?: number
  contextPercent?: number | null
  hasPendingApproval?: boolean
  isStreaming?: boolean
}

function formatThemeCommandArgs(): string {
  const themeIds = themeOptions.map((theme) => theme.id).join('|')
  return `<${themeIds}|toggle|prev|current>`
}

export function buildCommands(agentModes: DaemonAgentDescriptor[] = []): SlashCommand[] {
  const modeIds = agentModes.map((agent) => agent.id)
  const modeDescription =
    modeIds.length > 0
      ? `Open mode picker or set agent mode (${modeIds.slice(0, 4).join(', ')}${modeIds.length > 4 ? ', ...' : ''})`
      : 'Open mode picker or set agent mode'

  return [
    {
      name: '/model',
      description:
        'Open the model/provider picker, switch the session model, or save/apply daemon defaults',
      args: '<provider/model|name|current|default>',
    },
    {
      name: '/provider',
      description: 'Open the model/provider picker, switch provider, or configure one',
      args: '<name|current|setup|edit|default|delete>',
    },
    { name: '/mode', description: modeDescription, args: '<mode|current>' },
    {
      name: '/recap',
      description: 'Toggle the end-of-turn recap summary (default on)',
      args: '<on|off|status>',
    },
    {
      name: '/capability',
      description:
        'Show or toggle per-model tool transport/scaffolding overrides (promptReactPreferred, adaptivePromptReact, deepCoderAnalysis)',
      args: '[<capability> <on|off> [provider/model]]',
    },
    {
      name: '/autonomy',
      description: 'Open autonomy picker or set agent autonomy',
      args: '<level|current>',
    },
    {
      name: '/session',
      description:
        'Open the session timeline with continue, fork, compact, export, and delete actions',
      args: '[current|branch|compact|export [markdown|json] [path]|delete confirm]',
    },
    {
      name: '/project',
      description: 'Select or inspect the active project',
      args: '<name|auto|none>',
    },
    { name: '/files', description: 'Open file picker for message attachments' },
    { name: '/artifacts', description: 'List saved artifacts for the current session' },
    {
      name: '/approvals',
      description: 'Show pending + remembered approvals, or clear saved rules',
      args: '[clear|clear session|clear always]',
    },
    {
      name: '/resume',
      description:
        'Resume a saved checkpoint, or open the newest session when no context is active',
      args: '[--force]',
    },
    {
      name: '/continue',
      description: 'Alias for /resume with the same checkpoint safety checks',
      args: '[--force]',
    },
    {
      name: '/rewind',
      description:
        'Fork the current session before an earlier user turn without touching the source',
      args: '[turns]',
    },
    { name: '/compact', description: 'Summarize conversation to free context space' },
    {
      name: '/context',
      description: 'Show the context map: token pressure, resume readiness, and rewind targets',
      args: '[map|resume|rewind]',
    },
    { name: '/new', description: 'Start new session' },
    { name: '/thinking', description: 'Set thinking level', args: '<off|low|medium|high|max>' },
    {
      name: '/max-tokens',
      description: 'Override the output token cap for this session',
      args: '<number|off|current>',
    },
    {
      name: '/run',
      description: 'Run a named skill with explicit skill context',
      args: '<skill> [prompt]',
    },
    {
      name: '/clear',
      description:
        'Clear the active conversation context. Past messages stay in the terminal scrollback.',
    },
    { name: '/providers', description: 'List available providers' },
    {
      name: '/usage',
      description: 'Open or refresh the usage dashboard',
      args: '[days|current|close]',
    },
    {
      name: '/tps',
      description: 'Show token-per-second speed for the last response and current session',
      args: '[current|reset]',
    },
    {
      name: '/memory',
      description: 'Search semantic memory and manage markdown file memory',
      args: '<query|current|close|file|backlog|lifecycle|audit|maintenance>',
    },
    {
      name: '/rag',
      description: 'Open, search, add, or sync local RAG sources',
      args: '[open|current|search <query>|add <path>|sync|close|help]',
    },
    { name: '/remember', description: 'Save a manual memory entry', args: '<content>' },
    {
      name: '/skills',
      description:
        'Open installed skill manager, browse configured skill sources, or install a skill',
      args: '<installed|manage|enable|disable|store|search|install>',
    },
    {
      name: '/hooks',
      description: 'Manage outbound delivery hooks',
      args: '<list|add|remove|enable|disable|deliveries|dead-letters|replay|replay-failed|ack>',
    },
    {
      name: '/theme',
      description: 'Switch or inspect the active theme',
      args: formatThemeCommandArgs(),
    },
    {
      name: '/mcp',
      description: 'Toggle or manage MCP servers',
      args: '[list|add|search|install|marketplace|enable|disable|remove|tools|prompts|resources|logging|metrics|call]',
    },
    {
      name: '/acp',
      description: 'Inspect ACP setup for editor and external-agent integration',
      args: '[help|status|config|opencode|codex|zed]',
    },
    {
      name: '/a2a',
      description: 'Inspect A2A setup for agent-to-agent integration',
      args: '[help|status|card|send]',
    },
    {
      name: '/doctor',
      description: 'Run an operational health summary for the daemon, providers, and usage',
    },
    {
      name: '/swarm',
      description: 'Orchestrate external CLI agents (claude/codex/gemini/opencode) in tmux',
      args: '<run|run!|list|history|status|agents|logs|kill|attach|help>',
    },
    {
      name: '/containers',
      description: 'List sepilot-managed Docker containers from this and previous sessions',
    },
    ...DAEMON_ADMIN_SLASH_COMMANDS,
    { name: '/exit', description: 'Quit the interactive chat' },
    { name: '/help', description: 'Show help' },
  ]
}

export function buildCommandPaletteItems({
  input,
  agentModes = [],
  providers = [],
  projects = [],
  skills = [],
  currentMode,
  currentProvider,
  currentModel,
  defaultProvider,
  defaultModel,
  currentProjectName,
  currentSessionId,
  currentThemeId,
  contextPercent,
  hasPendingApproval,
  isStreaming,
  lastSwarmRunId,
}: BuildCommandPaletteItemsOptions): CommandPaletteItem[] {
  if (!input.startsWith('/')) {
    return []
  }

  const commands = buildCommands(agentModes)
  const body = input.slice(1)
  const hasTrailingSpace = /\s$/.test(input)
  const tokens = body.split(/\s+/).filter(Boolean)

  if (tokens.length === 0) {
    return dedupeItems([
      ...buildRecommendedCommandItems({
        currentProjectName,
        currentSessionId,
        providerCount: providers.length,
        contextPercent,
        hasPendingApproval,
        isStreaming,
      }),
      ...commands.map(commandToItem),
    ])
  }

  const commandToken = tokens[0].toLowerCase()
  const matchingCommands = findMatchingCommands(commands, commandToken)
  const exactCommand = commands.find(
    (command) => command.name.slice(1).toLowerCase() === commandToken,
  )

  if (tokens.length === 1 && !hasTrailingSpace && exactCommand) {
    const argItems = buildArgumentItems(
      exactCommand.name,
      agentModes,
      providers,
      projects,
      skills,
      currentMode,
      currentProvider,
      currentModel,
      defaultProvider,
      defaultModel,
      currentProjectName,
      currentSessionId,
      currentThemeId,
      lastSwarmRunId,
    )

    if (argItems.length > 0) {
      return dedupeItems(
        [
          buildExactCommandActionItem(exactCommand.name, {
            currentMode,
            currentProvider,
            currentModel,
            currentProjectName,
            currentSessionId,
            currentThemeId,
            defaultProvider,
            defaultModel,
            contextPercent,
            lastSwarmRunId,
          }),
          ...argItems,
        ].filter((item): item is CommandPaletteItem => item !== null),
      )
    }
  }

  if (tokens.length === 1 && !hasTrailingSpace) {
    return matchingCommands.map(commandToItem)
  }

  const resolvedCommand =
    exactCommand ?? (matchingCommands.length === 1 ? matchingCommands[0] : null)

  if (!resolvedCommand) {
    return matchingCommands.map(commandToItem)
  }

  const rawArgQuery = body.slice(tokens[0].length).trimStart().toLowerCase()
  const argQuery = hasTrailingSpace && tokens.length === 1 ? '' : rawArgQuery.trimEnd()
  const argItems = buildArgumentItems(
    resolvedCommand.name,
    agentModes,
    providers,
    projects,
    skills,
    currentMode,
    currentProvider,
    currentModel,
    defaultProvider,
    defaultModel,
    currentProjectName,
    currentSessionId,
    currentThemeId,
    lastSwarmRunId,
  )

  if (argItems.length === 0) {
    return [commandToItem(resolvedCommand)]
  }

  return dedupeItems(
    argItems
      .filter((item) => matchesArgumentItem(item, resolvedCommand.name, argQuery))
      .sort((left, right) => compareArgumentItems(left, right, resolvedCommand.name, argQuery)),
  )
}

function commandToItem(command: SlashCommand): CommandPaletteItem {
  return {
    label: command.args ? `${command.name} ${command.args}` : command.name,
    description: command.description,
    insertValue: command.args ? `${command.name} ` : command.name,
  }
}

export function buildRecommendedCommandItems(
  context: RecommendedCommandContext = {},
): CommandPaletteItem[] {
  const items: CommandPaletteItem[] = []
  const highContextPressure =
    context.contextPercent != null &&
    Number.isFinite(context.contextPercent) &&
    context.contextPercent >= 80

  if ((context.providerCount ?? 0) <= 0) {
    items.push({
      label: '/provider setup',
      description: 'Configure a provider first so the CLI can run agent sessions',
      insertValue: '/provider setup',
    })
  }

  if (context.hasPendingApproval) {
    items.push(
      {
        label: '/approvals',
        description: 'Inspect the pending approval and any remembered approval rules',
        insertValue: '/approvals',
      },
      {
        label: '/context resume',
        description: 'Check resume and replay safety before you continue the blocked run',
        insertValue: '/context resume',
      },
      {
        label: '/session',
        description: 'Open the session timeline and maintenance actions for the blocked run',
        insertValue: '/session',
      },
    )
  } else if (context.isStreaming) {
    items.push(
      {
        label: '/context map',
        description:
          'Inspect the current run state, token pressure, and rewind targets without interrupting it',
        insertValue: '/context map',
      },
      {
        label: '/session',
        description: 'Open the session timeline and related session actions',
        insertValue: '/session',
      },
    )
  } else if (!context.currentSessionId) {
    items.push(
      {
        label: '/resume',
        description: 'Open the newest resumable or recent session',
        insertValue: '/resume',
      },
      {
        label: '/new',
        description: 'Start a fresh session in the current workspace',
        insertValue: '/new',
      },
    )
  } else {
    items.push({
      label: '/context map',
      description:
        'Inspect token pressure, rewind targets, and resume readiness for the active session',
      insertValue: '/context map',
    })
    if (highContextPressure) {
      items.push({
        label: '/compact',
        description:
          'Context usage is high; compact older conversation while preserving recent context',
        insertValue: '/compact',
      })
    }
    items.push(
      {
        label: '/rewind 1',
        description: 'Fork before the latest user turn and retry safely from there',
        insertValue: '/rewind 1',
      },
      {
        label: '/session',
        description: context.currentProjectName
          ? `Open the session timeline for ${context.currentProjectName}`
          : 'Open the session timeline and maintenance actions',
        insertValue: '/session',
      },
    )
  }

  items.push(
    {
      label: '/doctor',
      description:
        'Run a quick operational health summary for daemon connectivity, providers, and usage',
      insertValue: '/doctor',
    },
    {
      label: '/help',
      description: 'Open the full keyboard shortcut and slash command reference',
      insertValue: '/help',
    },
    {
      label: '/exit',
      description: 'Quit the interactive chat',
      insertValue: '/exit',
    },
  )

  return dedupeItems(items)
}

function buildArgumentItems(
  commandName: string,
  agentModes: DaemonAgentDescriptor[],
  providers: DaemonProviderInfo[],
  projects: DaemonProject[],
  skills: DaemonSkill[],
  currentMode?: string,
  currentProvider?: string,
  currentModel?: string,
  defaultProvider?: string,
  defaultModel?: string,
  currentProjectName?: string | null,
  currentSessionId?: string | null,
  currentThemeId?: ThemeId,
  lastSwarmRunId?: string | null,
): CommandPaletteItem[] {
  switch (commandName) {
    case '/provider':
      return [
        {
          label: '/provider current',
          description: `Inspect current provider${currentProvider ? ` (${currentProvider})` : ''}`,
          insertValue: '/provider current',
        },
        {
          label: '/provider default',
          description: 'Open the picker in daemon-default mode',
          insertValue: '/provider default',
        },
        ...providers.map((provider) => ({
          label: `/provider default ${provider.id}`,
          description: `Set ${provider.name} as the daemon default provider`,
          insertValue: `/provider default ${provider.id}`,
        })),
        ...providers.map((provider) => ({
          label: `/provider delete ${provider.id}`,
          description: `Delete ${provider.name} from daemon config`,
          insertValue: `/provider delete ${provider.id}`,
        })),
        {
          label: '/provider setup',
          description: 'Open the provider setup wizard',
          insertValue: '/provider setup',
        },
        ...PROVIDER_WIZARD_PRESETS.map((preset) => ({
          label: `/provider setup ${preset.type}`,
          description: `Configure ${preset.label}`,
          insertValue: `/provider setup ${preset.type}`,
        })),
        ...providers.map((provider) => ({
          label: `/provider edit ${provider.id}`,
          description: `Edit config for ${provider.name}`,
          insertValue: `/provider edit ${provider.id}`,
        })),
        ...providers.map((provider) => ({
          label: `/provider ${provider.id}`,
          description: `${provider.name} (${provider.models.length} models)`,
          insertValue: `/provider ${provider.id}`,
        })),
      ]
    case '/model': {
      const orderedProviders = currentProvider
        ? [
            ...providers.filter((provider) => provider.id === currentProvider),
            ...providers.filter((provider) => provider.id !== currentProvider),
          ]
        : providers

      return [
        {
          label: '/model current',
          description:
            currentProvider && currentModel
              ? `Inspect current model (${currentProvider}/${currentModel})`
              : 'Inspect current model selection',
          insertValue: '/model current',
        },
        {
          label: '/model default',
          description:
            defaultProvider && defaultModel
              ? `Open daemon-default picker (${defaultProvider}/${defaultModel})`
              : 'Open the picker in daemon-default mode',
          insertValue: '/model default',
        },
        {
          label: '/model default apply',
          description:
            defaultProvider && defaultModel
              ? `Switch the current session to ${defaultProvider}/${defaultModel}`
              : 'Switch the current session to the saved daemon default',
          insertValue: '/model default apply',
        },
        ...orderedProviders
          .flatMap((provider) =>
            provider.models.map((model) => ({
              label: `/model ${provider.id}/${model.id}`,
              description: `${provider.name}${formatProviderModelBadges(model)}`,
              insertValue: `/model ${provider.id}/${model.id}`,
            })),
          )
          .slice(0, 16),
        ...orderedProviders
          .flatMap((provider) =>
            provider.models.map((model) => ({
              label: `/model default ${provider.id}/${model.id}`,
              description: `Save ${provider.name}${formatProviderModelBadges(model)} as daemon default`,
              insertValue: `/model default ${provider.id}/${model.id}`,
            })),
          )
          .slice(0, 16),
      ]
    }
    case '/mode':
      return [
        {
          label: '/mode current',
          description: `Inspect current mode${currentMode ? ` (${currentMode})` : ''}`,
          insertValue: '/mode current',
        },
        ...agentModes.map((agent) => ({
          label: `/mode ${agent.id}`,
          description: agent.description || `Switch to ${agent.name || agent.id}`,
          insertValue: `/mode ${agent.id}`,
        })),
      ]
    case '/autonomy':
      return [
        {
          label: '/autonomy current',
          description: 'Inspect the current autonomy level',
          insertValue: '/autonomy current',
        },
        {
          label: '/autonomy readonly',
          description: 'Require approval before all non-read actions',
          insertValue: '/autonomy readonly',
        },
        {
          label: '/autonomy accept-edits',
          description: 'Auto-accept simple file writes',
          insertValue: '/autonomy accept-edits',
        },
        {
          label: '/autonomy workspace-write',
          description: 'Auto-accept edits inside the active workspace',
          insertValue: '/autonomy workspace-write',
        },
        {
          label: '/autonomy supervised',
          description: 'Allow guided execution with approvals',
          insertValue: '/autonomy supervised',
        },
        {
          label: '/autonomy autonomous',
          description: 'Allow autonomous execution',
          insertValue: '/autonomy autonomous',
        },
      ]
    case '/approvals':
      return [
        {
          label: '/approvals clear',
          description: 'Clear all remembered approval rules',
          insertValue: '/approvals clear',
        },
        {
          label: '/approvals clear session',
          description: 'Clear remembered approvals saved for this session only',
          insertValue: '/approvals clear session',
        },
        {
          label: '/approvals clear always',
          description: 'Clear persistent remembered approvals',
          insertValue: '/approvals clear always',
        },
      ]
    case '/thinking':
      return ['off', 'low', 'medium', 'high', 'max'].map((level) => ({
        label: `/thinking ${level}`,
        description: `Set thinking level to ${level}`,
        insertValue: `/thinking ${level}`,
      }))
    case '/max-tokens':
      return [
        {
          label: '/max-tokens current',
          description: 'Inspect the current output token cap',
          insertValue: '/max-tokens current',
        },
        {
          label: '/max-tokens off',
          description: 'Remove the session override (fall back to provider default)',
          insertValue: '/max-tokens off',
        },
        ...[1024, 2048, 4096, 8192, 16_384, 32_768].map((value) => ({
          label: `/max-tokens ${value}`,
          description: `Cap output at ${value.toLocaleString()} tokens`,
          insertValue: `/max-tokens ${value}`,
        })),
      ]
    case '/run': {
      const runnableSkills = skills
        .filter((skill) => skill.enabled !== false)
        .slice()
        .sort((left, right) => `${left.name} ${left.id}`.localeCompare(`${right.name} ${right.id}`))

      if (runnableSkills.length === 0) {
        return [
          {
            label: '/skills installed',
            description: 'No enabled skills are loaded; inspect installed skills',
            insertValue: '/skills installed',
          },
          {
            label: '/skills store ',
            description: 'Browse configured skill sources',
            insertValue: '/skills store ',
          },
        ]
      }

      return runnableSkills.slice(0, 16).map((skill) => ({
        label: `/run ${skill.id}`,
        description: `${skill.name} v${skill.version} - ${skill.description}`,
        insertValue: `/run ${skill.id} `,
      }))
    }
    case '/session':
      return [
        {
          label: '/session current',
          description: `Inspect current session${currentSessionId ? ` (${currentSessionId.slice(0, 8)})` : ''}`,
          insertValue: '/session current',
        },
        {
          label: '/session branch',
          description: currentSessionId
            ? `Branch current session ${currentSessionId.slice(0, 8)} at the tail`
            : 'Branch the current session at the tail',
          insertValue: '/session branch',
        },
        {
          label: '/session compact',
          description: currentSessionId
            ? `Compact current session ${currentSessionId.slice(0, 8)}`
            : 'Compact the current session',
          insertValue: '/session compact',
        },
        {
          label: '/session export',
          description: 'Export the current session to the workspace as markdown',
          insertValue: '/session export',
        },
        {
          label: '/session export markdown',
          description: 'Export the current session as markdown',
          insertValue: '/session export markdown',
        },
        {
          label: '/session export json',
          description: 'Export the current session as JSON',
          insertValue: '/session export json',
        },
        {
          label: '/session delete confirm',
          description: currentSessionId
            ? `Delete current session ${currentSessionId.slice(0, 8)} after confirmation`
            : 'Delete the current session after confirmation',
          insertValue: '/session delete confirm',
        },
      ]
    case '/project':
      return [
        {
          label: '/project current',
          description: `Inspect active project${currentProjectName ? ` (${currentProjectName})` : ''}`,
          insertValue: '/project current',
        },
        {
          label: '/project auto',
          description: 'Auto-detect the workspace project',
          insertValue: '/project auto',
        },
        {
          label: '/project none',
          description: 'Clear the current project selection',
          insertValue: '/project none',
        },
        ...projects.slice(0, 8).map((project) => ({
          label: `/project ${project.name}`,
          description: `Select project ${project.name}`,
          insertValue: `/project ${project.name}`,
        })),
      ]
    case '/resume':
    case '/continue':
      return [
        {
          label: `${commandName} --force`,
          description: 'Resume even when replay safety requires explicit confirmation',
          insertValue: `${commandName} --force`,
        },
      ]
    case '/rewind':
      return [1, 2, 3, 5].map((turns) => ({
        label: `/rewind ${turns}`,
        description:
          turns === 1
            ? 'Fork before the last user turn and retry from there'
            : `Fork before the last ${turns} user turns and retry from there`,
        insertValue: `/rewind ${turns}`,
      }))
    case '/context':
      return [
        {
          label: '/context map',
          description: 'Show token pressure, session history, resume state, and next actions',
          insertValue: '/context map',
        },
        {
          label: '/context resume',
          description: 'Focus the context map on checkpoint and replay safety',
          insertValue: '/context resume',
        },
        {
          label: '/context rewind',
          description: 'List recent user-turn fork points with event counts',
          insertValue: '/context rewind',
        },
      ]
    case '/usage':
      return [
        {
          label: '/usage current',
          description: 'Inspect the currently loaded usage panel state',
          insertValue: '/usage current',
        },
        {
          label: '/usage close',
          description: 'Close the usage dashboard',
          insertValue: '/usage close',
        },
        ...[1, 7, 30, 90].map((days) => ({
          label: `/usage ${days}`,
          description: `Open usage dashboard for the last ${days} day${days === 1 ? '' : 's'}`,
          insertValue: `/usage ${days}`,
        })),
      ]
    case '/tps':
      return [
        {
          label: '/tps current',
          description: 'Show current token-per-second stats',
          insertValue: '/tps current',
        },
        {
          label: '/tps reset',
          description: 'Reset observed token-per-second stats',
          insertValue: '/tps reset',
        },
      ]
    case '/memory':
      return [
        {
          label: '/memory current',
          description: 'Inspect the current memory search panel state',
          insertValue: '/memory current',
        },
        {
          label: '/memory close',
          description: 'Close the memory search panel',
          insertValue: '/memory close',
        },
        {
          label: '/memory file',
          description: 'List markdown MEMORY.md sections and daily note status',
          insertValue: '/memory file',
        },
        {
          label: '/memory backlog',
          description: 'List open-loop backlog and today reflection entries',
          insertValue: '/memory backlog',
        },
        {
          label: '/memory backlog add ',
          description: 'Add an open-loop backlog item',
          insertValue: '/memory backlog add ',
        },
        {
          label: '/memory backlog done ',
          description: 'Resolve open-loop backlog items matching text',
          insertValue: '/memory backlog done ',
        },
        {
          label: '/memory lifecycle',
          description: 'Show semantic memory lifecycle and cleanup candidate counts',
          insertValue: '/memory lifecycle',
        },
        {
          label: '/memory audit ',
          description: 'List recent semantic memory audit entries',
          insertValue: '/memory audit ',
        },
        {
          label: '/memory maintenance',
          description: 'Preview memory cleanup candidates without pruning',
          insertValue: '/memory maintenance',
        },
        {
          label: '/memory maintenance --apply',
          description: 'Run memory cleanup and record an audit entry',
          insertValue: '/memory maintenance --apply',
        },
        {
          label: '/memory file show ',
          description: 'Show a markdown memory section',
          insertValue: '/memory file show ',
        },
        {
          label: '/memory file today',
          description: 'Show today daily memory note',
          insertValue: '/memory file today',
        },
        {
          label: '/memory file set ',
          description: 'Save a markdown memory section: /memory file set <section> -- <content>',
          insertValue: '/memory file set ',
        },
      ]
    case '/rag':
      return [
        {
          label: '/rag current',
          description: 'Inspect configured local RAG sources and vector index status',
          insertValue: '/rag current',
        },
        {
          label: '/rag open',
          description: 'Open the local RAG panel',
          insertValue: '/rag open',
        },
        {
          label: '/rag search ',
          description: 'Search local RAG sources in the inline panel',
          insertValue: '/rag search ',
        },
        {
          label: '/rag add ',
          description: 'Add a local git/worktree path as a RAG source',
          insertValue: '/rag add ',
        },
        {
          label: '/rag sync',
          description: 'Sync local RAG sources into the vector index',
          insertValue: '/rag sync',
        },
        {
          label: '/rag close',
          description: 'Close the local RAG panel',
          insertValue: '/rag close',
        },
        {
          label: '/rag help',
          description: 'Show local RAG command usage',
          insertValue: '/rag help',
        },
      ]
    case '/skills':
      return [
        {
          label: '/skills help',
          description: 'Show skill command usage',
          insertValue: '/skills help',
        },
        {
          label: '/skills installed',
          description: 'Open the installed skill manager with enable/disable checkboxes',
          insertValue: '/skills installed',
        },
        {
          label: '/skills manage',
          description: 'Alias for the installed skill manager',
          insertValue: '/skills manage',
        },
        {
          label: '/skills enable ',
          description: 'Enable an installed opt-in or disabled skill by id',
          insertValue: '/skills enable ',
        },
        {
          label: '/skills disable ',
          description: 'Disable an installed skill by id',
          insertValue: '/skills disable ',
        },
        {
          label: '/skills store ',
          description: 'Open the interactive skill catalog',
          insertValue: '/skills store ',
        },
        {
          label: '/skills search ',
          description: 'Search configured skill sources by keyword in the picker',
          insertValue: '/skills search ',
        },
        {
          label: '/skills install ',
          description: 'Install a skill from a store result, URL, or local path',
          insertValue: '/skills install ',
        },
        ...skills
          .filter((skill) => skill.enabled === false)
          .slice(0, 16)
          .map((skill) => ({
            label: `/skills enable ${skill.id}`,
            description: `Enable ${skill.name} v${skill.version}`,
            insertValue: `/skills enable ${skill.id}`,
          })),
        ...skills
          .filter((skill) => skill.enabled !== false)
          .slice(0, 16)
          .map((skill) => ({
            label: `/skills disable ${skill.id}`,
            description: `Disable ${skill.name} v${skill.version}`,
            insertValue: `/skills disable ${skill.id}`,
          })),
      ]
    case '/hooks':
      return [
        {
          label: '/hooks list',
          description: 'List configured outbound hooks',
          insertValue: '/hooks list',
        },
        {
          label: '/hooks add ',
          description: 'Add an outbound hook',
          insertValue: '/hooks add ',
        },
        {
          label: '/hooks remove ',
          description: 'Remove an outbound hook',
          insertValue: '/hooks remove ',
        },
        {
          label: '/hooks enable ',
          description: 'Enable an outbound hook',
          insertValue: '/hooks enable ',
        },
        {
          label: '/hooks disable ',
          description: 'Disable an outbound hook',
          insertValue: '/hooks disable ',
        },
        {
          label: '/hooks deliveries ',
          description: 'List outbound hook deliveries',
          insertValue: '/hooks deliveries ',
        },
        {
          label: '/hooks dead-letters ',
          description: 'List outbound hook dead letters',
          insertValue: '/hooks dead-letters ',
        },
        {
          label: '/hooks replay ',
          description: 'Replay an outbound hook delivery',
          insertValue: '/hooks replay ',
        },
        {
          label: '/hooks replay-failed ',
          description: 'Replay open outbound hook dead letters',
          insertValue: '/hooks replay-failed ',
        },
        {
          label: '/hooks ack ',
          description: 'Acknowledge an outbound hook dead letter',
          insertValue: '/hooks ack ',
        },
      ]
    case '/mcp':
      return [
        {
          label: '/mcp list',
          description: 'List configured MCP servers and connection status',
          insertValue: '/mcp list',
        },
        {
          label: '/mcp search ',
          description: 'Search MCP marketplaces by keyword',
          insertValue: '/mcp search ',
        },
        {
          label: '/mcp add ',
          description: 'Add or replace an MCP server config',
          insertValue: '/mcp add ',
        },
        {
          label: '/mcp install ',
          description: 'Install an MCP server from a marketplace',
          insertValue: '/mcp install ',
        },
        {
          label: '/mcp tools ',
          description: 'List enabled and disabled tools for a server',
          insertValue: '/mcp tools ',
        },
        {
          label: '/mcp tools disable ',
          description: 'Disable one MCP tool for a server',
          insertValue: '/mcp tools disable ',
        },
        {
          label: '/mcp tools enable ',
          description: 'Enable one disabled MCP tool for a server',
          insertValue: '/mcp tools enable ',
        },
        {
          label: '/mcp prompts list',
          description: 'List MCP prompts exposed by configured servers',
          insertValue: '/mcp prompts list',
        },
        {
          label: '/mcp resources list',
          description: 'List MCP resources exposed by configured servers',
          insertValue: '/mcp resources list',
        },
        {
          label: '/mcp resources templates',
          description: 'List MCP resource URI templates',
          insertValue: '/mcp resources templates',
        },
        {
          label: '/mcp logging logs ',
          description: 'Show received MCP log messages for a server',
          insertValue: '/mcp logging logs ',
        },
        {
          label: '/mcp metrics',
          description: 'Show MCP tool-call metrics',
          insertValue: '/mcp metrics',
        },
        {
          label: '/mcp call ',
          description: 'Call a tool on an MCP server with JSON input',
          insertValue: '/mcp call ',
        },
        {
          label: '/mcp enable ',
          description: 'Enable a configured MCP server',
          insertValue: '/mcp enable ',
        },
        {
          label: '/mcp disable ',
          description: 'Disable a configured MCP server',
          insertValue: '/mcp disable ',
        },
        {
          label: '/mcp remove ',
          description: 'Remove a configured MCP server',
          insertValue: '/mcp remove ',
        },
        {
          label: '/mcp marketplace list',
          description: 'List MCP server marketplaces',
          insertValue: '/mcp marketplace list',
        },
        {
          label: '/mcp marketplace add ',
          description: 'Add an MCP server marketplace',
          insertValue: '/mcp marketplace add ',
        },
        {
          label: '/mcp marketplace remove ',
          description: 'Remove an MCP server marketplace',
          insertValue: '/mcp marketplace remove ',
        },
      ]
    case '/acp':
      return [
        {
          label: '/acp help',
          description: 'Show ACP command usage and supported protocol methods',
          insertValue: '/acp help',
        },
        {
          label: '/acp status',
          description: 'Check daemon connectivity and print the active ACP endpoint command',
          insertValue: '/acp status',
        },
        {
          label: '/acp config',
          description: 'Print editor registration snippets for the sepilot ACP stdio server',
          insertValue: '/acp config',
        },
        {
          label: '/acp opencode',
          description: 'Show how opencode ACP fits the external-agent adapter path',
          insertValue: '/acp opencode',
        },
        {
          label: '/acp codex',
          description: 'Show how Codex ACP adapter support is wired',
          insertValue: '/acp codex',
        },
        {
          label: '/acp zed',
          description: 'Print a Zed-style ACP registration hint',
          insertValue: '/acp zed',
        },
      ]
    case '/a2a':
      return [
        {
          label: '/a2a help',
          description: 'Show A2A command usage and supported methods',
          insertValue: '/a2a help',
        },
        {
          label: '/a2a status',
          description: 'Check daemon connectivity and A2A endpoint details',
          insertValue: '/a2a status',
        },
        {
          label: '/a2a card',
          description: 'Print the public agent card endpoint and client hints',
          insertValue: '/a2a card',
        },
        {
          label: '/a2a send',
          description: 'Show how to send an A2A message from an agent turn',
          insertValue: '/a2a send',
        },
      ]
    case '/theme':
      return [
        {
          label: '/theme current',
          description: `Inspect the active theme${currentThemeId ? ` (${currentThemeId})` : ''}`,
          insertValue: '/theme current',
        },
        {
          label: '/theme toggle',
          description: 'Toggle between dark and light theme presets',
          insertValue: '/theme toggle',
        },
        ...themeOptions.map((theme) => ({
          label: `/theme ${theme.id}`,
          description: theme.description,
          insertValue: `/theme ${theme.id}`,
        })),
      ]
    case '/swarm': {
      const defaultRun = lastSwarmRunId
        ? `last run ${lastSwarmRunId.slice(0, 8)}`
        : 'last run when available'

      return [
        {
          label: '/swarm run <goal>',
          description: 'Start a supervised swarm run; type the goal after this prefix',
          insertValue: '/swarm run ',
        },
        {
          label: '/swarm run! <goal>',
          description: 'Start a no-supervisor swarm run; type the goal after this prefix',
          insertValue: '/swarm run! ',
        },
        {
          label: '/swarm list',
          description: 'List active swarm runs',
          insertValue: '/swarm list',
        },
        {
          label: '/swarm history',
          description: 'Show recent active and completed swarm runs',
          insertValue: '/swarm history',
        },
        {
          label: '/swarm status',
          description: `Show swarm status, defaulting to ${defaultRun}`,
          insertValue: '/swarm status',
        },
        {
          label: '/swarm agents',
          description: `List agents and handles, defaulting to ${defaultRun}`,
          insertValue: '/swarm agents',
        },
        {
          label: '/swarm logs',
          description: `Show recent swarm events, defaulting to ${defaultRun}`,
          insertValue: '/swarm logs',
        },
        {
          label: '/swarm kill',
          description: `Cancel a swarm run, defaulting to ${defaultRun}`,
          insertValue: '/swarm kill',
        },
        {
          label: '/swarm attach',
          description: `Open an interactive tmux mirror, defaulting to ${defaultRun}`,
          insertValue: '/swarm attach',
        },
        {
          label: '/swarm help',
          description: 'Show detailed swarm command help',
          insertValue: '/swarm help',
        },
      ]
    }
    default:
      return []
  }
}

interface ExactCommandContext {
  currentMode?: string
  currentProvider?: string
  currentModel?: string
  defaultProvider?: string
  defaultModel?: string
  currentProjectName?: string | null
  currentSessionId?: string | null
  currentThemeId?: ThemeId
  contextPercent?: number | null
  lastSwarmRunId?: string | null
}

function buildExactCommandActionItem(
  commandName: string,
  context: ExactCommandContext,
): CommandPaletteItem | null {
  switch (commandName) {
    case '/model':
      return {
        label: '/model',
        description:
          context.currentProvider && context.currentModel
            ? `Open the model/provider picker from ${context.currentProvider}/${context.currentModel}`
            : 'Open the model/provider picker',
        insertValue: '/model',
      }
    case '/provider':
      return {
        label: '/provider',
        description: context.currentProvider
          ? `Open the model/provider picker from ${context.currentProvider}`
          : 'Open the model/provider picker',
        insertValue: '/provider',
      }
    case '/mode':
      return {
        label: '/mode',
        description: context.currentMode
          ? `Open the mode picker from ${context.currentMode}`
          : 'Open the mode picker',
        insertValue: '/mode',
      }
    case '/autonomy':
      return {
        label: '/autonomy',
        description: 'Open the autonomy picker',
        insertValue: '/autonomy',
      }
    case '/approvals':
      return {
        label: '/approvals',
        description: 'Inspect pending and remembered approvals',
        insertValue: '/approvals',
      }
    case '/session':
      return {
        label: '/session',
        description: context.currentSessionId
          ? `Open the session timeline from ${context.currentSessionId.slice(0, 8)}`
          : 'Open the session timeline',
        insertValue: '/session',
      }
    case '/project':
      return {
        label: '/project',
        description: context.currentProjectName
          ? `Inspect active project ${context.currentProjectName}`
          : 'Inspect active project selection',
        insertValue: '/project',
      }
    case '/resume':
      return {
        label: '/resume',
        description: context.currentSessionId
          ? 'Check and resume the current session'
          : 'Open the newest resumable or recent session',
        insertValue: '/resume',
      }
    case '/continue':
      return {
        label: '/continue',
        description: context.currentSessionId
          ? 'Check and continue the current session'
          : 'Open the newest resumable or recent session',
        insertValue: '/continue',
      }
    case '/rewind':
      return {
        label: '/rewind',
        description: 'Fork before the latest user turn',
        insertValue: '/rewind',
      }
    case '/context':
      return {
        label: '/context',
        description:
          context.contextPercent != null
            ? `Show context map (${context.contextPercent}% used)`
            : 'Show context map',
        insertValue: '/context',
      }
    case '/max-tokens':
      return {
        label: '/max-tokens',
        description: 'Inspect the current output token cap',
        insertValue: '/max-tokens',
      }
    case '/usage':
      return {
        label: '/usage',
        description: 'Open or refresh the usage dashboard',
        insertValue: '/usage',
      }
    case '/tps':
      return {
        label: '/tps',
        description: 'Show token-per-second speed for the current session',
        insertValue: '/tps',
      }
    case '/theme':
      return {
        label: '/theme',
        description: context.currentThemeId
          ? `Toggle from ${context.currentThemeId} theme`
          : 'Toggle the active theme',
        insertValue: '/theme',
      }
    case '/swarm':
      return {
        label: '/swarm help',
        description: context.lastSwarmRunId
          ? `Show swarm help; last run ${context.lastSwarmRunId.slice(0, 8)} is available`
          : 'Show swarm help',
        insertValue: '/swarm',
      }
    default:
      return null
  }
}

function dedupeItems(items: CommandPaletteItem[]): CommandPaletteItem[] {
  const seen = new Set<string>()
  return items.filter((item) => {
    if (seen.has(item.insertValue)) {
      return false
    }
    seen.add(item.insertValue)
    return true
  })
}

function findMatchingCommands(commands: SlashCommand[], query: string): SlashCommand[] {
  const startsWith = commands.filter((command) =>
    command.name.slice(1).toLowerCase().startsWith(query),
  )
  if (startsWith.length > 0) {
    return startsWith
  }

  const includes = commands.filter((command) => command.name.slice(1).toLowerCase().includes(query))
  if (includes.length > 0) {
    return includes
  }

  return commands.filter((command) => command.description.toLowerCase().includes(query))
}

function matchesArgumentItem(
  item: CommandPaletteItem,
  commandName: string,
  query: string,
): boolean {
  if (!query) {
    return true
  }

  const argument = item.insertValue.slice(commandName.length).trim().toLowerCase()
  return argument.includes(query) || item.description.toLowerCase().includes(query)
}

function compareArgumentItems(
  left: CommandPaletteItem,
  right: CommandPaletteItem,
  commandName: string,
  query: string,
): number {
  if (!query) {
    return 0
  }

  const leftArgument = left.insertValue.slice(commandName.length).trim().toLowerCase()
  const rightArgument = right.insertValue.slice(commandName.length).trim().toLowerCase()
  const leftExact = leftArgument === query
  const rightExact = rightArgument === query
  if (leftExact !== rightExact) {
    return leftExact ? -1 : 1
  }

  const leftFirstToken = leftArgument.split(/\s+/, 1)[0] ?? ''
  const rightFirstToken = rightArgument.split(/\s+/, 1)[0] ?? ''
  const leftFirstTokenExact = leftFirstToken === query
  const rightFirstTokenExact = rightFirstToken === query
  if (leftFirstTokenExact !== rightFirstTokenExact) {
    return leftFirstTokenExact ? -1 : 1
  }

  const leftStartsWith = leftArgument.startsWith(query)
  const rightStartsWith = rightArgument.startsWith(query)

  if (leftStartsWith !== rightStartsWith) {
    return leftStartsWith ? -1 : 1
  }

  return 0
}
