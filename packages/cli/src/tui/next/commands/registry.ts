import type { CommandDef, SettingsCategory } from './types.js'

export type { CommandContext, CommandDef, SettingsCategory } from './types.js'

export const COMMANDS: CommandDef[] = [
  dialogCommand('shell.help', 'Show help', 'session', ['/help', '/?'], ['leader ?'], 'help'),
  dialogCommand(
    'settings.open',
    'Open settings',
    'appearance',
    ['/settings'],
    ['leader /'],
    'settings',
  ),
  dialogCommand(
    'model.pick',
    'Change model',
    'model',
    ['/model', '/m'],
    ['leader m'],
    'model',
  ),
  dialogCommand('mode.pick', 'Change agent mode', 'agent', ['/mode'], ['leader a'], 'mode'),
  slashCommand('session.new', 'Start a new session', 'session', ['/new'], ['leader n']),
  slashCommand('session.clear', 'Clear the transcript', 'session', ['/clear']),
  dialogCommand(
    'session.resume',
    'Resume a session',
    'session',
    ['/resume', '/continue'],
    ['leader s'],
    'sessions',
  ),
  slashCommand(
    'session.compact',
    'Compact the conversation',
    'session',
    ['/compact'],
    ['leader c'],
  ),
  dialogCommand(
    'project.pick',
    'Switch project',
    'session',
    ['/project'],
    ['leader p'],
    'projects',
  ),
  dialogCommand('skills.open', 'Manage skills', 'skills', ['/skills'], [], 'skills'),
  dialogCommand('memory.open', 'Search memory', 'memory', ['/memory'], [], 'memory'),
  slashCommand('diagnostics.doctor', 'Run diagnostics', 'diagnostics', ['/doctor']),

  slashCommand('run.skill', 'Run a named skill', 'skills', ['/run'], [], 'secondary'),
  slashCommand('run.steer', 'Steer the active run', 'agent', ['/steer'], [], 'secondary'),
  slashCommand('run.followup', 'Cancel queued follow-ups', 'agent', ['/followup'], [], 'secondary'),
  dialogCommand('thinking.pick', 'Thinking level', 'agent', ['/thinking'], [], 'thinking', 'secondary'),
  dialogCommand('maxTokens.pick', 'Max output tokens', 'agent', ['/max-tokens'], [], 'maxTokens', 'secondary'),
  slashCommand('memory.remember', 'Save a memory', 'memory', ['/remember'], [], 'secondary'),
  slashCommand('approvals.list', 'List approvals', 'permissions', ['/approvals'], [], 'secondary'),
  slashCommand('questions.list', 'List pending questions', 'agent', ['/questions'], [], 'secondary'),
  slashCommand('questions.answer', 'Answer a pending question', 'agent', ['/answer'], [], 'secondary'),
  slashCommand('context.show', 'Show session context', 'session', ['/context', '/status'], [], 'secondary'),
  slashCommand('workspace.diff', 'Show working tree diff', 'diagnostics', ['/diff'], [], 'secondary'),
  slashCommand('session.manage', 'Manage the current session', 'session', ['/session'], [], 'secondary'),
  slashCommand('session.rewind', 'Rewind user turns into a branch', 'session', ['/rewind', '/undo'], [], 'secondary'),
  slashCommand('session.redo', 'Return to the session before the last rewind', 'session', ['/redo'], [], 'secondary'),
  slashCommand('session.editLast', 'Edit the previous prompt in a new branch', 'session', ['/edit-last'], [], 'secondary'),
  slashCommand('session.rename', 'Rename the current session', 'session', ['/rename'], [], 'secondary'),
  slashCommand('session.share', 'Share the current session', 'session', ['/share'], [], 'secondary'),
  slashCommand('session.fork', 'Fork the current session', 'session', ['/fork'], [], 'secondary'),
  slashCommand('model.capability', 'Model capability overrides', 'model', ['/capability'], [], 'secondary'),
  slashCommand('diagnostics.tps', 'Token generation speed', 'diagnostics', ['/tps'], [], 'secondary'),
  slashCommand('conversation.details', 'Toggle tool execution details', 'diagnostics', ['/details'], ['leader d'], 'secondary'),
  slashCommand('commands.manage', 'Manage custom slash commands', 'agent', ['/commands'], [], 'secondary'),
  slashCommand('files.image', 'Attach an image path or clipboard PNG', 'session', ['/image'], [], 'secondary'),
  slashCommand('session.recap', 'End-of-turn recap', 'session', ['/recap'], [], 'secondary'),
  dialogCommand('files.open', 'Attach files', 'session', ['/files'], [], 'files', 'secondary'),
  slashCommand('integration.acp', 'ACP integration', 'diagnostics', ['/acp'], [], 'secondary'),
  slashCommand('integration.a2a', 'A2A integration', 'diagnostics', ['/a2a'], [], 'secondary'),
  slashCommand('hooks.manage', 'Outbound delivery hooks', 'permissions', ['/hooks'], [], 'secondary'),
  slashCommand('containers.list', 'Managed containers', 'diagnostics', ['/containers'], [], 'secondary'),
  slashCommand('swarm.manage', 'External agent swarm', 'agent', ['/swarm'], [], 'secondary'),
  slashCommand('copy.last', 'Copy last assistant reply', 'session', ['/copy'], ['leader y'], 'secondary'),
  slashCommand('copy.all', 'Copy whole conversation', 'session', ['/copy-all'], ['leader Y'], 'secondary'),
  slashCommand('composer.editor', 'Compose in external editor', 'session', ['/editor'], ['leader e'], 'secondary'),
  dialogCommand('mcp.open', 'MCP servers', 'mcp', ['/mcp'], [], 'mcp', 'secondary'),
  dialogCommand('rag.open', 'RAG sources', 'memory', ['/rag'], [], 'rag', 'secondary'),
  dialogCommand('usage.open', 'Usage dashboard', 'diagnostics', ['/usage'], [], 'usage', 'secondary'),
  dialogCommand(
    'artifacts.open',
    'Artifacts',
    'diagnostics',
    ['/artifacts'],
    [],
    'artifacts',
    'secondary',
  ),
  dialogCommand(
    'providers.open',
    'Change provider',
    'model',
    ['/provider'],
    [],
    'providers',
    'secondary',
  ),
  slashCommand('providers.inspect', 'Provider health', 'diagnostics', ['/providers'], [], 'secondary'),
  slashCommand('agent.planner', 'Live plan and todo progress', 'agent', ['/planner'], ['F11'], 'secondary'),
  slashCommand('agent.loop', 'Agent loop traversal', 'agent', ['/loop', '/graph'], ['F12'], 'secondary'),
  slashCommand('agent.rollbacks', 'Edit rollback checkpoints', 'agent', ['/rollbacks'], [], 'secondary'),
  slashCommand('agent.debate', 'Debate rounds', 'agent', ['/debate'], [], 'secondary'),
  dialogCommand(
    'autonomy.pick',
    'Autonomy level',
    'permissions',
    ['/autonomy'],
    [],
    'autonomy',
    'secondary',
  ),
  dialogCommand('theme.pick', 'Theme', 'appearance', ['/theme'], [], 'theme', 'secondary'),
  {
    id: 'shell.exit',
    title: 'Exit',
    category: 'session',
    aliases: ['/exit', '/quit', '/q'],
    keys: [],
    surface: 'secondary',
    run: (context) => context.exit(),
  },
]

function slashCommand(
  id: string,
  title: string,
  category: CommandDef['category'],
  aliases: string[],
  keys: string[] = [],
  surface: CommandDef['surface'] = 'primary',
): CommandDef {
  return {
    id,
    title,
    category,
    aliases,
    keys,
    surface,
    run: (context) => context.runSlash(aliases[0] ?? ''),
  }
}

function dialogCommand(
  id: string,
  title: string,
  category: CommandDef['category'],
  aliases: string[],
  keys: string[],
  dialogId: string,
  surface: CommandDef['surface'] = 'primary',
): CommandDef {
  return {
    id,
    title,
    category,
    aliases,
    keys,
    surface,
    run: (context) => context.openDialog(dialogId),
  }
}

const ALIAS_INDEX = new Map(
  COMMANDS.flatMap((command) => command.aliases.map((alias) => [alias, command] as const)),
)

export function findByAlias(alias: string): CommandDef | undefined {
  return ALIAS_INDEX.get(alias)
}

export function primaryCommands(): CommandDef[] {
  return COMMANDS.filter((command) => command.surface === 'primary')
}

export function commandsByCategory(category: SettingsCategory): CommandDef[] {
  return COMMANDS.filter((command) => command.category === category)
}

export function leaderBindings(): Map<string, CommandDef> {
  const bindings = new Map<string, CommandDef>()

  for (const command of COMMANDS) {
    for (const key of command.keys) {
      if (key.startsWith('leader ')) bindings.set(key.slice('leader '.length), command)
    }
  }

  return bindings
}
