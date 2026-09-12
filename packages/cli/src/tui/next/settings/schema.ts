import type { SettingsCategory } from '../commands/types.js'

export type { SettingsCategory } from '../commands/types.js'

export type SettingsScope = 'session' | 'persisted' | 'command'

export interface SettingsItem {
  id: string
  category: SettingsCategory
  label: string
  scope: SettingsScope
  kind: 'choice' | 'toggle' | 'text' | 'action'
  sessionKey?: string
  /** Dotted key accepted by PUT /api/v1/config. */
  configPath?: string
  /** Existing picker or panel opened for non-scalar settings. */
  dialogId?: string
}

const CATEGORY_ORDER: SettingsCategory[] = [
  'model',
  'agent',
  'permissions',
  'mcp',
  'skills',
  'memory',
  'appearance',
  'diagnostics',
]

export const SETTINGS_ITEMS: SettingsItem[] = [
  sessionChoice('model.session', 'model', 'Model (this session)', 'model'),
  persistedChoice('model.default', 'model', 'Default model', 'agent.defaultModel'),
  persistedAction('model.providers', 'model', 'Providers', 'providers'),
  sessionChoice('agent.mode', 'agent', 'Agent mode (this session)', 'mode'),
  sessionChoice('agent.thinking', 'agent', 'Thinking level (this session)', 'thinkingLevel'),
  {
    id: 'agent.maxTokens',
    category: 'agent',
    label: 'Max output tokens (this session)',
    scope: 'session',
    kind: 'text',
    sessionKey: 'maxTokens',
  },
  sessionChoice(
    'permissions.autonomy',
    'permissions',
    'Autonomy (this session)',
    'autonomy',
  ),
  persistedChoice(
    'permissions.defaultAutonomy',
    'permissions',
    'Default autonomy',
    'agent.autonomy',
  ),
  persistedAction('mcp.servers', 'mcp', 'MCP servers', 'mcp'),
  persistedAction('skills.manage', 'skills', 'Installed skills', 'skills'),
  persistedAction('memory.search', 'memory', 'Search memory', 'memory'),
  persistedAction('memory.rag', 'memory', 'RAG sources', 'rag'),
  persistedAction('appearance.theme', 'appearance', 'Theme', 'theme'),
  persistedAction('diagnostics.doctor', 'diagnostics', 'Run doctor', 'doctor'),
  persistedAction('diagnostics.usage', 'diagnostics', 'Usage dashboard', 'usage'),
]

function sessionChoice(
  id: string,
  category: SettingsCategory,
  label: string,
  sessionKey: string,
): SettingsItem {
  return { id, category, label, scope: 'session', kind: 'choice', sessionKey }
}

function persistedChoice(
  id: string,
  category: SettingsCategory,
  label: string,
  configPath: string,
): SettingsItem {
  return { id, category, label, scope: 'persisted', kind: 'choice', configPath }
}

function persistedAction(
  id: string,
  category: SettingsCategory,
  label: string,
  dialogId: string,
): SettingsItem {
  return { id, category, label, scope: 'persisted', kind: 'action', dialogId }
}

export function settingsCategories(): SettingsCategory[] {
  return [...CATEGORY_ORDER]
}

export function itemsForCategory(category: SettingsCategory): SettingsItem[] {
  return SETTINGS_ITEMS.filter((item) => item.category === category)
}

export function scopeBadge(scope: SettingsScope): string {
  if (scope === 'session') return '[세션]'
  if (scope === 'persisted') return '[저장됨]'
  return '[명령]'
}
