export type SettingsCategory =
  | 'model'
  | 'agent'
  | 'permissions'
  | 'mcp'
  | 'skills'
  | 'memory'
  | 'appearance'
  | 'diagnostics'

export type CommandCategory = SettingsCategory | 'session' | 'run'

export interface CommandContext {
  openDialog(id: string): void
  setSessionValue(key: string, value: unknown): void
  runSlash(raw: string): void
  copy(text: string): Promise<boolean>
  exit(): void
}

export interface CommandDef {
  id: string
  title: string
  category: CommandCategory
  aliases: string[]
  keys: string[]
  surface: 'primary' | 'secondary'
  run(ctx: CommandContext): void | Promise<void>
}
