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

/** Shell-owned effects available to declarative command definitions. */
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
  /** Slash aliases. The first alias is canonical. */
  aliases: string[]
  /** Key bindings in `leader <key>` form. */
  keys: string[]
  /** Primary commands appear in the default slash suggestions. */
  surface: 'primary' | 'secondary'
  run(ctx: CommandContext): void | Promise<void>
}
