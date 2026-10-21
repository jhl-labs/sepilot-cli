interface SlashCommandContext {
  args: string[]
  rawCommand: string
}

export type SlashCommandHandler = (
  ctx: SlashCommandContext,
) => void | Promise<void>

export interface SlashCommand {
  name: string
  description?: string
  handler: SlashCommandHandler
}

export class SlashCommandRegistry {
  private readonly entries = new Map<string, SlashCommand>()

  register(command: SlashCommand): void {
    if (!command.name.startsWith('/')) {
      throw new Error(
        `SlashCommandRegistry.register: name must start with "/", got ${command.name}`,
      )
    }
    if (this.entries.has(command.name)) {
      throw new Error(
        `SlashCommandRegistry.register: duplicate command ${command.name}`,
      )
    }
    this.entries.set(command.name, command)
  }

  get(name: string): SlashCommand | undefined {
    return this.entries.get(name)
  }

  has(name: string): boolean {
    return this.entries.has(name)
  }

  list(): SlashCommand[] {
    return Array.from(this.entries.values())
  }
}
