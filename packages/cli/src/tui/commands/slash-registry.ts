// A tiny registry of slash command handlers.
//
// App.tsx historically dispatched ~28 slash commands through a single
// `switch (command)` block, which both blew up the file and forced every new
// command to edit existing code (an OCP smell). The registry lets us migrate
// commands out of the switch one at a time:
//
// 1. Build a registry inside the component (so handlers can keep capturing
//    the React state and dispatchers they need).
// 2. The dispatcher consults the registry first; anything not registered
//    falls back to the existing switch.
// 3. As commands move into the registry the corresponding switch case is
//    removed. Once the switch is empty, the dispatcher is just a registry
//    lookup.

interface SlashCommandContext {
  /** Tokens after the command, e.g. for `/usage 7d` -> `['7d']`. */
  args: string[]
  /** The original input including the slash prefix, useful for error copy. */
  rawCommand: string
}

export type SlashCommandHandler = (
  ctx: SlashCommandContext,
) => void | Promise<void>

export interface SlashCommand {
  /** Command including leading slash, e.g. `/help`. Case-sensitive. */
  name: string
  /** Short one-line description for autocomplete and help surfaces. */
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
