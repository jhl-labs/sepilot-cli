import { discoverCustomDefs } from './loader.js'
import { compileCustomAgents, type CustomAgentRecord } from './agents.js'
import { compileCustomCommands, type CustomCommand } from './commands.js'

/**
 * Per-request resolver for custom command/agent definitions.
 *
 * The daemon is shared by N projects (1 daemon ↔ N desktop/cli/web), so
 * definitions under `<project>/.sepilotd/{commands,agents}` must be resolved
 * against the *request's* cwd, not the daemon's boot cwd. Discovery re-reads
 * the markdown sources on every call — the def sets are a handful of small
 * files, so a fresh read per chat turn is cheap and edits apply without a
 * daemon restart. Requests without a cwd only see home-level definitions.
 */
export class CustomDefsService {
  private readonly home?: string

  constructor(opts: { home?: string } = {}) {
    this.home = opts.home
  }

  async commandsForCwd(
    cwd: string | undefined,
    boundaryRoot?: string,
  ): Promise<CustomCommand[]> {
    return compileCustomCommands(
      await discoverCustomDefs('commands', {
        home: this.home,
        cwd: cwd ?? null,
        boundaryRoot,
      }),
    )
  }

  async agentsForCwd(
    cwd: string | undefined,
    boundaryRoot?: string,
  ): Promise<CustomAgentRecord[]> {
    return compileCustomAgents(
      await discoverCustomDefs('agents', {
        home: this.home,
        cwd: cwd ?? null,
        boundaryRoot,
      }),
    )
  }
}
