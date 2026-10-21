import { discoverCustomDefs } from './loader.js'
import { compileCustomAgents, type CustomAgentRecord } from './agents.js'
import { compileCustomCommands, type CustomCommand } from './commands.js'
import { createLogger } from '../../logger.js'

const log = createLogger('custom-agents')
const MAX_REPORTED_WARNINGS = 512
const reportedWarnings = new Set<string>()

function reportAgentWarnings(records: CustomAgentRecord[]): CustomAgentRecord[] {
  for (const record of records) {
    for (const warning of record.warnings ?? []) {
      const key = `${record.source}\0${warning}`
      if (reportedWarnings.has(key) || reportedWarnings.size >= MAX_REPORTED_WARNINGS) continue
      reportedWarnings.add(key)
      log.warn('custom agent setting ignored', { agent: record.id, source: record.source, warning })
    }
  }
  return records
}

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
    return reportAgentWarnings(compileCustomAgents(
      await discoverCustomDefs('agents', {
        home: this.home,
        cwd: cwd ?? null,
        boundaryRoot,
      }),
    ))
  }
}
