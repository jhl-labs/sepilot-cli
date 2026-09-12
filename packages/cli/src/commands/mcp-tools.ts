import { readFile } from 'node:fs/promises'
import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export interface McpToolsOptions {
  url?: string
}

export interface McpToolsCallOptions extends McpToolsOptions {
  input?: string
  inputFile?: string
}

function parseToolInput(raw: string, source: string): Record<string, unknown> {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(`Invalid JSON in ${source}: ${message}`)
  }

  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error(`${source} must be a JSON object`)
  }

  return parsed as Record<string, unknown>
}

async function readStdin(): Promise<string> {
  process.stdin.setEncoding('utf8')
  let text = ''
  for await (const chunk of process.stdin) {
    text += chunk
  }
  return text
}

async function resolveToolInput(options: McpToolsCallOptions): Promise<Record<string, unknown>> {
  if (options.input && options.inputFile) {
    throw new Error('Use either --input or --input-file, not both')
  }
  if (options.inputFile) {
    const raw = options.inputFile === '-'
      ? await readStdin()
      : await readFile(options.inputFile, 'utf-8')
    return parseToolInput(raw, options.inputFile === '-' ? 'stdin' : options.inputFile)
  }
  if (options.input) {
    return parseToolInput(options.input, '--input')
  }
  return {}
}

export async function mcpToolsListCommand(server: string | undefined, options: McpToolsOptions) {
  const client = new DaemonClient(options.url)

  if (server) {
    const result = await client.mcpServerTools(server)
    output(result, (r) => {
      const lines: string[] = [`${chalk.bold(server)} tools:`]
      if (r.quarantined) {
        lines.push(chalk.red('  QUARANTINED — review these advertised tools, then run `sepilot mcp trust-manifest <server>`.'))
        for (const tool of r.advertised ?? []) {
          lines.push(`  ${chalk.yellow('advertised')} ${tool}`)
        }
        return lines.join('\n')
      }
      for (const tool of r.enabled) {
        lines.push(`  ${chalk.green('enabled')}   ${tool}`)
      }
      for (const tool of r.disabled) {
        lines.push(`  ${chalk.red('disabled')}  ${tool}`)
      }
      return lines.join('\n')
    })
  } else {
    const servers = await client.mcpServers()
    output(servers, (list) => {
      if (!list.length) return 'No MCP servers configured.'
      return list
        .map((s) => {
          const status = s.status === 'connected' ? chalk.green(s.status) : chalk.red(s.status)
          return `${chalk.bold(s.name)} (${status}) — ${s.toolCount} tools`
        })
        .join('\n')
    })
  }
}

export async function mcpToolsDisableCommand(
  server: string,
  tool: string,
  options: McpToolsOptions,
) {
  const client = new DaemonClient(options.url)
  await client.mcpDisableTool(server, tool)
  output({ ok: true, server, tool, action: 'disabled' }, (r) =>
    `Disabled tool ${chalk.bold(r.tool)} on server ${chalk.bold(r.server)}`,
  )
}

export async function mcpToolsEnableCommand(
  server: string,
  tool: string,
  options: McpToolsOptions,
) {
  const client = new DaemonClient(options.url)
  await client.mcpEnableTool(server, tool)
  output({ ok: true, server, tool, action: 'enabled' }, (r) =>
    `Enabled tool ${chalk.bold(r.tool)} on server ${chalk.bold(r.server)}`,
  )
}

export async function mcpToolsCallCommand(
  server: string,
  tool: string,
  options: McpToolsCallOptions,
) {
  const input = await resolveToolInput(options)
  const client = new DaemonClient(options.url)
  const result = await client.mcpCallTool(server, tool, input)
  output(result, (r) => {
    if (r.output.trim().length > 0) return r.output
    return `${r.status} (${r.durationMs}ms)`
  })
}
