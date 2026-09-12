import type { McpCompletionRef } from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

interface McpCompletionOptions {
  url?: string
  prompt?: string
  resource?: string
  arg: string
  context?: string[]
}

function parsePair(raw: string, source: string): [string, string] {
  const eq = raw.indexOf('=')
  if (eq <= 0) throw new Error(`Invalid ${source}: ${raw}. Expected key=value.`)
  return [raw.slice(0, eq), raw.slice(eq + 1)]
}

function parseContext(entries: string[] | undefined): Record<string, string> | undefined {
  if (!entries?.length) return undefined
  const context: Record<string, string> = {}
  for (const entry of entries) {
    const [key, value] = parsePair(entry, '--context')
    context[key] = value
  }
  return context
}

function resolveRef(options: McpCompletionOptions): McpCompletionRef {
  if (options.prompt && options.resource) {
    throw new Error('Use either --prompt or --resource, not both')
  }
  if (options.prompt) return { type: 'ref/prompt', name: options.prompt }
  if (options.resource) return { type: 'ref/resource', uri: options.resource }
  throw new Error('Use --prompt <name> or --resource <uri>')
}

export async function mcpCompletionCommand(
  server: string,
  options: McpCompletionOptions,
) {
  const [name, value] = parsePair(options.arg, '--arg')
  const context = parseContext(options.context)
  const client = new DaemonClient(options.url)
  const result = await client.mcpComplete(server, {
    ref: resolveRef(options),
    argument: { name, value },
    ...(context ? { context: { arguments: context } } : {}),
  })
  output(result, (data) => {
    if (!data.completion.values.length) return 'No completions.'
    const suffix = data.completion.hasMore ? '\n... more available' : ''
    return `${data.completion.values.map((item) => `  ${item}`).join('\n')}${suffix}`
  })
}
