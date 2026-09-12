import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export interface McpSearchOptions {
  url?: string
}

export async function mcpSearchCommand(query: string, options: McpSearchOptions) {
  const client = new DaemonClient(options.url)
  const results = await client.mcpMarketplaceSearch(query)
  output(results, (templates) => {
    if (!templates.length) {
      return `No MCP servers found matching "${query}".`
    }
    return templates
      .map((t) => {
        const tags = t.tags.length ? ` ${chalk.gray(`[${t.tags.join(', ')}]`)}` : ''
        const marketplace = t.marketplace ? ` ${chalk.dim(`@${t.marketplace}`)}` : ''
        return `  ${chalk.bold(t.name)}${marketplace}  ${chalk.dim(t.transport)}  ${t.description}${tags}`
      })
      .join('\n')
  })
}
