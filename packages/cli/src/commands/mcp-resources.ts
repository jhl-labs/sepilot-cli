import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

type ResourceListEntry = Awaited<ReturnType<DaemonClient['mcpResources']>>[number]
type ResourceTemplateEntry = Awaited<ReturnType<DaemonClient['mcpResourceTemplates']>>[number]
type ResourceReadResult = Awaited<ReturnType<DaemonClient['mcpReadResource']>>
type ResourceUpdate = Awaited<ReturnType<DaemonClient['mcpResourceUpdates']>>[number]

function formatResourceLine(resource: ResourceListEntry): string {
  return `  ${resource.uri.padEnd(40)} ${resource.mimeType ?? ''}`.trimEnd()
}

function formatTemplateLine(template: ResourceTemplateEntry): string {
  return `  ${template.uriTemplate.padEnd(40)} ${template.mimeType ?? ''}`.trimEnd()
}

function formatResourceReadResult(result: ResourceReadResult): string {
  return result.contents
    .map((content) => {
      if (typeof content.text === 'string') return content.text
      if (typeof content.blob === 'string') {
        const mimeType = content.mimeType ?? 'application/octet-stream'
        const bytes = Buffer.byteLength(content.blob, 'base64')
        return `<blob:${mimeType} ${bytes} bytes>`
      }
      return ''
    })
    .join('\n')
}

function formatResourceUpdate(update: ResourceUpdate): string {
  return `${update.timestamp}  ${update.uri}`
}

export async function mcpResourcesListCommand(
  server: string | undefined,
  options: { url?: string },
) {
  const client = new DaemonClient(options.url)
  if (server) {
    const resources = await client.mcpResources(server)
    output(resources, (items) =>
      items.length ? items.map(formatResourceLine).join('\n') : `No resources on ${server}.`,
    )
    return
  }
  const servers = await client.mcpServers()
  const entries: Array<{ server: string; resources: ResourceListEntry[] }> = []
  for (const s of servers) {
    if (s.status !== 'connected') continue
    const resources = await client.mcpResources(s.name).catch(() => [])
    if (!resources.length) continue
    entries.push({ server: s.name, resources })
  }
  output(entries, (items) => {
    if (!items.length) return 'No resources found on connected MCP servers.'
    return items
      .map((entry) => [
        chalk.cyan(entry.server),
        ...entry.resources.map(formatResourceLine),
      ].join('\n'))
      .join('\n')
  })
}

export async function mcpResourcesTemplatesCommand(
  server: string | undefined,
  options: { url?: string },
) {
  const client = new DaemonClient(options.url)
  if (server) {
    const templates = await client.mcpResourceTemplates(server)
    output(templates, (items) =>
      items.length ? items.map(formatTemplateLine).join('\n') : `No resource templates on ${server}.`,
    )
    return
  }
  const servers = await client.mcpServers()
  const entries: Array<{ server: string; templates: ResourceTemplateEntry[] }> = []
  for (const s of servers) {
    if (s.status !== 'connected') continue
    const templates = await client.mcpResourceTemplates(s.name).catch(() => [])
    if (!templates.length) continue
    entries.push({ server: s.name, templates })
  }
  output(entries, (items) => {
    if (!items.length) return 'No resource templates found on connected MCP servers.'
    return items
      .map((entry) => [
        chalk.cyan(entry.server),
        ...entry.templates.map(formatTemplateLine),
      ].join('\n'))
      .join('\n')
  })
}

export async function mcpResourcesReadCommand(
  server: string,
  uri: string,
  options: { url?: string },
) {
  const client = new DaemonClient(options.url)
  const result = await client.mcpReadResource(server, uri)
  output(result, formatResourceReadResult)
}

export async function mcpResourcesSubscribeCommand(
  server: string,
  uri: string,
  options: { url?: string },
) {
  const client = new DaemonClient(options.url)
  const result = await client.mcpSubscribeResource(server, uri)
  output(result, (value) => `Subscribed ${chalk.bold(value.uri)} on ${chalk.bold(server)}`)
}

export async function mcpResourcesUnsubscribeCommand(
  server: string,
  uri: string,
  options: { url?: string },
) {
  const client = new DaemonClient(options.url)
  const result = await client.mcpUnsubscribeResource(server, uri)
  output(result, (value) => `Unsubscribed ${chalk.bold(value.uri)} on ${chalk.bold(server)}`)
}

export async function mcpResourcesSubscriptionsCommand(
  server: string,
  options: { url?: string },
) {
  const client = new DaemonClient(options.url)
  const subscriptions = await client.mcpResourceSubscriptions(server)
  output(subscriptions, (items) =>
    items.length ? items.map((uri) => `  ${uri}`).join('\n') : `No resource subscriptions on ${server}.`,
  )
}

export async function mcpResourcesUpdatesCommand(
  server: string,
  options: { url?: string; limit?: string },
) {
  const client = new DaemonClient(options.url)
  let limit: number | undefined
  if (options.limit) {
    limit = Number.parseInt(options.limit, 10)
    if (!Number.isInteger(limit) || limit <= 0) {
      throw new Error('--limit must be a positive integer')
    }
  }
  const updates = await client.mcpResourceUpdates(server, limit)
  output(updates, (items) =>
    items.length ? items.map(formatResourceUpdate).join('\n') : `No resource updates on ${server}.`,
  )
}
