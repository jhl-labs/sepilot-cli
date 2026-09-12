import type {
  DaemonMcpServerConfig,
  DaemonMcpServerStatus,
  McpMarketplace,
  McpInstallResult,
  McpCompletionInput,
  McpLoggingLevel,
  McpServerTemplate,
  McpServerToolsResult,
} from '@sepilotd/api-client'
import {
  buildPlaywrightMcpServerConfig,
  type PlaywrightMcpOptions,
} from '../../utils/playwright-mcp.js'
import { truncateJsonOutput } from './admin-output.js'

interface McpServerStatus {
  name: DaemonMcpServerStatus['name']
  enabled: DaemonMcpServerStatus['enabled']
  status: DaemonMcpServerStatus['status'] | string
  transport?: DaemonMcpServerStatus['transport']
  command?: DaemonMcpServerStatus['command']
  args?: DaemonMcpServerStatus['args']
  url?: DaemonMcpServerStatus['url']
  toolCount: DaemonMcpServerStatus['toolCount']
  tools?: DaemonMcpServerStatus['tools']
  allTools?: DaemonMcpServerStatus['allTools']
  disabledTools?: DaemonMcpServerStatus['disabledTools']
  toolManifestStatus?: DaemonMcpServerStatus['toolManifestStatus']
  securityAlerts?: DaemonMcpServerStatus['securityAlerts']
  error?: DaemonMcpServerStatus['error']
}

interface McpSlashClient {
  mcpServers(): Promise<McpServerStatus[]>
  mcpMarketplaceSearch(query: string): Promise<McpServerTemplate[]>
  mcpInstall(name: string, marketplace?: string): Promise<McpInstallResult>
  deleteMcpServer(name: string): Promise<unknown>
  setMcpServerEnabled(name: string, enabled: boolean): Promise<unknown>
  trustMcpServerManifest(name: string): Promise<{ name: string; digest: string; trusted: true }>
  upsertMcpServer(server: DaemonMcpServerConfig): Promise<unknown>
  mcpServerTools(serverName: string): Promise<McpServerToolsResult>
  mcpDisableTool(serverName: string, tool: string): Promise<unknown>
  mcpEnableTool(serverName: string, tool: string): Promise<unknown>
  mcpMarketplaceList(): Promise<McpMarketplace[]>
  mcpMarketplaceAdd(name: string, url: string): Promise<McpMarketplace>
  mcpMarketplaceRemove(name: string): Promise<{ removed: boolean }>
  mcpPrompts(
    server: string,
  ): Promise<
    Array<{
      name: string
      description?: string
      arguments?: Array<{ name: string; required?: boolean }>
    }>
  >
  getMcpPrompt(server: string, prompt: string, args?: Record<string, string>): Promise<unknown>
  mcpResources(
    server: string,
  ): Promise<
    Array<{ uri: string; name?: string; title?: string; description?: string; mimeType?: string }>
  >
  mcpResourceTemplates(
    server: string,
  ): Promise<Array<{ uriTemplate: string; name?: string; title?: string; description?: string }>>
  mcpReadResource(server: string, uri: string): Promise<unknown>
  mcpSubscribeResource(server: string, uri: string): Promise<unknown>
  mcpUnsubscribeResource(server: string, uri: string): Promise<unknown>
  mcpResourceSubscriptions(server: string): Promise<string[]>
  mcpResourceUpdates(server: string, limit?: number): Promise<unknown[]>
  mcpComplete(server: string, input: McpCompletionInput): Promise<unknown>
  mcpSetLoggingLevel(server: string, level: McpLoggingLevel): Promise<unknown>
  mcpLogs(server: string): Promise<unknown>
  mcpMetrics(): Promise<unknown>
  mcpMetricsForServer(server: string): Promise<unknown>
  mcpCallTool(serverName: string, tool: string, args?: Record<string, unknown>): Promise<unknown>
}

function formatServer(server: McpServerStatus): string {
  const status = server.enabled ? server.status : 'disabled'
  const transport = server.transport ?? (server.command ? 'stdio' : 'unknown')
  const command = server.command
    ? ` · ${server.command}${server.args?.length ? ` ${server.args.join(' ')}` : ''}`
    : server.url
      ? ` · ${server.url}`
      : ''
  const error = server.error ? ` · error: ${server.error}` : ''
  const security = server.toolManifestStatus === 'changed'
    ? ' · manifest:changed · tools:quarantined'
    : server.securityAlerts?.length
      ? ` · security-alerts:${server.securityAlerts.length}`
      : ''
  return `${server.name} · ${status} · ${transport} · tools:${server.toolCount}${command}${security}${error}`
}

function help(): string {
  return [
    'Usage:',
    '  /mcp                         Toggle MCP status panel',
    '  /mcp list                    List configured MCP servers',
    '  /mcp add <name> --command <cmd> [--arg value] [--env KEY=VALUE] [--disabled]',
    '  /mcp add <name> --transport sse|http --url <url> [--header KEY=VALUE]',
    '  /mcp search <query>          Search MCP marketplaces',
    '  /mcp install <name> [source] Install a marketplace server',
    '  /mcp playwright [--mode auto|visible|headless] [--name playwright] [--user-agent value] [--proxy-server url]',
    '  /mcp enable <name>           Enable a server',
    '  /mcp disable <name>          Disable a server',
    '  /mcp trust-manifest <name>   Trust quarantined tool changes after review',
    '  /mcp remove <name>           Remove a server',
    '  /mcp tools <name>            List enabled/disabled tools',
    '  /mcp tools disable <server> <tool>',
    '  /mcp tools enable <server> <tool>',
    '  /mcp prompts list [server] | get <server/prompt> [--arg KEY=VALUE]',
    '  /mcp resources list|templates [server] | read|subscribe|unsubscribe <server> <uri>',
    '  /mcp complete <server> --prompt <name>|--resource <uri> --arg KEY=VALUE',
    '  /mcp logging set-level <server> <debug|info|notice|warning|error|critical|alert|emergency>',
    '  /mcp logging logs <server>',
    '  /mcp metrics [server]',
    '  /mcp call <server> <tool> [--input JSON]',
    '  /mcp marketplace list        List MCP marketplaces',
    '  /mcp marketplace add <name> <url>',
    '  /mcp marketplace remove <name>',
  ].join('\n')
}

function parseValueFlag(
  args: string[],
  index: number,
  flag: string,
): { value: string | null; nextIndex: number } {
  const current = args[index]
  if (!current) return { value: null, nextIndex: index + 1 }
  if (current.startsWith(`${flag}=`)) {
    return { value: current.slice(flag.length + 1), nextIndex: index + 1 }
  }
  const next = args[index + 1]
  if (!next || next.startsWith('--')) {
    return { value: null, nextIndex: index + 1 }
  }
  return { value: next, nextIndex: index + 2 }
}

function parseKeyValueEntries(entries: string[], label: string): Record<string, string> {
  const record: Record<string, string> = {}
  for (const entry of entries) {
    const separator = entry.indexOf('=')
    if (separator <= 0) {
      throw new Error(`Invalid ${label} entry: ${entry}. Use KEY=VALUE.`)
    }
    record[entry.slice(0, separator)] = entry.slice(separator + 1)
  }
  return record
}

function parseJsonObject(value: string, label: string): Record<string, unknown> | string {
  try {
    const parsed = JSON.parse(value) as unknown
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
  } catch {
    // Return a user-facing usage string below.
  }
  return `Invalid ${label}: expected a JSON object.`
}

function parsePositiveInt(value: string | undefined): number | undefined {
  if (!value) return undefined
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined
}

function parseLoggingLevel(value: string | undefined): McpLoggingLevel | null {
  if (
    value === 'debug' ||
    value === 'info' ||
    value === 'notice' ||
    value === 'warning' ||
    value === 'error' ||
    value === 'critical' ||
    value === 'alert' ||
    value === 'emergency'
  ) {
    return value
  }
  return null
}

function formatJson(value: unknown): string {
  return truncateJsonOutput(JSON.stringify(value, null, 2))
}

function parseAddServerConfig(args: string[]): DaemonMcpServerConfig | string {
  const name = args[0]
  if (!name) return 'Usage: /mcp add <name> --command <cmd> [--arg value] [--env KEY=VALUE]'

  let transport: 'stdio' | 'sse' | 'http' = 'stdio'
  let command: string | null = null
  let url: string | null = null
  const serverArgs: string[] = []
  const envEntries: string[] = []
  const headerEntries: string[] = []
  let enabled = true

  for (let index = 1; index < args.length; ) {
    const arg = args[index]!
    if (arg === '--disabled') {
      enabled = false
      index += 1
      continue
    }
    if (arg === '--transport' || arg.startsWith('--transport=')) {
      const parsed = parseValueFlag(args, index, '--transport')
      if (parsed.value === 'stdio' || parsed.value === 'sse' || parsed.value === 'http') {
        transport = parsed.value
      } else {
        return 'Usage: /mcp add <name> --transport stdio|sse|http ...'
      }
      index = parsed.nextIndex
      continue
    }
    if (arg === '--command' || arg.startsWith('--command=')) {
      const parsed = parseValueFlag(args, index, '--command')
      command = parsed.value
      index = parsed.nextIndex
      continue
    }
    if (arg === '--url' || arg.startsWith('--url=')) {
      const parsed = parseValueFlag(args, index, '--url')
      url = parsed.value
      index = parsed.nextIndex
      continue
    }
    if (arg === '--arg' || arg.startsWith('--arg=')) {
      const parsed = parseValueFlag(args, index, '--arg')
      if (parsed.value != null) serverArgs.push(parsed.value)
      index = parsed.nextIndex
      continue
    }
    if (arg === '--env' || arg.startsWith('--env=')) {
      const parsed = parseValueFlag(args, index, '--env')
      if (parsed.value != null) envEntries.push(parsed.value)
      index = parsed.nextIndex
      continue
    }
    if (arg === '--header' || arg.startsWith('--header=')) {
      const parsed = parseValueFlag(args, index, '--header')
      if (parsed.value != null) headerEntries.push(parsed.value)
      index = parsed.nextIndex
      continue
    }
    return `Unknown /mcp add flag: ${arg}`
  }

  if (transport === 'stdio') {
    if (!command) return 'Usage: /mcp add <name> --command <cmd> [--arg value] [--env KEY=VALUE]'
    return {
      name,
      enabled,
      transport,
      command,
      args: serverArgs,
      env: parseKeyValueEntries(envEntries, '--env'),
    }
  }

  if (!url) return 'Usage: /mcp add <name> --transport sse|http --url <url> [--header KEY=VALUE]'
  return {
    name,
    enabled,
    transport,
    url,
    headers: parseKeyValueEntries(headerEntries, '--header'),
  }
}

function parsePlaywrightConfig(args: string[]): PlaywrightMcpOptions | string {
  const options: PlaywrightMcpOptions = {}

  for (let index = 0; index < args.length; ) {
    const arg = args[index]!
    if (arg === '--isolated') {
      options.isolated = true
      index += 1
      continue
    }
    if (arg === '--no-sandbox') {
      options.noSandbox = true
      index += 1
      continue
    }
    if (arg === '--ignore-https-errors') {
      options.ignoreHttpsErrors = true
      index += 1
      continue
    }
    if (arg === '--block-service-workers') {
      options.blockServiceWorkers = true
      index += 1
      continue
    }
    if (arg === '--save-session') {
      options.saveSession = true
      index += 1
      continue
    }
    if (arg === '--disabled') {
      options.disabled = true
      index += 1
      continue
    }

    const valueFlags: Record<string, keyof PlaywrightMcpOptions> = {
      '--name': 'name',
      '--mode': 'mode',
      '--browser': 'browser',
      '--user-data-dir': 'userDataDir',
      '--caps': 'caps',
      '--mcp-package': 'mcpPackage',
      '--storage-state': 'storageState',
      '--output-dir': 'outputDir',
      '--viewport-size': 'viewportSize',
      '--user-agent': 'userAgent',
      '--device': 'device',
      '--proxy-server': 'proxyServer',
      '--proxy-bypass': 'proxyBypass',
      '--timeout-action': 'timeoutAction',
      '--timeout-navigation': 'timeoutNavigation',
      '--image-responses': 'imageResponses',
      '--allowed-origins': 'allowedOrigins',
      '--blocked-origins': 'blockedOrigins',
    }
    const flag = Object.keys(valueFlags).find(
      (candidate) => arg === candidate || arg.startsWith(`${candidate}=`),
    )
    if (flag) {
      const parsed = parseValueFlag(args, index, flag)
      if (!parsed.value) return `Usage: /mcp playwright ${flag} <value>`
      options[valueFlags[flag]!] = parsed.value as never
      index = parsed.nextIndex
      continue
    }

    return `Unknown /mcp playwright flag: ${arg}`
  }

  return options
}

export async function runMcpSlashCommand(client: McpSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'help'

  switch (action) {
    case 'help':
      return help()
    case 'list': {
      const servers = await client.mcpServers()
      return servers.length > 0
        ? servers.map(formatServer).join('\n')
        : 'No MCP servers configured. Use /mcp search <query> or /mcp install <name>.'
    }
    case 'add': {
      const config = parseAddServerConfig(args.slice(1))
      if (typeof config === 'string') return config
      await client.upsertMcpServer(config)
      return `MCP server saved: ${config.name} · ${config.transport ?? 'stdio'}${config.enabled === false ? ' · disabled' : ''}`
    }
    case 'search': {
      const query = args.slice(1).join(' ').trim()
      if (!query) return 'Usage: /mcp search <query>'
      const results = await client.mcpMarketplaceSearch(query)
      return results.length > 0
        ? results
            .map((template) => {
              const tags = template.tags?.length ? ` [${template.tags.join(', ')}]` : ''
              return `${template.name} · ${template.transport} · ${template.marketplace}\n  ${template.description}${tags}\n  install: /mcp install ${template.name} ${template.marketplace}`
            })
            .join('\n')
        : `No MCP servers found matching "${query}".`
    }
    case 'install': {
      const name = args[1]
      const marketplace = args[2]
      if (!name) return 'Usage: /mcp install <name> [marketplace]'
      const result = await client.mcpInstall(name, marketplace)
      return result.installed
        ? `Installed MCP server: ${result.serverName}`
        : `Failed to install MCP server: ${name}`
    }
    case 'playwright': {
      const parsed = parsePlaywrightConfig(args.slice(1))
      if (typeof parsed === 'string') return parsed
      try {
        const config = buildPlaywrightMcpServerConfig(parsed)
        await client.upsertMcpServer(config.server)
        const profile = config.isolated ? 'isolated' : config.userDataDir
        return [
          `Playwright MCP saved: ${config.server.name}`,
          `mode: ${config.resolvedMode}${config.requestedMode !== config.resolvedMode ? ` (${config.requestedMode})` : ''}`,
          `browser: ${config.browser}`,
          `profile: ${profile ?? 'default'}`,
        ].join('\n')
      } catch (error) {
        return error instanceof Error ? error.message : String(error)
      }
    }
    case 'enable':
    case 'disable': {
      const name = args[1]
      if (!name) return `Usage: /mcp ${action} <name>`
      await client.setMcpServerEnabled(name, action === 'enable')
      return `MCP server ${action === 'enable' ? 'enabled' : 'disabled'}: ${name}`
    }
    case 'trust-manifest': {
      const name = args[1]
      if (!name) return 'Usage: /mcp trust-manifest <name>'
      const result = await client.trustMcpServerManifest(name)
      return `Trusted current MCP tool manifest: ${result.name}\ndigest: ${result.digest}`
    }
    case 'remove': {
      const name = args[1]
      if (!name) return 'Usage: /mcp remove <name>'
      await client.deleteMcpServer(name)
      return `MCP server removed: ${name}`
    }
    case 'tools': {
      const subcommand = args[1]?.toLowerCase()
      if (subcommand === 'disable' || subcommand === 'enable') {
        const server = args[2]
        const tool = args[3]
        if (!server || !tool) return `Usage: /mcp tools ${subcommand} <server> <tool>`
        if (subcommand === 'disable') {
          await client.mcpDisableTool(server, tool)
        } else {
          await client.mcpEnableTool(server, tool)
        }
        return `MCP tool ${subcommand === 'disable' ? 'disabled' : 'enabled'}: ${server}/${tool}`
      }

      const name = args[1]
      if (!name) return 'Usage: /mcp tools <name>'
      const tools = await client.mcpServerTools(name)
      if (tools.quarantined) {
        const advertised = tools.advertised?.length
          ? tools.advertised.map((tool) => `  ! ${tool}`).join('\n')
          : '  (none advertised)'
        return [
          `${name} tools — QUARANTINED`,
          'Review advertised tools:',
          advertised,
          `Trust only after review: /mcp trust-manifest ${name}`,
        ].join('\n')
      }
      const enabled =
        tools.enabled.length > 0
          ? tools.enabled.map((tool) => `  + ${tool}`).join('\n')
          : '  (none)'
      const disabled =
        tools.disabled.length > 0
          ? tools.disabled.map((tool) => `  - ${tool}`).join('\n')
          : '  (none)'
      return [`${name} tools`, 'Enabled:', enabled, 'Disabled:', disabled].join('\n')
    }
    case 'prompts': {
      const subcommand = args[1]?.toLowerCase() ?? 'list'
      if (subcommand === 'list' || subcommand === 'ls') {
        const server = args[2]
        const servers = server ? [server] : (await client.mcpServers()).map((entry) => entry.name)
        const groups = await Promise.all(
          servers.map(async (name) => ({
            name,
            prompts: await client.mcpPrompts(name).catch(() => []),
          })),
        )
        const lines = groups.flatMap((group) =>
          group.prompts.map((prompt) => {
            const argsLabel = prompt.arguments?.length
              ? ` args=${prompt.arguments.map((arg) => `${arg.name}${arg.required ? '*' : ''}`).join(',')}`
              : ''
            return `${group.name}/${prompt.name}${argsLabel}\n  ${prompt.description ?? ''}`.trimEnd()
          }),
        )
        return lines.length ? lines.join('\n') : 'No MCP prompts.'
      }
      if (subcommand === 'get') {
        const ref = args[2]
        if (!ref?.includes('/')) return 'Usage: /mcp prompts get <server/prompt> [--arg KEY=VALUE]'
        const [server, prompt] = ref.split('/', 2) as [string, string]
        const argEntries: string[] = []
        for (let index = 3; index < args.length; ) {
          const arg = args[index]!
          if (arg === '--arg' || arg.startsWith('--arg=')) {
            const parsed = parseValueFlag(args, index, '--arg')
            if (parsed.value) argEntries.push(parsed.value)
            index = parsed.nextIndex
            continue
          }
          return `Unknown /mcp prompts get flag: ${arg}`
        }
        return formatJson(
          await client.getMcpPrompt(server, prompt, parseKeyValueEntries(argEntries, '--arg')),
        )
      }
      return 'Usage: /mcp prompts <list|get>'
    }
    case 'resources': {
      const subcommand = args[1]?.toLowerCase() ?? 'list'
      if (subcommand === 'list' || subcommand === 'ls') {
        const server = args[2]
        const servers = server ? [server] : (await client.mcpServers()).map((entry) => entry.name)
        const groups = await Promise.all(
          servers.map(async (name) => ({
            name,
            resources: await client.mcpResources(name).catch(() => []),
          })),
        )
        const lines = groups.flatMap((group) =>
          group.resources.map((resource) =>
            `${group.name} ${resource.uri}${resource.mimeType ? ` ${resource.mimeType}` : ''}\n  ${resource.title ?? resource.name ?? resource.description ?? ''}`.trimEnd(),
          ),
        )
        return lines.length ? lines.join('\n') : 'No MCP resources.'
      }
      if (subcommand === 'templates') {
        const server = args[2]
        const servers = server ? [server] : (await client.mcpServers()).map((entry) => entry.name)
        const groups = await Promise.all(
          servers.map(async (name) => ({
            name,
            templates: await client.mcpResourceTemplates(name).catch(() => []),
          })),
        )
        const lines = groups.flatMap((group) =>
          group.templates.map((template) =>
            `${group.name} ${template.uriTemplate}\n  ${template.title ?? template.name ?? template.description ?? ''}`.trimEnd(),
          ),
        )
        return lines.length ? lines.join('\n') : 'No MCP resource templates.'
      }
      if (subcommand === 'read' || subcommand === 'subscribe' || subcommand === 'unsubscribe') {
        const server = args[2]
        const uri = args[3]
        if (!server || !uri) return `Usage: /mcp resources ${subcommand} <server> <uri>`
        if (subcommand === 'read') return formatJson(await client.mcpReadResource(server, uri))
        if (subcommand === 'subscribe')
          return formatJson(await client.mcpSubscribeResource(server, uri))
        return formatJson(await client.mcpUnsubscribeResource(server, uri))
      }
      if (subcommand === 'subscriptions') {
        const server = args[2]
        if (!server) return 'Usage: /mcp resources subscriptions <server>'
        const subscriptions = await client.mcpResourceSubscriptions(server)
        return subscriptions.length
          ? subscriptions.join('\n')
          : `No MCP resource subscriptions for ${server}.`
      }
      if (subcommand === 'updates') {
        const server = args[2]
        if (!server) return 'Usage: /mcp resources updates <server> [--limit n]'
        let limit: number | undefined
        for (let index = 3; index < args.length; ) {
          const arg = args[index]!
          if (arg === '--limit' || arg.startsWith('--limit=')) {
            const parsed = parseValueFlag(args, index, '--limit')
            limit = parsePositiveInt(parsed.value ?? undefined)
            index = parsed.nextIndex
            continue
          }
          return `Unknown /mcp resources updates flag: ${arg}`
        }
        const updates = await client.mcpResourceUpdates(server, limit)
        return updates.length
          ? updates.map(formatJson).join('\n')
          : `No MCP resource updates for ${server}.`
      }
      return 'Usage: /mcp resources <list|templates|read|subscribe|unsubscribe|subscriptions|updates>'
    }
    case 'complete': {
      const server = args[1]
      if (!server)
        return 'Usage: /mcp complete <server> --prompt <name>|--resource <uri> --arg KEY=VALUE'
      let prompt: string | undefined
      let resource: string | undefined
      let argEntry: string | undefined
      const contextEntries: string[] = []
      for (let index = 2; index < args.length; ) {
        const arg = args[index]!
        if (arg === '--prompt' || arg.startsWith('--prompt=')) {
          const parsed = parseValueFlag(args, index, '--prompt')
          prompt = parsed.value ?? undefined
          index = parsed.nextIndex
          continue
        }
        if (arg === '--resource' || arg.startsWith('--resource=')) {
          const parsed = parseValueFlag(args, index, '--resource')
          resource = parsed.value ?? undefined
          index = parsed.nextIndex
          continue
        }
        if (arg === '--arg' || arg.startsWith('--arg=')) {
          const parsed = parseValueFlag(args, index, '--arg')
          argEntry = parsed.value ?? undefined
          index = parsed.nextIndex
          continue
        }
        if (arg === '--context' || arg.startsWith('--context=')) {
          const parsed = parseValueFlag(args, index, '--context')
          if (parsed.value) contextEntries.push(parsed.value)
          index = parsed.nextIndex
          continue
        }
        return `Unknown /mcp complete flag: ${arg}`
      }
      if (!argEntry)
        return 'Usage: /mcp complete <server> --prompt <name>|--resource <uri> --arg KEY=VALUE'
      const [argumentName, argumentValue] = argEntry.split('=', 2)
      if (!argumentName || argumentValue == null)
        return 'Usage: /mcp complete <server> --prompt <name>|--resource <uri> --arg KEY=VALUE'
      const ref = prompt
        ? { type: 'ref/prompt' as const, name: prompt }
        : resource
          ? { type: 'ref/resource' as const, uri: resource }
          : null
      if (!ref)
        return 'Usage: /mcp complete <server> --prompt <name>|--resource <uri> --arg KEY=VALUE'
      return formatJson(
        await client.mcpComplete(server, {
          ref,
          argument: { name: argumentName, value: argumentValue },
          context: { arguments: parseKeyValueEntries(contextEntries, '--context') },
        }),
      )
    }
    case 'logging': {
      const subcommand = args[1]?.toLowerCase()
      if (subcommand === 'set-level') {
        const server = args[2]
        const level = parseLoggingLevel(args[3])
        if (!server || !level) return 'Usage: /mcp logging set-level <server> <level>'
        await client.mcpSetLoggingLevel(server, level)
        return `MCP logging level set: ${server} ${level}`
      }
      if (subcommand === 'logs') {
        const server = args[2]
        if (!server) return 'Usage: /mcp logging logs <server>'
        return formatJson(await client.mcpLogs(server))
      }
      return 'Usage: /mcp logging <set-level|logs>'
    }
    case 'metrics': {
      const server = args[1]
      return formatJson(
        server ? await client.mcpMetricsForServer(server) : await client.mcpMetrics(),
      )
    }
    case 'call': {
      if (!args[1] || !args[2]) return 'Usage: /mcp call <server> <tool> [--input JSON]'
      if (!args.includes('--yes')) {
        return "'/mcp call' executes a tool on the server. Re-run with --yes to confirm."
      }
      const effectiveArgs = args.filter((token) => token !== '--yes')
      const server = effectiveArgs[1]
      const tool = effectiveArgs[2]
      if (!server || !tool) return 'Usage: /mcp call <server> <tool> [--input JSON]'
      let input: Record<string, unknown> = {}
      for (let index = 3; index < effectiveArgs.length; ) {
        const arg = effectiveArgs[index]!
        if (arg === '--input' || arg.startsWith('--input=')) {
          const parsed = parseValueFlag(effectiveArgs, index, '--input')
          const parsedInput = parseJsonObject(parsed.value ?? '', '--input')
          if (typeof parsedInput === 'string') return parsedInput
          input = parsedInput
          index = parsed.nextIndex
          continue
        }
        return `Unknown /mcp call flag: ${arg}`
      }
      return formatJson(await client.mcpCallTool(server, tool, input))
    }
    case 'marketplace':
    case 'marketplaces': {
      const subcommand = args[1]?.toLowerCase() ?? 'list'
      if (subcommand === 'list' || subcommand === 'ls') {
        const marketplaces = await client.mcpMarketplaceList()
        return marketplaces.length > 0
          ? marketplaces
              .map((entry) => {
                const sync = entry.lastSync ? `synced:${entry.lastSync}` : 'never-synced'
                return `${entry.name} · ${entry.url} · ${sync}`
              })
              .join('\n')
          : 'No MCP marketplaces registered.'
      }
      if (subcommand === 'add') {
        const name = args[2]
        const url = args[3]
        if (!name || !url) return 'Usage: /mcp marketplace add <name> <url>'
        const marketplace = await client.mcpMarketplaceAdd(name, url)
        return `MCP marketplace added: ${marketplace.name} · ${marketplace.url}`
      }
      if (subcommand === 'remove' || subcommand === 'rm') {
        const name = args[2]
        if (!name) return 'Usage: /mcp marketplace remove <name>'
        const result = await client.mcpMarketplaceRemove(name)
        return result.removed
          ? `MCP marketplace removed: ${name}`
          : `MCP marketplace not found: ${name}`
      }
      return 'Usage: /mcp marketplace <list|add|remove>'
    }
    default:
      return help()
  }
}
