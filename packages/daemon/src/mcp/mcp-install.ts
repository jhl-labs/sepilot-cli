import type { McpServerTemplate } from './marketplace-source.js'
import type { ConfigWriter } from './config-writer.js'
import { buildMcpServerProvenance } from './security.js'

export interface InstallMcpServerOptions {
  template: McpServerTemplate
  configPath: string
  configWriter: ConfigWriter
  variables?: Record<string, string>
  allowUnverified?: boolean
}

function substitutePlaceholders(
  value: string,
  variables: Record<string, string>,
): string {
  // Replace {workdir} with cwd
  let result = value.replace(/\{workdir\}/g, process.cwd())

  // Replace {env.VAR} with process.env.VAR (leave as-is if undefined)
  result = result.replace(/\{env\.([^}]+)\}/g, (_match, varName: string) => {
    const envValue = process.env[varName]
    return envValue !== undefined ? envValue : `{env.${varName}}`
  })

  result = result.replace(/\{input\.([^}]+)\}/g, (_match, varName: string) => {
    if (Object.hasOwn(variables, varName)) {
      return variables[varName] ?? ''
    }
    throw new Error(`Missing MCP install variable: ${varName}`)
  })

  return result
}

function substituteArray(
  arr: string[],
  variables: Record<string, string>,
): string[] {
  return arr.map((value) => substitutePlaceholders(value, variables))
}

function substituteRecord(
  record: Record<string, string>,
  variables: Record<string, string>,
): Record<string, string> {
  const out: Record<string, string> = {}
  for (const [key, value] of Object.entries(record)) {
    out[key] = substitutePlaceholders(value, variables)
  }
  return out
}

export async function installMcpServer(
  opts: InstallMcpServerOptions,
): Promise<void> {
  const { template, configWriter } = opts
  const variables = opts.variables ?? {}

  for (const variable of template.variables ?? []) {
    if (variable.required && !variables[variable.name]?.trim()) {
      throw new Error(`Missing MCP install variable: ${variable.name}`)
    }
  }

  const entry: Record<string, unknown> = {
    name: template.name,
    enabled: true,
    transport: template.transport,
    disabledTools: [],
    provenance: buildMcpServerProvenance({
      template,
      marketplace: {
        name: template.marketplace,
        url: template.marketplaceUrl ?? template.repository ?? template.homepage ?? '',
        publisher: template.marketplacePublisher,
        publicKey: template.marketplacePublicKey,
        addedAt: '',
        lastSync: null,
      },
      allowUnverified: opts.allowUnverified,
    }),
  }

  if (template.transport === 'stdio') {
    entry.command = template.command
      ? substitutePlaceholders(template.command, variables)
      : ''
    entry.args = template.args ? substituteArray(template.args, variables) : []
    entry.env = template.env ? substituteRecord(template.env, variables) : {}
  } else {
    // sse or http
    entry.url = template.url ? substitutePlaceholders(template.url, variables) : ''
    entry.headers = template.headers
      ? substituteRecord(template.headers, variables)
      : {}
  }

  await configWriter.addMcpServer(entry)
}
