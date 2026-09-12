import chalk from 'chalk'
import type {
  DaemonExtensionTokenScope,
  DaemonExtensionTokenSummary,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export interface TokenCommandOptions {
  url?: string
}

export interface IssueTokenCommandOptions extends TokenCommandOptions {
  scope?: string[]
  expiresAt?: string
}

const ALLOWED_SCOPES: DaemonExtensionTokenScope[] = [
  'all',
  'inspect',
  'chat',
  'ws',
  'sessions',
  'memory',
  'files',
  'skills',
  'projects',
  'approvals',
  'extensions',
  'personas',
  'artifacts',
]

function parseScopes(
  scopes: string[] | undefined,
): DaemonExtensionTokenScope[] {
  const normalized = (scopes ?? [])
    .map((scope) => scope.trim())
    .filter(Boolean)

  if (normalized.length === 0) {
    throw new Error(
      `At least one --scope is required. Allowed scopes: ${ALLOWED_SCOPES.join(', ')}`,
    )
  }

  for (const scope of normalized) {
    if (!ALLOWED_SCOPES.includes(scope as DaemonExtensionTokenScope)) {
      throw new Error(
        `Unsupported scope: ${scope}. Allowed scopes: ${ALLOWED_SCOPES.join(', ')}`,
      )
    }
  }

  return normalized as DaemonExtensionTokenScope[]
}

function formatTokenSummary(summary: DaemonExtensionTokenSummary): string {
  const status = summary.active
    ? chalk.green('active')
    : chalk.gray(summary.revokedAt ? 'revoked' : 'inactive')
  const scopes = summary.scopes.join(',')
  const expires = summary.expiresAt ? ` expires=${summary.expiresAt}` : ''
  // Surface the id so `tokens revoke <id>` is discoverable from the list.
  return `  ${chalk.bold(summary.id.padEnd(12))} ${summary.label.padEnd(20)} ${status.padEnd(16)} scopes=${scopes}${expires}`
}

export async function tokenListCommand(options: TokenCommandOptions) {
  const client = new DaemonClient(options.url)
  const tokens = await client.extensionTokens()
  output(tokens, (records) => {
    if (!records.length) {
      return 'No daemon-issued extension tokens.\nUse `sepilot tokens issue <label> --scope <scope>` to create one.'
    }
    return records.map(formatTokenSummary).join('\n')
  })
}

export async function tokenIssueCommand(
  label: string,
  options: IssueTokenCommandOptions,
) {
  const client = new DaemonClient(options.url)
  if (options.expiresAt && Number.isNaN(Date.parse(options.expiresAt))) {
    throw new Error(
      `Invalid --expires-at value: ${options.expiresAt} (expected RFC3339 timestamp like 2027-01-01T00:00:00Z)`,
    )
  }
  const issued = await client.issueExtensionToken({
    label,
    scopes: parseScopes(options.scope),
    expiresAt: options.expiresAt,
  })

  output({ ok: true, ...issued }, (record) => [
    `Extension token issued: ${record.label}`,
    `Token: ${chalk.yellow(record.token)}`,
    `Scopes: ${record.scopes.join(', ')}`,
    ...(record.expiresAt ? [`Expires: ${record.expiresAt}`] : []),
    'Store this token now; it will not be shown again.',
  ].join('\n'))
}

export async function tokenRevokeCommand(
  id: string,
  options: TokenCommandOptions,
) {
  const client = new DaemonClient(options.url)
  const revoked = await client.revokeExtensionToken(id)
  output({ ok: true, ...(revoked ?? { id, label: id }) }, (record) => {
    const label = record?.label ?? id
    const recordId = record?.id ?? id
    return `Extension token revoked: ${label} (${recordId})`
  })
}
