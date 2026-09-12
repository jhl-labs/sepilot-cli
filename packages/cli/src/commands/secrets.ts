import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

export interface SecretsSetCommandOptions {
  url?: string
  stdin?: boolean
}

export interface SecretsCommandOptions {
  url?: string
}

export async function secretsSetCommand(
  key: string,
  value: string | undefined,
  options: SecretsSetCommandOptions,
) {
  const trimmedKey = key?.trim() ?? ''
  if (trimmedKey.length === 0) {
    console.error(chalk.red('Secret key cannot be empty'))
    process.exit(1)
  }
  if (!/^[A-Za-z_][A-Za-z0-9_-]*$/.test(trimmedKey)) {
    console.error(chalk.red(
      `Invalid secret key: ${trimmedKey} (use letters, digits, '_', '-' only; must start with a letter or '_')`,
    ))
    process.exit(1)
  }
  const client = new DaemonClient(options.url)
  let val = value
  if (options.stdin || !val) {
    const chunks: Buffer[] = []
    for await (const chunk of process.stdin) chunks.push(chunk as Buffer)
    val = Buffer.concat(chunks).toString('utf-8').trim()
  }
  if (!val) {
    console.error(chalk.red('No value provided'))
    process.exit(1)
  }
  await client.setSecret(trimmedKey, val)
  output({ ok: true, key: trimmedKey, action: 'set' }, () =>
    chalk.green(`Secret set: ${chalk.bold(trimmedKey)}`),
  )
}

export async function secretsListCommand(options: SecretsCommandOptions) {
  const client = new DaemonClient(options.url)
  const res = await client.listSecrets()
  output(res ?? { keys: [] }, (data) => {
    if (!data.keys?.length) return 'No secrets stored.'
    return data.keys.map((k) => `  ${k}`).join('\n')
  })
}

export async function secretsRemoveCommand(
  key: string,
  options: SecretsCommandOptions,
) {
  const client = new DaemonClient(options.url)
  const res = await client.removeSecret(key)
  output(
    { ok: res.removed, key, removed: res.removed },
    (data) => (data.removed
      ? chalk.green(`Removed secret: ${chalk.bold(data.key)}`)
      : chalk.yellow(`Secret not found: ${data.key}`)),
  )
}
