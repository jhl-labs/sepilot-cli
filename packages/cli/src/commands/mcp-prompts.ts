import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'

function parseArgs(entries: string[] | undefined): Record<string, string> {
  const args: Record<string, string> = {}
  for (const e of entries ?? []) {
    const eq = e.indexOf('=')
    if (eq <= 0) throw new Error(`Invalid --arg entry: ${e}. Expected key=value.`)
    args[e.slice(0, eq)] = e.slice(eq + 1)
  }
  return args
}

export async function mcpPromptsListCommand(
  server: string | undefined,
  options: { url?: string },
) {
  const client = new DaemonClient(options.url)
  if (server) {
    const prompts = await client.mcpPrompts(server)
    if (!prompts?.length) {
      console.log(`No prompts on ${server}.`)
      return
    }
    for (const p of prompts) console.log(`  ${p.name.padEnd(30)} ${p.description}`)
    return
  }
  const servers = await client.mcpServers()
  for (const s of servers) {
    if (s.status !== 'connected') continue
    const prompts = await client.mcpPrompts(s.name).catch(() => [])
    if (!prompts.length) continue
    console.log(chalk.cyan(s.name))
    for (const p of prompts) console.log(`  ${p.name.padEnd(30)} ${p.description}`)
  }
}

export async function mcpPromptsGetCommand(
  ref: string,
  options: { url?: string; arg?: string[] },
) {
  const [server, prompt] = ref.includes('/') ? ref.split('/', 2) : [null, ref]
  if (!server) {
    console.error(chalk.red('Use <server>/<prompt> form.'))
    process.exit(1)
  }
  const client = new DaemonClient(options.url)
  const res = await client.getMcpPrompt(server, prompt, parseArgs(options.arg))
  for (const msg of res.messages) {
    const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
    console.log(`[${msg.role}] ${content}`)
  }
}
