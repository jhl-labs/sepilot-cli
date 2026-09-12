import chalk from 'chalk'
import type { DaemonPolicyRule, DaemonToolPolicyMode } from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'

function modeColor(mode: DaemonToolPolicyMode): (text: string) => string {
  switch (mode) {
    case 'blocked': return chalk.red
    case 'supervised': return chalk.yellow
    case 'autonomous': return chalk.green
    default: return chalk.gray
  }
}

function ruleSummary(rule: DaemonPolicyRule): string {
  const parts: string[] = []
  if (rule.deny_patterns?.length) parts.push(`${rule.deny_patterns.length} deny_patterns`)
  if (rule.deny_paths?.length) parts.push(`${rule.deny_paths.length} deny_paths`)
  if (rule.deny_urls?.length) parts.push(`${rule.deny_urls.length} deny_urls`)
  if (rule.deny_executables?.length) parts.push(`${rule.deny_executables.length} deny_executables`)
  if (rule.allow_patterns?.length) parts.push(`${rule.allow_patterns.length} allow_patterns`)
  if (rule.max_timeout_ms) parts.push(`timeout=${rule.max_timeout_ms}ms`)
  return parts.join(', ') || 'no extra rules'
}

/**
 * Print the active tool policy in a single readable table. Mirrors the
 * chat-shell `/policy` command but is reachable from a one-shot cli
 * invocation, which means automation, audits, and `--json` consumers can
 * inspect the live rule set without entering the chat shell.
 */
export async function policyCommand(options: { url?: string }) {
  const client = new DaemonClient(options.url)
  const data = await client.policy()
  output(data, (d) => {
    const tools = Object.entries(d.tools).sort(([a], [b]) => a.localeCompare(b))
    const lines = [
      `Defaults: mode=${d.defaults.mode}, unmatched=${d.defaults.unmatched_policy}, timeout=${d.defaults.max_timeout_ms}ms`,
      '',
      'Tool rules:',
    ]
    if (tools.length === 0) {
      lines.push('  (no tool-specific rules)')
    } else {
      for (const [name, rule] of tools) {
        const colour = modeColor(rule.mode)
        lines.push(
          `  ${colour(rule.mode.padEnd(11))} ${name.padEnd(22)} ${chalk.gray(ruleSummary(rule))}`,
        )
      }
    }
    return lines.join('\n')
  })
}
