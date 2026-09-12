import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { loadCliVersion } from '../entrypoint.js'
import { getOutputFormat, output } from '../output/formatter.js'

interface VersionPayload {
  cli: string
  daemon?: string
  daemonReachable: boolean
  latest: string | null
  latestUrl: string | null
  updateAvailable: boolean | null
}

export async function versionCommand(options: { url?: string; check?: boolean }) {
  const cliVersion = loadCliVersion()
  const payload: VersionPayload = {
    cli: cliVersion,
    daemonReachable: false,
    latest: null,
    latestUrl: null,
    updateAvailable: null,
  }

  try {
    const client = new DaemonClient(options.url)
    const health = await client.health()
    payload.daemon = health.version
    payload.daemonReachable = true
  } catch {
    payload.daemonReachable = false
  }

  if (options.check) {
    try {
      const res = await fetch('https://api.github.com/repos/jhl-labs/sepilot-cli/releases/latest', {
        headers: { 'User-Agent': 'sepilot-cli' },
      })
      if (res.ok) {
        const data = await res.json() as { tag_name: string; html_url: string }
        payload.latest = data.tag_name.replace(/^v/, '')
        payload.latestUrl = data.html_url
        payload.updateAvailable = payload.latest !== cliVersion
      }
    } catch {
      // best-effort; silent failure mirrors prior behaviour.
    }
  }

  if (getOutputFormat() === 'json') {
    output(payload)
    return
  }

  output(payload, (data) => {
    const lines: string[] = [`sepilot CLI v${data.cli}`]
    if (data.daemonReachable && data.daemon) {
      lines.push(`sepilotd daemon v${data.daemon}`)
    } else {
      lines.push(chalk.gray('Daemon: not running'))
    }
    if (options.check) {
      lines.push(chalk.gray('\nChecking for updates...'))
      if (data.latest === undefined) {
        lines.push(chalk.gray('Could not check for updates.'))
      } else if (data.updateAvailable) {
        lines.push(chalk.yellow(`\nUpdate available: v${data.latest}`))
        if (data.latestUrl) lines.push(chalk.gray(`  ${data.latestUrl}`))
        lines.push(chalk.gray('  Run: curl -fsSL https://raw.githubusercontent.com/jhl-labs/sepilot-cli/main/packages/bundle/scripts/install.sh | sh'))
      } else {
        lines.push(chalk.green('You are on the latest version.'))
      }
    }
    return lines.join('\n')
  })
}
