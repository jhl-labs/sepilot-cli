import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import chalk from 'chalk'
import { resolveDaemonInvocation } from '@sepilotd/api-client/node'
import { resolveStandaloneDaemonInvocation } from '../client/standalone-daemon.js'
import { maybeRunInitWizard } from '../utils/init-wizard.js'
import { detectCliLocale } from '../utils/locale.js'

const execAsync = promisify(execFile)

const INIT_COPY = {
  en: {
    daemonMissing: 'Cannot find sepilotd binary. Build or install @sepilotd/daemon first.',
  },
  ko: {
    daemonMissing: 'sepilotd 바이너리를 찾을 수 없습니다. 먼저 @sepilotd/daemon을 빌드하거나 설치하세요.',
  },
} as const

export async function initCommand(options: { name?: string; role?: string; wizard?: boolean }) {
  const copy = INIT_COPY[detectCliLocale()] ?? INIT_COPY.en
  const daemon = resolveStandaloneDaemonInvocation()
    ?? resolveDaemonInvocation({ moduleSearchRoots: [import.meta.dirname] })
  if (!daemon) {
    console.error(chalk.red(copy.daemonMissing))
    process.exit(1)
  }

  const args = ['init']
  if (options.name) args.push(options.name)
  if (options.role) args.push(options.role)

  try {
    const { stdout, stderr } = await execAsync(daemon.command, [...daemon.args, ...args])
    if (stdout) process.stdout.write(stdout)
    if (stderr) process.stderr.write(stderr)
  } catch (err) {
    const execErr = err as { stdout?: Buffer | string; stderr?: Buffer | string }
    if (execErr.stdout) process.stdout.write(execErr.stdout)
    if (execErr.stderr) process.stderr.write(execErr.stderr)
    throw err
  }

  await maybeRunInitWizard({
    enabled: options.wizard !== false,
  })
}
