import { execSync } from 'node:child_process'
import { mkdir, readFile, writeFile, unlink } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'
import chalk from 'chalk'
import { resolveDaemonInvocation } from '@sepilotd/api-client/node'
import { detectCliLocale } from '../utils/locale.js'
import { resolveStandaloneDaemonInvocation } from '../client/standalone-daemon.js'

const DAEMON_SERVICE_COPY = {
  en: {
    daemonBinNotFound: 'Cannot find a sepilotd daemon binary. Build or install @sepilotd/daemon, or set SEPILOTD_BIN.',
    systemctlFailed: (args: string) => `systemctl --user ${args} failed`,
    installLinuxOnly: 'install-service is implemented for Linux/systemd only. On Windows, enable "Start at login" from the desktop tray menu.',
    wrotePrefix: (path: string) => `Wrote ${path}`,
    skipEnableHint: (unit: string) => `Skipping enable. Run \`systemctl --user daemon-reload && systemctl --user enable --now ${unit}\` when ready.`,
    willStartOnLogin: (unit: string) => `sepilotd will now start on login (systemctl --user status ${unit} to inspect).`,
    finishManuallyHint: (unit: string) => 'You can finish manually:\n' + `  systemctl --user daemon-reload\n` + `  systemctl --user enable --now ${unit}`,
    uninstallLinuxOnly: 'uninstall-service is implemented for Linux/systemd only. On Windows, untick "Start at login" from the desktop tray menu.',
    removedPrefix: (path: string) => `Removed ${path}`,
    nothingToRemove: 'No systemd unit was installed; nothing to remove.',
    statusLinuxOnly: 'service status is implemented for Linux/systemd only.',
    unitFilePrefix: (path: string) => `Unit file: ${path}`,
    noUnitInstalled: 'No systemd unit installed. Run `sepilot daemon install-service`.',
  },
  ko: {
    daemonBinNotFound: 'sepilotd daemon 바이너리를 찾을 수 없습니다. @sepilotd/daemon을 빌드하거나 설치하거나 SEPILOTD_BIN을 설정하세요.',
    systemctlFailed: (args: string) => `systemctl --user ${args} 실패`,
    installLinuxOnly: 'install-service는 Linux/systemd 전용입니다. Windows에서는 데스크톱 트레이 메뉴에서 "로그인 시 시작"을 활성화하세요.',
    wrotePrefix: (path: string) => `${path} 작성됨`,
    skipEnableHint: (unit: string) => `활성화 건너뜀. 준비되면 \`systemctl --user daemon-reload && systemctl --user enable --now ${unit}\`을 실행하세요.`,
    willStartOnLogin: (unit: string) => `이제 sepilotd가 로그인 시 시작됩니다 (확인: systemctl --user status ${unit}).`,
    finishManuallyHint: (unit: string) => '수동으로 마무리할 수 있습니다:\n' + `  systemctl --user daemon-reload\n` + `  systemctl --user enable --now ${unit}`,
    uninstallLinuxOnly: 'uninstall-service는 Linux/systemd 전용입니다. Windows에서는 데스크톱 트레이 메뉴에서 "로그인 시 시작"을 해제하세요.',
    removedPrefix: (path: string) => `${path} 제거됨`,
    nothingToRemove: '설치된 systemd 유닛이 없습니다. 제거할 것이 없습니다.',
    statusLinuxOnly: 'service status는 Linux/systemd 전용입니다.',
    unitFilePrefix: (path: string) => `유닛 파일: ${path}`,
    noUnitInstalled: '설치된 systemd 유닛이 없습니다. `sepilot daemon install-service`를 실행하세요.',
  },
} as const

function daemonServiceCopy() {
  return DAEMON_SERVICE_COPY[detectCliLocale()] ?? DAEMON_SERVICE_COPY.en
}

function quoteSystemdEnvironment(name: string, value: string): string {
  const escaped = value.replace(/\\/g, '\\\\').replace(/"/g, '\\"')
  return `Environment="${name}=${escaped}"`
}

const SYSTEMD_USER_DIR = join(homedir(), '.config', 'systemd', 'user')
const SYSTEMD_UNIT_NAME = 'sepilotd.service'
const SYSTEMD_UNIT_PATH = join(SYSTEMD_USER_DIR, SYSTEMD_UNIT_NAME)

export interface DaemonServiceOptions {
  /** Skip running `systemctl` after writing the unit file. */
  noEnable?: boolean
}

function buildUnitFile(execLine: string, hostToolchainPath = process.env.PATH ?? ''): string {
  const toolchainEnvironment = hostToolchainPath
    ? `${quoteSystemdEnvironment('SEPILOTD_HOST_TOOLCHAIN_PATH', hostToolchainPath)}\n`
    : ''
  return `[Unit]
Description=sepilotd — local AI agent daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=${execLine}
${toolchainEnvironment}Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
`
}

function quoteShellArg(value: string): string {
  if (/^[A-Za-z0-9._\-/=:@+]+$/.test(value)) return value
  return `'${value.replace(/'/g, `'\\''`)}'`
}

function resolveExecLine(): string {
  // Standalone single-file build: there is no separate `sepilotd` entry, so the
  // unit re-enters this binary with `__daemon` (e.g. `ExecStart=/path/to/sepilot __daemon`).
  const standalone = resolveStandaloneDaemonInvocation()
  if (standalone) {
    return [standalone.command, ...standalone.args].map(quoteShellArg).join(' ')
  }
  const inv = resolveDaemonInvocation({
    moduleSearchRoots: [import.meta.dirname],
    execPath: process.execPath,
  })
  if (!inv) {
    throw new Error(daemonServiceCopy().daemonBinNotFound)
  }
  const parts = [inv.command, ...inv.args].map(quoteShellArg)
  return parts.join(' ')
}

function runSystemctl(args: string[]): void {
  try {
    execSync(['systemctl', '--user', ...args].join(' '), { stdio: 'inherit' })
  } catch {
    // Surface a clearer error than execSync's verbose dump — we re-throw so
    // the caller can decide whether the missing piece is fatal.
    throw new Error(daemonServiceCopy().systemctlFailed(args.join(' ')))
  }
}

export async function installServiceCommand(
  options: DaemonServiceOptions = {},
): Promise<void> {
  const copy = daemonServiceCopy()
  if (process.platform !== 'linux') {
    console.error(chalk.yellow(copy.installLinuxOnly))
    process.exit(2)
  }

  const execLine = resolveExecLine()
  const unit = buildUnitFile(execLine)

  await mkdir(dirname(SYSTEMD_UNIT_PATH), { recursive: true })
  await writeFile(SYSTEMD_UNIT_PATH, unit, 'utf-8')
  console.log(chalk.green(copy.wrotePrefix(SYSTEMD_UNIT_PATH)))

  if (options.noEnable) {
    console.log(chalk.gray(copy.skipEnableHint(SYSTEMD_UNIT_NAME)))
    return
  }

  try {
    runSystemctl(['daemon-reload'])
    runSystemctl(['enable', '--now', SYSTEMD_UNIT_NAME])
    console.log(chalk.green(copy.willStartOnLogin(SYSTEMD_UNIT_NAME)))
  } catch (err) {
    console.error(chalk.red(err instanceof Error ? err.message : String(err)))
    console.error(chalk.gray(copy.finishManuallyHint(SYSTEMD_UNIT_NAME)))
    process.exit(1)
  }
}

export async function uninstallServiceCommand(): Promise<void> {
  const copy = daemonServiceCopy()
  if (process.platform !== 'linux') {
    console.error(chalk.yellow(copy.uninstallLinuxOnly))
    process.exit(2)
  }

  try {
    runSystemctl(['disable', '--now', SYSTEMD_UNIT_NAME])
  } catch {
    /* unit may not be enabled; continue */
  }

  try {
    await unlink(SYSTEMD_UNIT_PATH)
    console.log(chalk.green(copy.removedPrefix(SYSTEMD_UNIT_PATH)))
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err
    console.log(chalk.gray(copy.nothingToRemove))
  }

  try {
    runSystemctl(['daemon-reload'])
  } catch {
    /* best-effort */
  }
}

export async function showServiceStatus(): Promise<void> {
  const copy = daemonServiceCopy()
  if (process.platform !== 'linux') {
    console.error(chalk.yellow(copy.statusLinuxOnly))
    process.exit(2)
  }

  try {
    const unit = await readFile(SYSTEMD_UNIT_PATH, 'utf-8')
    console.log(chalk.gray(copy.unitFilePrefix(SYSTEMD_UNIT_PATH)))
    console.log(unit)
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
      console.log(chalk.gray(copy.noUnitInstalled))
      return
    }
    throw err
  }
  try {
    runSystemctl(['status', SYSTEMD_UNIT_NAME, '--no-pager'])
  } catch {
    /* status returns non-zero when not running; that's fine */
  }
}
