import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { friendlyErrorMessage } from '../utils/error-message.js'
import { detectCliLocale } from '../utils/locale.js'

const VALID_LEVELS = ['readonly', 'accept-edits', 'workspace-write', 'supervised', 'autonomous'] as const
type AutonomyLevel = (typeof VALID_LEVELS)[number]

const AUTONOMY_COPY = {
  en: {
    autonomyPrefix: 'Autonomy:',
    unknown: 'unknown',
    describeAutonomous: 'autonomous (allowed tools run unattended; approval-required tools are blocked)',
    describeAcceptEdits: 'accept-edits (auto-allows fs.write, supervises everything else)',
    describeWorkspaceWrite: 'workspace-write (auto-allows workspace file edits; supervises commands)',
    describeReadonly: 'readonly (no mutations; only read-only tools fire)',
    describeSupervised: 'supervised (every risky tool needs approval)',
    unknownLevel: (level: string, valid: string) => `Unknown autonomy level: ${level}. Use one of: ${valid}`,
    failedUpdate: (msg: string) => `Failed to update autonomy: ${msg}`,
    autonomousWarning: '  Warning: allowed tools run without prompts, but approval-required tools are blocked. Use `sepilot autonomy supervised` when you want approval prompts.',
  },
  ko: {
    autonomyPrefix: '자율성:',
    unknown: '알 수 없음',
    describeAutonomous: 'autonomous (허용된 도구는 무인 실행, 승인 필요 도구는 차단)',
    describeAcceptEdits: 'accept-edits (fs.write 자동 허용, 나머지는 모두 감독)',
    describeWorkspaceWrite: 'workspace-write (작업공간 파일 편집 자동 허용, 명령은 감독)',
    describeReadonly: 'readonly (변경 없음; 읽기 전용 도구만 실행)',
    describeSupervised: 'supervised (모든 위험한 도구에 승인 필요)',
    unknownLevel: (level: string, valid: string) => `알 수 없는 자율성 레벨: ${level}. 다음 중 하나를 사용하세요: ${valid}`,
    failedUpdate: (msg: string) => `자율성 업데이트 실패: ${msg}`,
    autonomousWarning: '  경고: 허용된 도구는 프롬프트 없이 실행되지만, 승인 필요 도구는 차단됩니다. 승인 프롬프트가 필요하면 `sepilot autonomy supervised`를 사용하세요.',
  },
} as const

function autonomyCopy() {
  return AUTONOMY_COPY[detectCliLocale()] ?? AUTONOMY_COPY.en
}

function describeLevel(level: AutonomyLevel): string {
  const copy = autonomyCopy()
  switch (level) {
    case 'autonomous':
      return copy.describeAutonomous
    case 'accept-edits':
      return copy.describeAcceptEdits
    case 'workspace-write':
      return copy.describeWorkspaceWrite
    case 'readonly':
      return copy.describeReadonly
    case 'supervised':
      return copy.describeSupervised
  }
}

function formatLevelLine(level: string | undefined): string {
  const copy = autonomyCopy()
  if (!level) return chalk.gray(`${copy.autonomyPrefix} ${copy.unknown}`)
  const known = (VALID_LEVELS as readonly string[]).includes(level)
    ? (level as AutonomyLevel)
    : null
  const body = known ? describeLevel(known) : level
  const colour = level === 'autonomous'
    ? chalk.red
    : level === 'workspace-write'
      ? chalk.magenta
    : level === 'accept-edits'
      ? chalk.yellow
      : level === 'readonly'
        ? chalk.cyan
        : level === 'supervised'
          ? chalk.green
          : chalk.gray
  return `${copy.autonomyPrefix} ${colour(body)}`
}

/**
 * Read-or-write the daemon's `agent.autonomy` config knob.
 *
 * `sepilot autonomy` (no arg)         → print the current level.
 * `sepilot autonomy <level>`          → set it via PUT /config and re-print.
 *
 * The chat shell already has a `/autonomy` slash command for the same
 * thing; this is the equivalent for one-shot cli usage / scripts.
 */
export async function autonomyCommand(
  level: string | undefined,
  options: { url?: string },
) {
  const client = new DaemonClient(options.url)

  if (!level) {
    const config = await client.config()
    output(
      { autonomy: config.agent?.autonomy },
      () => formatLevelLine(config.agent?.autonomy),
    )
    return
  }

  const copy = autonomyCopy()
  if (!(VALID_LEVELS as readonly string[]).includes(level)) {
    console.error(
      chalk.red(copy.unknownLevel(level, VALID_LEVELS.join(', '))),
    )
    process.exit(1)
  }

  try {
    await client.updateConfig({ 'agent.autonomy': level as AutonomyLevel })
  } catch (err) {
    console.error(chalk.red(copy.failedUpdate(friendlyErrorMessage(err))))
    process.exit(1)
  }

  output(
    { autonomy: level },
    () => {
      const lines = [formatLevelLine(level)]
      if (level === 'autonomous') {
        // Loud red warning so the user can't accidentally page-up the cli
        // history later and miss that they turned approvals off.
        lines.push(chalk.red(copy.autonomousWarning))
      }
      return lines.join('\n')
    },
  )
}
