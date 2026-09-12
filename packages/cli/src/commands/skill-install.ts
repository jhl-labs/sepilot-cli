import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { friendlyErrorMessage as errorMessage } from '../utils/error-message.js'
import { detectCliLocale } from '../utils/locale.js'

const SKILL_INSTALL_COPY = {
  en: {
    noSkillsInstalled: 'No skills installed.',
    installedSkill: (name: string, version: string) => `Installed skill: ${name} v${version}`,
    errorPrefix: (e: string) => `error: ${e}`,
    warningPrefix: (w: string) => `warning: ${w}`,
    useForceHint: 'Use --force to install anyway.',
    installFailed: (msg: string) => `install failed: ${msg}`,
  },
  ko: {
    noSkillsInstalled: '설치된 스킬이 없습니다.',
    installedSkill: (name: string, version: string) => `스킬 설치됨: ${name} v${version}`,
    errorPrefix: (e: string) => `오류: ${e}`,
    warningPrefix: (w: string) => `경고: ${w}`,
    useForceHint: '그래도 설치하려면 --force를 사용하세요.',
    installFailed: (msg: string) => `설치 실패: ${msg}`,
  },
} as const

interface SkillInstallValidationError {
  response?: {
    data?: {
      error?: {
        code?: string
        validation?: {
          errors?: string[]
          warnings?: string[]
        }
      }
    }
  }
}

export async function skillInstallCommand(
  source: string,
  options: { url?: string; force?: boolean },
) {
  const copy = SKILL_INSTALL_COPY[detectCliLocale()] ?? SKILL_INSTALL_COPY.en
  const client = new DaemonClient(options.url)
  try {
    const preview = await client.previewSkillInstall({ source })
    const res = await client.installSkill({
      source,
      force: options.force === true,
      expectedDigest: preview.digest,
    })
    output(
      { ok: true, source, installed: res.installed ?? [] },
      (data) => {
        if (!data.installed.length) return chalk.yellow(copy.noSkillsInstalled)
        return data.installed
          .map((s) => chalk.green(copy.installedSkill(chalk.bold(s.name), s.version)))
          .join('\n')
      },
    )
  } catch (err) {
    const body = (err as SkillInstallValidationError)?.response?.data
    if (body?.error?.code === 'VALIDATION_FAILED' && body.error.validation) {
      const v = body.error.validation
      for (const e of v.errors ?? []) console.error(chalk.red(copy.errorPrefix(e)))
      for (const w of v.warnings ?? []) console.error(chalk.yellow(copy.warningPrefix(w)))
      console.error(chalk.gray(copy.useForceHint))
    } else {
      console.error(chalk.red(copy.installFailed(errorMessage(err))))
    }
    process.exit(1)
  }
}
