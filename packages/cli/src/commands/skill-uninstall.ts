import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output, outputError } from '../output/formatter.js'
import { friendlyErrorMessage as errorMessage } from '../utils/error-message.js'
import { detectCliLocale } from '../utils/locale.js'

const SKILL_UNINSTALL_COPY = {
  en: {
    removedSkill: (name: string) => `Removed skill: ${name}`,
    skillNotFound: (name: string) => `Skill not found: ${name}`,
    failedUninstall: (name: string, msg: string) => `Failed to uninstall ${name}: ${msg}`,
  },
  ko: {
    removedSkill: (name: string) => `스킬 제거됨: ${name}`,
    skillNotFound: (name: string) => `스킬을 찾을 수 없습니다: ${name}`,
    failedUninstall: (name: string, msg: string) => `${name} 제거 실패: ${msg}`,
  },
} as const

export async function skillUninstallCommand(name: string, options: { url?: string }) {
  const copy = SKILL_UNINSTALL_COPY[detectCliLocale()] ?? SKILL_UNINSTALL_COPY.en
  const client = new DaemonClient(options.url)
  try {
    const res = await client.uninstallSkill(name)
    if (res.removed) {
      output({ ok: true, name, removed: true }, () =>
        chalk.green(copy.removedSkill(chalk.bold(name))),
      )
      return
    }
    outputError({ ok: false, name, error: 'skill-not-found' }, () =>
      chalk.yellow(copy.skillNotFound(name)),
    )
    process.exit(1)
  } catch (err) {
    outputError(
      { ok: false, name, error: errorMessage(err) },
      () => chalk.red(copy.failedUninstall(name, errorMessage(err))),
    )
    process.exit(1)
  }
}
