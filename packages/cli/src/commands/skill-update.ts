import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { friendlyErrorMessage as errorMessage } from '../utils/error-message.js'
import { detectCliLocale } from '../utils/locale.js'

const SKILL_UPDATE_COPY = {
  en: {
    provideNameOrAll: 'provide a skill name or --all',
    upToDate: 'up to date',
  },
  ko: {
    provideNameOrAll: '스킬 이름 또는 --all을 제공하세요',
    upToDate: '최신 상태',
  },
} as const

export async function skillUpdateCommand(
  name: string | undefined,
  options: { url?: string; all?: boolean },
) {
  const copy = SKILL_UPDATE_COPY[detectCliLocale()] ?? SKILL_UPDATE_COPY.en
  const client = new DaemonClient(options.url)
  const targets: string[] = []
  if (options.all) {
    const list = await client.skills()
    for (const s of list) targets.push(s.id)
  } else if (name) {
    targets.push(name)
  } else {
    console.error(chalk.red(copy.provideNameOrAll))
    process.exit(1)
  }

  for (const id of targets) {
    try {
      const res = await client.updateSkill(id)
      if (res.changed) {
        console.log(chalk.green(`${id}: ${res.from} → ${res.to}`))
      } else {
        console.log(chalk.gray(`${id}: ${res.reason ?? copy.upToDate}`))
      }
    } catch (err) {
      console.error(chalk.red(`${id}: ${errorMessage(err)}`))
    }
  }
}
