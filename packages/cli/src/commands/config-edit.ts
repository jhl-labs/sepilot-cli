import { spawn } from 'node:child_process'
import { join } from 'node:path'
import { access } from 'node:fs/promises'
import chalk from 'chalk'
import { resolveDaemonDataDir } from '../client/token.js'
import { detectCliLocale } from '../utils/locale.js'

const CONFIG_EDIT_COPY = {
  en: {
    configNotFound: 'Config not found. Run "sepilot init" first.',
    openingPrefix: (path: string, editor: string) => `Opening ${path} with ${editor}...`,
    configSaved: 'Config saved. Restart daemon to apply changes: sepilot restart',
  },
  ko: {
    configNotFound: '구성을 찾을 수 없습니다. 먼저 "sepilot init"을 실행하세요.',
    openingPrefix: (path: string, editor: string) => `${path}을(를) ${editor}로 열고 있습니다...`,
    configSaved: '구성이 저장되었습니다. 변경 사항을 적용하려면 daemon을 다시 시작하세요: sepilot restart',
  },
} as const

export async function configEditCommand() {
  const locale = detectCliLocale()
  const copy = CONFIG_EDIT_COPY[locale] ?? CONFIG_EDIT_COPY.en
  const configPath = join(resolveDaemonDataDir(), 'config.yaml')

  try {
    await access(configPath)
  } catch {
    console.error(chalk.red(copy.configNotFound))
    process.exit(1)
  }

  const editor = process.env.EDITOR ?? process.env.VISUAL ?? 'nano'
  console.log(chalk.gray(copy.openingPrefix(configPath, editor)))

  const child = spawn(editor, [configPath], { stdio: 'inherit' })
  child.on('exit', (code: number) => {
    if (code === 0) {
      console.log(chalk.green(copy.configSaved))
    }
  })
}
