import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output, outputError } from '../output/formatter.js'
import { friendlyErrorMessage as errorMessage } from '../utils/error-message.js'
import { detectCliLocale } from '../utils/locale.js'

const MARKETPLACE_COPY = {
  en: {
    noMarketplaces: 'No marketplaces registered.',
    invalidName: (name: string) =>
      `Invalid marketplace name: ${name} (use letters, digits, '-', '_', '.' only; must start with a letter or digit)`,
    addedPrefix: (name: string, url: string) => `Added marketplace: ${name} — ${url}`,
    addFailedPrefix: (name: string, msg: string) => `Failed to add marketplace ${name}: ${msg}`,
    removedPrefix: (name: string) => `Removed marketplace: ${name}`,
    notFound: (name: string) => `Marketplace not found: ${name}`,
  },
  ko: {
    noMarketplaces: '등록된 마켓플레이스가 없습니다.',
    invalidName: (name: string) =>
      `잘못된 마켓플레이스 이름: ${name} (문자, 숫자, '-', '_', '.'만 사용; 문자나 숫자로 시작해야 함)`,
    addedPrefix: (name: string, url: string) => `마켓플레이스 추가됨: ${name} — ${url}`,
    addFailedPrefix: (name: string, msg: string) => `마켓플레이스 ${name} 추가 실패: ${msg}`,
    removedPrefix: (name: string) => `마켓플레이스 제거됨: ${name}`,
    notFound: (name: string) => `마켓플레이스를 찾을 수 없습니다: ${name}`,
  },
} as const

export async function marketplaceListCommand(options: { url?: string }) {
  const copy = MARKETPLACE_COPY[detectCliLocale()] ?? MARKETPLACE_COPY.en
  const client = new DaemonClient(options.url)
  const list = await client.listMarketplaces()
  output(list ?? [], (entries) => {
    if (!entries.length) return copy.noMarketplaces
    return entries.map((m) => `  ${chalk.bold(m.name.padEnd(12))} ${m.url}`).join('\n')
  })
}

export async function marketplaceAddCommand(name: string, url: string, options: { url?: string }) {
  const copy = MARKETPLACE_COPY[detectCliLocale()] ?? MARKETPLACE_COPY.en
  if (!/^[a-z0-9][a-z0-9-_.]*$/i.test(name)) {
    console.error(chalk.red(copy.invalidName(name)))
    process.exit(1)
  }
  const client = new DaemonClient(options.url)
  try {
    await client.addMarketplace(name, url)
    output({ ok: true, name, url, action: 'added' }, () =>
      chalk.green(copy.addedPrefix(chalk.bold(name), url)),
    )
  } catch (err) {
    outputError(
      { ok: false, name, url, error: errorMessage(err) },
      () => chalk.red(copy.addFailedPrefix(name, errorMessage(err))),
    )
    process.exit(1)
  }
}

export async function marketplaceRemoveCommand(name: string, options: { url?: string }) {
  const copy = MARKETPLACE_COPY[detectCliLocale()] ?? MARKETPLACE_COPY.en
  const client = new DaemonClient(options.url)
  const res = await client.removeMarketplace(name)
  if (res.removed) {
    output({ ok: true, name, removed: true }, () =>
      chalk.green(copy.removedPrefix(chalk.bold(name))),
    )
    return
  }
  outputError({ ok: false, name, error: 'marketplace-not-found' }, () =>
    chalk.yellow(copy.notFound(name)),
  )
  process.exit(1)
}
