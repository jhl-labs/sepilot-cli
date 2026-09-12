import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output, getOutputFormat } from '../output/formatter.js'
import { detectCliLocale } from '../utils/locale.js'

const HISTORY_COPY = {
  en: {
    noMatch: 'No matching sessions found.',
    foundPrefix: (count: number, query: string) => `Found ${count} sessions matching "${query}":\n`,
    untitled: '(untitled)',
    msgsSuffix: (n: number) => `${n} msgs`,
    andMore: (count: number) => `\n  ... and ${count} more. Use --limit to show more.`,
    cannotConnect: 'Cannot connect to sepilotd.',
  },
  ko: {
    noMatch: '일치하는 세션을 찾을 수 없습니다.',
    foundPrefix: (count: number, query: string) => `"${query}"에 일치하는 세션 ${count}개 발견:\n`,
    untitled: '(제목 없음)',
    msgsSuffix: (n: number) => `메시지 ${n}개`,
    andMore: (count: number) => `\n  ... 그리고 ${count}개 더. 더 표시하려면 --limit을 사용하세요.`,
    cannotConnect: 'sepilotd에 연결할 수 없습니다.',
  },
} as const

export async function historyCommand(query: string, options: { url?: string; limit?: string }) {
  const copy = HISTORY_COPY[detectCliLocale()] ?? HISTORY_COPY.en
  const client = new DaemonClient(options.url)

  try {
    const sessions = await client.sessions(query)

    if (getOutputFormat() === 'json') {
      output(sessions)
      return
    }

    if (!sessions.items?.length) {
      console.log(chalk.gray(copy.noMatch))
      return
    }

    const limit = parseInt(options.limit ?? '20')
    const items = sessions.items.slice(0, limit)

    console.log(chalk.gray(copy.foundPrefix(sessions.totalCount, query)))

    for (const s of items) {
      // ISO date so the output is locale-independent — matches what
      // sessions list / sessions show produce.
      const ts = new Date(s.updatedAt).toISOString().slice(0, 16).replace('T', ' ')
      const id = s.id.length > 18 ? `${s.id.slice(0, 17)}…` : s.id.padEnd(18)
      console.log(`  ${chalk.cyan(id)}  ${chalk.gray(ts)}  ${s.title ?? copy.untitled}`)
      console.log(chalk.gray(`                                      ${s.provider}/${s.model} · ${copy.msgsSuffix(s.messageCount)} · ${s.status}`))
    }

    if (sessions.totalCount > limit) {
      console.log(chalk.gray(copy.andMore(sessions.totalCount - limit)))
    }
  } catch {
    console.error(chalk.red(copy.cannotConnect))
    process.exit(1)
  }
}
