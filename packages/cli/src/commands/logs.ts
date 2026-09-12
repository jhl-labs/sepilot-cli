import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { homedir } from 'node:os'
import chalk from 'chalk'
import { output, getOutputFormat } from '../output/formatter.js'
import { detectCliLocale } from '../utils/locale.js'

const LOGS_COPY = {
  en: {
    lastEntriesPrefix: (count: number, type: string) => `Last ${count} entries from ${type} log:\n`,
    logNotFound: (path: string) => `Log file not found: ${path}`,
    initHint: 'Run "sepilot init" to initialize, then start the daemon.',
  },
  ko: {
    lastEntriesPrefix: (count: number, type: string) => `${type} 로그의 마지막 항목 ${count}개:\n`,
    logNotFound: (path: string) => `로그 파일을 찾을 수 없습니다: ${path}`,
    initHint: '초기화하려면 "sepilot init"을 실행한 후 daemon을 시작하세요.',
  },
} as const

export async function logsCommand(options: { lines?: string; type?: string }) {
  const copy = LOGS_COPY[detectCliLocale()] ?? LOGS_COPY.en
  const dataDir = join(homedir(), '.sepilotd')
  const maxLines = parseInt(options.lines ?? '50')
  const logType = options.type ?? 'audit'

  let logPath: string
  switch (logType) {
    case 'audit': logPath = join(dataDir, 'security', 'audit.log'); break
    case 'daemon': logPath = join(dataDir, 'logs', 'daemon.log'); break
    default: logPath = join(dataDir, 'security', 'audit.log')
  }

  try {
    const content = await readFile(logPath, 'utf-8')
    const lines = content.trim().split('\n').filter(Boolean)
    const recent = lines.slice(-maxLines)

    if (getOutputFormat() === 'json') {
      output(recent.map(l => { try { return JSON.parse(l) } catch { return { raw: l } } }))
      return
    }

    console.log(chalk.gray(copy.lastEntriesPrefix(recent.length, logType)))
    for (const line of recent) {
      try {
        const entry = JSON.parse(line)
        const ts = entry.timestamp ?? entry.ts ?? ''
        const event = entry.event ?? ''
        // `denied` / `rejected` / `failed` carry the same weight as
        // `error` / `violation` for an operator skimming the log; lump
        // them together so red shows on the lines that matter.
        const severity = /error|violation|denied|rejected|failed/i.test(event)
          ? chalk.red
          : /warn|throttle/i.test(event)
            ? chalk.yellow
            : chalk.gray
        console.log(`${chalk.gray(ts)} ${severity(event.padEnd(30))} ${entry.tool ?? entry.channel ?? ''}`)
      } catch {
        console.log(chalk.gray(line))
      }
    }
  } catch {
    console.error(chalk.red(copy.logNotFound(logPath)))
    console.error(chalk.gray(copy.initHint))
  }
}
