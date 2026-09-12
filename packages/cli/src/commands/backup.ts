import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { createInterface } from 'node:readline'
import { dirname, join } from 'node:path'
import { homedir } from 'node:os'
import { access, mkdir, stat } from 'node:fs/promises'
import chalk from 'chalk'
import { errorMessage } from '../utils/error-message.js'
import { getOutputFormat, output } from '../output/formatter.js'
import { tildify } from '../utils/path-display.js'
import { detectCliLocale } from '../utils/locale.js'

const execAsync = promisify(execFile)

const BACKUP_COPY = {
  en: {
    noDataDir: 'No sepilotd data directory found. Run "sepilot init" first.',
    backupCreated: (out: string) => `Backup created: ${out} `,
    backupCreatedMeta: (size: string, src: string) => `(${size} from ${src})`,
    backupFailedPrefix: (reason: string) => `Backup failed: ${reason}`,
    cannotWritePermission: (path: string) => `cannot write to ${path} (permission denied)`,
    isDirectory: (path: string) => `${path} is a directory; pass a file path to --output`,
    parentMissing: (path: string) => `parent directory missing for ${path}`,
    diskFull: (path: string) => `disk full while writing ${path}`,
    notGzipArchive: (path: string) => `${path} is not a valid gzipped tar archive`,
    backupNotFound: (path: string) => `Backup file not found: ${path}`,
    refusingOverwrite: (path: string) => `Refusing to overwrite ${path} without confirmation.`,
    passYesHint: 'Pass --yes to skip the prompt when scripting.',
    confirmOverwrite: (path: string) => `This will overwrite ${path}. Continue? [y/N] `,
    aborted: 'Aborted.',
    restoredPrefix: (from: string, to: string) => `Restored from ${from} → ${to}`,
    restoreFailedPrefix: (reason: string) => `Restore failed: ${reason}`,
  },
  ko: {
    noDataDir: 'sepilotd 데이터 디렉토리를 찾을 수 없습니다. 먼저 "sepilot init"을 실행하세요.',
    backupCreated: (out: string) => `백업 생성됨: ${out} `,
    backupCreatedMeta: (size: string, src: string) => `(${src}에서 ${size})`,
    backupFailedPrefix: (reason: string) => `백업 실패: ${reason}`,
    cannotWritePermission: (path: string) => `${path}에 쓸 수 없습니다 (권한 거부됨)`,
    isDirectory: (path: string) => `${path}은(는) 디렉토리입니다. --output에 파일 경로를 전달하세요`,
    parentMissing: (path: string) => `${path}의 상위 디렉토리가 없습니다`,
    diskFull: (path: string) => `${path} 쓰는 중 디스크가 가득 찼습니다`,
    notGzipArchive: (path: string) => `${path}은(는) 유효한 gzip tar 아카이브가 아닙니다`,
    backupNotFound: (path: string) => `백업 파일을 찾을 수 없습니다: ${path}`,
    refusingOverwrite: (path: string) => `확인 없이 ${path}을(를) 덮어쓰기를 거부합니다.`,
    passYesHint: '스크립트 작성 시 프롬프트를 건너뛰려면 --yes를 전달하세요.',
    confirmOverwrite: (path: string) => `${path}을(를) 덮어씁니다. 계속하시겠습니까? [y/N] `,
    aborted: '중단됨.',
    restoredPrefix: (from: string, to: string) => `${from}에서 복원됨 → ${to}`,
    restoreFailedPrefix: (reason: string) => `복원 실패: ${reason}`,
  },
} as const

type BackupCopy = (typeof BACKUP_COPY)[keyof typeof BACKUP_COPY]

async function promptYesNo(question: string): Promise<boolean> {
  const rl = createInterface({ input: process.stdin, output: process.stdout })
  try {
    const answer: string = await new Promise((resolve) => rl.question(question, resolve))
    return /^y(es)?$/i.test(answer.trim())
  } finally {
    rl.close()
  }
}

function formatBytes(size: number): string {
  if (size < 1024) return `${size}B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)}K`
  if (size < 1024 * 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)}M`
  return `${(size / 1024 / 1024 / 1024).toFixed(2)}G`
}

export async function backupCommand(options: { output?: string }) {
  const copy = BACKUP_COPY[detectCliLocale()] ?? BACKUP_COPY.en
  const dataDir = join(homedir(), '.sepilotd')

  try {
    await access(dataDir)
  } catch {
    if (getOutputFormat() === 'json') {
      output({ ok: false, error: 'no-data-dir', dataDir })
    } else {
      console.error(chalk.red(copy.noDataDir))
    }
    process.exit(1)
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  const outputPath = options.output ?? join(homedir(), `sepilotd-backup-${timestamp}.tar.gz`)

  try {
    // Check the filesystem directly; BSD tar omits errno details for bad outputs.
    const parent = await stat(dirname(outputPath))
    if (!parent.isDirectory()) throw Object.assign(new Error('Output parent is not a directory'), { code: 'ENOTDIR' })
    const target = await stat(outputPath).catch((error: NodeJS.ErrnoException) => {
      if (error.code === 'ENOENT') return null
      throw error
    })
    if (target?.isDirectory()) throw Object.assign(new Error('Output path is a directory'), { code: 'EISDIR' })
    await execAsync('tar', ['-czf', outputPath, '-C', homedir(), '.sepilotd'])
    const info = await stat(outputPath)
    output(
      { ok: true, dataDir, outputPath, sizeBytes: info.size },
      (data) =>
        chalk.green(
          `${copy.backupCreated(tildify(data.outputPath))}${chalk.gray(copy.backupCreatedMeta(formatBytes(data.sizeBytes), tildify(data.dataDir)))}`,
        ),
    )
  } catch (err) {
    const reason = friendlyTarError(err, outputPath, copy)
    if (getOutputFormat() === 'json') {
      output({ ok: false, error: reason, dataDir, outputPath })
    } else {
      console.error(chalk.red(copy.backupFailedPrefix(reason)))
    }
    process.exit(1)
  }
}

/**
 * Translate the multi-line tar error spew into a single user-facing
 * sentence. The raw stderr from `tar` is locale-dependent (Korean
 * '허가 거부' on this host) and dumps four lines per failure, which
 * looked like a stack trace to operators.
 */
function friendlyTarError(
  err: unknown,
  outputPath: string,
  copy: BackupCopy = BACKUP_COPY.en,
): string {
  // execFile failures stash the actual diagnostic in `stderr`; the
  // top-level message is just `Command failed: tar ...`. Combine
  // both so we can match locale-specific tar output.
  const code = (err as NodeJS.ErrnoException).code
  if (code === 'ENOENT' || code === 'ENOTDIR') return copy.parentMissing(outputPath)
  const raw = errorMessage(err)
  const stderr = (err as { stderr?: string | Buffer }).stderr
  const stderrText = typeof stderr === 'string'
    ? stderr
    : Buffer.isBuffer(stderr) ? stderr.toString('utf-8') : ''
  const combined = `${raw}\n${stderrText}`

  if (/permission denied|허가 거부|권한 없음|EACCES/i.test(combined)) {
    return copy.cannotWritePermission(outputPath)
  }
  if (/디렉터리입니다|is a directory|EISDIR/i.test(combined)) {
    return copy.isDirectory(outputPath)
  }
  if (/no such file|존재하지 않|디렉터리가? 없|파일이나 디렉터리|ENOENT/i.test(combined)) {
    return copy.parentMissing(outputPath)
  }
  if (/disk full|no space|ENOSPC/i.test(combined)) {
    return copy.diskFull(outputPath)
  }
  if (/invalid compressed data|gzip|not in gzip format/i.test(combined)) {
    return copy.notGzipArchive(outputPath)
  }
  // Last-resort: keep just the first stderr line (or message line)
  // so we don't bury the user under five lines of tar diagnostics.
  const firstStderrLine = stderrText.split('\n').find((line) => line.trim().length > 0)
  return firstStderrLine ?? raw.split('\n')[0] ?? raw
}

export async function restoreCommand(
  input: string,
  options: { target?: string; yes?: boolean } = {},
) {
  const copy = BACKUP_COPY[detectCliLocale()] ?? BACKUP_COPY.en
  const targetParent = options.target ? options.target : homedir()
  const dataDir = join(targetParent, '.sepilotd')

  try {
    await access(input)
  } catch {
    if (getOutputFormat() === 'json') {
      output({ ok: false, error: 'backup-not-found', input })
    } else {
      console.error(chalk.red(copy.backupNotFound(tildify(input))))
    }
    process.exit(1)
  }

  if (!options.yes && getOutputFormat() !== 'json') {
    // Refuse to proceed without an explicit confirmation. The legacy
    // 3-2-1 countdown could overwrite ~/.sepilotd if a user couldn't
    // hit Ctrl+C in time. A real y/n prompt requires intent.
    if (!process.stdin.isTTY) {
      console.error(chalk.red(copy.refusingOverwrite(tildify(dataDir))))
      console.error(chalk.gray(copy.passYesHint))
      process.exit(1)
    }
    const ok = await promptYesNo(
      chalk.yellow(copy.confirmOverwrite(tildify(dataDir))),
    )
    if (!ok) {
      console.log(chalk.gray(copy.aborted))
      process.exit(0)
    }
  }

  try {
    await mkdir(targetParent, { recursive: true })
    await execAsync('tar', ['-xzf', input, '-C', targetParent])
    output(
      { ok: true, restoredFrom: input, target: dataDir },
      (data) => chalk.green(copy.restoredPrefix(tildify(data.restoredFrom), tildify(data.target))),
    )
  } catch (err) {
    const reason = friendlyTarError(err, dataDir, copy)
    if (getOutputFormat() === 'json') {
      output({ ok: false, error: reason, input, target: dataDir })
    } else {
      console.error(chalk.red(copy.restoreFailedPrefix(reason)))
    }
    process.exit(1)
  }
}

export const __testables = {
  formatBytes,
  friendlyTarError,
}
