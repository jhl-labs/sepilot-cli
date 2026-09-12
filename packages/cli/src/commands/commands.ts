import chalk from 'chalk'
import { readFile } from 'node:fs/promises'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { detectCliLocale } from '../utils/locale.js'

const COMMANDS_COPY = {
  en: {
    noCommands: 'No custom commands defined.',
    invalidCommandId: (id: string) =>
      `Invalid command id: ${id} (use letters, digits, '-', '_', '.' only; must start with a letter or digit)`,
    provideBodyOrFile: 'Provide --body "<text>" or --body-file <path>',
    bodyFileNotFound: (path: string) => `--body-file not found: ${path}`,
    bodyFileNotReadable: (path: string) => `--body-file not readable: ${path} (permission denied)`,
    createdCommand: (id: string) => `Created /${id}`,
    deletedCommand: (id: string) => `Deleted /${id}`,
  },
  ko: {
    noCommands: '정의된 사용자 정의 명령이 없습니다.',
    invalidCommandId: (id: string) =>
      `잘못된 명령 ID: ${id} (문자, 숫자, '-', '_', '.'만 사용; 문자나 숫자로 시작해야 함)`,
    provideBodyOrFile: '--body "<텍스트>" 또는 --body-file <경로>를 제공하세요',
    bodyFileNotFound: (path: string) => `--body-file을 찾을 수 없습니다: ${path}`,
    bodyFileNotReadable: (path: string) => `--body-file 읽기 불가: ${path} (권한 거부됨)`,
    createdCommand: (id: string) => `/${id} 생성됨`,
    deletedCommand: (id: string) => `/${id} 삭제됨`,
  },
} as const

function commandsCopy() {
  return COMMANDS_COPY[detectCliLocale()] ?? COMMANDS_COPY.en
}

interface BaseOptions {
  url?: string
}

export async function commandsListCommand(options: BaseOptions): Promise<void> {
  const copy = commandsCopy()
  const client = new DaemonClient(options.url)
  const list = await client.userCommands()
  output(list, (data) => {
    if (data.length === 0) return copy.noCommands
    return data
      .map((cmd) => `/${chalk.bold(cmd.id)}  ${chalk.gray(`args=${cmd.args}`)}\n  ${cmd.description}`)
      .join('\n\n')
  })
}

export interface CommandsCreateOptions extends BaseOptions {
  description: string
  args?: 'none' | 'optional' | 'required'
  agent?: string
  model?: string
  body?: string
  bodyFile?: string
}

export async function commandsCreateCommand(
  id: string,
  options: CommandsCreateOptions,
): Promise<void> {
  const copy = commandsCopy()
  if (!/^[a-z0-9][a-z0-9-_.]*$/i.test(id)) {
    throw new Error(copy.invalidCommandId(id))
  }
  if (!options.body && !options.bodyFile) {
    throw new Error(copy.provideBodyOrFile)
  }
  let body: string
  if (options.bodyFile) {
    try {
      body = await readFile(options.bodyFile, 'utf-8')
    } catch (err) {
      const code = (err as { code?: string }).code
      if (code === 'ENOENT') {
        throw new Error(copy.bodyFileNotFound(options.bodyFile))
      }
      if (code === 'EACCES') {
        throw new Error(copy.bodyFileNotReadable(options.bodyFile))
      }
      throw err
    }
  } else {
    body = options.body!
  }
  const client = new DaemonClient(options.url)
  const result = await client.createUserCommand({
    id,
    description: options.description,
    args: options.args,
    agent: options.agent,
    model: options.model,
    body,
  })
  output({ ok: true, ...result }, (data) => chalk.green(copy.createdCommand(data.id)))
}

export async function commandsDeleteCommand(
  id: string,
  options: BaseOptions,
): Promise<void> {
  const copy = commandsCopy()
  const client = new DaemonClient(options.url)
  await client.deleteUserCommand(id)
  output({ ok: true, id, deleted: true }, () => chalk.yellow(copy.deletedCommand(id)))
}
