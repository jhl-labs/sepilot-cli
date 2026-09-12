import chalk from 'chalk'
import { readFile } from 'node:fs/promises'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { detectCliLocale } from '../utils/locale.js'

const AGENTS_COPY = {
  en: {
    noAgentsRegistered: 'No agents registered.',
    invalidAgentId: (id: string) =>
      `Invalid agent id: ${id} (use letters, digits, '-' or '_' only; must start with a letter or digit)`,
    descriptionRequired: 'Provide --description "<text>"',
    providePromptOrFile: 'Provide --prompt "<text>" or --prompt-file <path>',
    emptyPrompt: 'Agent system prompt must not be empty',
    promptFileNotFound: (path: string) => `--prompt-file not found: ${path}`,
    promptFileNotReadable: (path: string) => `--prompt-file not readable: ${path} (permission denied)`,
    invalidTemperature: (raw: string) =>
      `--temperature must be a number from 0 to 2 (got ${raw})`,
    invalidMaxIterations: (raw: string) =>
      `--max-iterations must be a positive integer (got ${raw})`,
    createdAgentPrefix: (id: string, source: string) => `Created agent ${id} (source: ${source})`,
    tryCommand: (id: string) => `  try: sepilot subagent dispatch --agent ${id} "review this change"`,
    warningPrefix: '  warning: ',
    credentialWarning:
      'prompt may contain credentials or tokens; avoid storing secrets in agent prompts.',
    systemInfoWarning:
      'prompt may contain local user, host, email, or IP details; keep only intentional reusable context.',
    listRunHint: (id: string) => `  run: sepilot subagent dispatch --agent ${id} "<task>"`,
    deletedAgentPrefix: (id: string) => `Deleted agent ${id}`,
  },
  ko: {
    noAgentsRegistered: '등록된 에이전트가 없습니다.',
    invalidAgentId: (id: string) =>
      `잘못된 에이전트 ID: ${id} (문자, 숫자, '-' 또는 '_'만 사용; 문자나 숫자로 시작해야 함)`,
    descriptionRequired: '--description "<텍스트>"를 제공하세요',
    providePromptOrFile: '--prompt "<텍스트>" 또는 --prompt-file <경로>를 제공하세요',
    emptyPrompt: '에이전트 system prompt는 비어 있으면 안 됩니다',
    promptFileNotFound: (path: string) => `--prompt-file을 찾을 수 없습니다: ${path}`,
    promptFileNotReadable: (path: string) => `--prompt-file 읽기 불가: ${path} (권한 거부됨)`,
    invalidTemperature: (raw: string) =>
      `--temperature는 0 이상 2 이하의 숫자여야 합니다 (받은 값: ${raw})`,
    invalidMaxIterations: (raw: string) =>
      `--max-iterations는 양의 정수여야 합니다 (받은 값: ${raw})`,
    createdAgentPrefix: (id: string, source: string) => `에이전트 ${id} 생성됨 (소스: ${source})`,
    tryCommand: (id: string) => `  실행 예: sepilot subagent dispatch --agent ${id} "이 변경을 리뷰해줘"`,
    warningPrefix: '  경고: ',
    credentialWarning:
      'prompt에 credential 또는 token처럼 보이는 값이 있습니다. 에이전트 prompt에는 secret을 저장하지 마세요.',
    systemInfoWarning:
      'prompt에 로컬 사용자, host, email, IP 정보처럼 보이는 값이 있습니다. 재사용 가능한 의도적 컨텍스트만 남기세요.',
    listRunHint: (id: string) => `  실행: sepilot subagent dispatch --agent ${id} "<작업>"`,
    deletedAgentPrefix: (id: string) => `에이전트 ${id} 삭제됨`,
  },
} as const

const AGENT_ID_PATTERN = /^[a-z0-9][a-z0-9-_]*$/i
const SECRET_PATTERNS = [
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/i,
  /\b(?:api[_-]?key|secret|token|password)\b\s*[:=]/i,
  /\bBearer\s+[A-Za-z0-9._~+/=-]{12,}/i,
] as const
const SYSTEM_INFO_PATTERNS = [
  /\b(?:home|users)\/[A-Za-z0-9._-]+/i,
  /\b[A-Za-z0-9._-]+@[A-Za-z0-9._-]+\b/,
  /\b(?:\d{1,3}\.){3}\d{1,3}\b/,
] as const

interface BaseOptions {
  url?: string
}

export async function agentsListCommand(options: BaseOptions): Promise<void> {
  const copy = AGENTS_COPY[detectCliLocale()] ?? AGENTS_COPY.en
  const client = new DaemonClient(options.url)
  const agents = await client.agents()
  output(agents, (data) => {
    if (data.length === 0) return copy.noAgentsRegistered
    return data
      .map((agent) => {
        const tag = agent.source ? chalk.gray(`[${agent.source}]`) : ''
        const name = agent.name && agent.name !== agent.id ? ` ${agent.name}` : ''
        const lines = [
          `${chalk.bold(agent.id)}${name} ${tag}`.trimEnd(),
          `  ${agent.description}`,
        ]
        if (agent.source === 'user') {
          lines.push(copy.listRunHint(agent.id))
        }
        return lines.join('\n')
      })
      .join('\n\n')
  })
}

export interface AgentsCreateOptions extends BaseOptions {
  description: string
  base?: string
  model?: string
  temperature?: string
  maxIterations?: string
  prompt?: string
  promptFile?: string
}

function parseTemperature(raw: string | undefined): number | undefined {
  if (raw === undefined) return undefined
  const parsed = Number(raw)
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 2) {
    throw new Error((AGENTS_COPY[detectCliLocale()] ?? AGENTS_COPY.en).invalidTemperature(raw))
  }
  return parsed
}

function parseMaxIterations(raw: string | undefined): number | undefined {
  if (raw === undefined) return undefined
  const parsed = Number(raw)
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error((AGENTS_COPY[detectCliLocale()] ?? AGENTS_COPY.en).invalidMaxIterations(raw))
  }
  return parsed
}

function findAgentPromptWarnings(systemPrompt: string): string[] {
  const copy = AGENTS_COPY[detectCliLocale()] ?? AGENTS_COPY.en
  const warnings: string[] = []
  if (SECRET_PATTERNS.some((pattern) => pattern.test(systemPrompt))) {
    warnings.push(copy.credentialWarning)
  }
  if (SYSTEM_INFO_PATTERNS.some((pattern) => pattern.test(systemPrompt))) {
    warnings.push(copy.systemInfoWarning)
  }
  return warnings
}

export async function agentsCreateCommand(
  id: string,
  options: AgentsCreateOptions,
): Promise<void> {
  const copy = AGENTS_COPY[detectCliLocale()] ?? AGENTS_COPY.en
  if (!AGENT_ID_PATTERN.test(id)) {
    throw new Error(copy.invalidAgentId(id))
  }
  if (!options.description?.trim()) {
    throw new Error(copy.descriptionRequired)
  }
  if (!options.prompt && !options.promptFile) {
    throw new Error(copy.providePromptOrFile)
  }
  let systemPrompt: string
  if (options.promptFile) {
    try {
      systemPrompt = await readFile(options.promptFile, 'utf-8')
    } catch (err) {
      const code = (err as { code?: string }).code
      if (code === 'ENOENT') {
        throw new Error(copy.promptFileNotFound(options.promptFile))
      }
      if (code === 'EACCES') {
        throw new Error(copy.promptFileNotReadable(options.promptFile))
      }
      throw err
    }
  } else {
    systemPrompt = options.prompt!
  }
  if (!systemPrompt.trim()) {
    throw new Error(copy.emptyPrompt)
  }
  const promptWarnings = findAgentPromptWarnings(systemPrompt)
  const client = new DaemonClient(options.url)
  const result = await client.createUserAgent({
    id,
    description: options.description.trim(),
    base: options.base,
    model: options.model,
    temperature: parseTemperature(options.temperature),
    maxIterations: parseMaxIterations(options.maxIterations),
    systemPrompt,
  })
  output(
    { ok: true, ...result, promptWarnings },
    (data) => [
      chalk.green(copy.createdAgentPrefix(data.id, data.source ?? 'user')),
      copy.tryCommand(data.id),
      ...data.promptWarnings.map((warning) => `${copy.warningPrefix}${warning}`),
    ].join('\n'),
  )
}

export async function agentsDeleteCommand(
  id: string,
  options: BaseOptions,
): Promise<void> {
  const copy = AGENTS_COPY[detectCliLocale()] ?? AGENTS_COPY.en
  const client = new DaemonClient(options.url)
  await client.deleteUserAgent(id)
  output({ ok: true, id, deleted: true }, () => chalk.yellow(copy.deletedAgentPrefix(id)))
}
