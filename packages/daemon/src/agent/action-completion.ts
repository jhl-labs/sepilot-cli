import type { Message } from '@sepilotd/core'
import { containsPromptToolCallEnvelope } from './prompt-react.js'
import {
  inputExplicitlyRequestsRenderedUiValidation,
  inputExplicitlyForbidsFileMutation,
  inputExpressesMutationIntent,
  inputPositiveCapabilityScope,
  inputRequestsNonRenderedSurfaceInventory,
} from './task-contract.js'
import { CURRENT_AGENT_TURN_USER_METADATA_KEY } from './turn-context.js'
import { TOOL_RESULT_STATUS_METADATA_KEY } from './memory-write-completion.js'
import { APPROVAL_FAILURE_STATUS_METADATA_KEY } from './approval-failure.js'
import { isVisibleUrlOpenRequest } from './desktop-control-intent.js'
import {
  extractConstrainedProcessStart,
  extractFocusedProcessStart,
  extractFocusedTerminalCommand,
} from './request-shape.js'

export type ActionEvidenceKind =
  | 'file'
  | 'file-read'
  | 'file-inventory'
  | 'command'
  | 'process-observation'
  | 'browser'
  | 'visible-url'
export type ActionCompletionOutcome = 'not_required' | 'success' | 'blocked' | 'failed' | 'missing'

export interface ActionCompletionEvaluation {
  requiredKinds: ActionEvidenceKind[]
  missingKinds: ActionEvidenceKind[]
  outcome: ActionCompletionOutcome
  hasUnexecutedToolText: boolean
  hasAvailableEvidenceTool: boolean
  lastFailureSummary?: string
}

const FILE_MUTATION_TOOL_NAMES = new Set(['apply_patch', 'fs.append', 'fs.edit', 'fs.write'])
const FILE_READ_TOOL_NAMES = new Set([
  'fs.read',
  // A structured Office slide read is the authoritative content read for a
  // presentation. Requiring an additional fs.read after this succeeds makes
  // the completion guard retry a binary PPTX through the wrong abstraction.
  'office.read_slide',
])
const FILE_INVENTORY_TOOL_NAMES = new Set(['fs.list'])
const COMMAND_EXECUTION_TOOL_NAMES = new Set([
  'computer.launch_app',
  'process.start',
  'process.stop',
  'process.signal',
  'service.start',
  'service.stop',
  'service.restart',
  'service.remove',
  'service.healthcheck',
  'terminal.run',
])
const PROCESS_OBSERVATION_TOOL_NAMES = new Set([
  'process.follow',
  'process.list',
  'process.read',
  'process.sessions',
  'process.wait',
])
const BROWSER_EXECUTION_TOOL_NAMES = new Set([
  'browser.click',
  'browser.evaluate',
  'browser.extract',
  'browser.navigate',
  'browser.screenshot',
])
const VISIBLE_URL_EXECUTION_TOOL_NAMES = new Set(['computer.open_url'])
const EXPLICIT_BROWSER_OBSERVATION_ACTION_PATTERN =
  /\b(?:open|navigate|visit|inspect|validate|verify|test|audit|check|capture|take|exercise|interact|click|evaluate|extract)\b|(?:열어|접속|탐색|검사|검수|확인|테스트|검증|감사|캡처|촬영|상호작용|클릭|평가|추출)/iu
const PROCESS_OBSERVATION_TOOL_PATTERN =
  /\bprocess\.(?:follow|list|read|sessions|wait)\b/giu
const PROCESS_OBSERVATION_ACTION_PATTERN =
  /\b(?:call|check|follow|inspect|invoke|list|observe|read|show|use|wait)\b|(?:사용|호출|읽|확인|관찰|조회|대기|따라가)(?:고|기|어|어서|해|해서|하여|하고|하라|하세요|해줘|해주세요)?/iu
const PROCESS_OBSERVATION_META_PATTERN =
  /\b(?:how|what|why)\b[^\n]{0,80}\b(?:api|schema|use)\b|(?:사용법|사용\s*방법|API\s*(?:설명|문서)|스키마\s*(?:설명|문서))/iu
const STRUCTURED_TOOL_REFERENCE_PATTERN =
  /\b(?:a2a|apps|browser|code|computer|device|doc|external_acp|fs|git|image_gen|jpad|market|mcp|media|memory|monitor|notebook|notification|office|pages|plugin|process|self|service|skillhub|subagent|swarm|system|terminal|usage|web|workspace)\.(?:\*|[A-Za-z][\w.-]*)(?=$|[^\w.-])/giu
const TOOL_SHAPED_ARTIFACT_SUFFIX_PATTERN =
  /\.(?:cjs|conf|css|csv|cts|docx?|gif|go|gql|graphql|html?|ini|ipynb|java|jpe?g|js|json|jsonl|jsx|kt|kts|less|md|mdx|mjs|mts|pdf|php|png|proto|ps1|py|rb|rs|scss|sh|sql|svg|swift|tar|toml|ts|tsv|tsx|txt|webp|xlsx?|xml|ya?ml|zip)$/iu
const SEARCH_RETRIEVAL_LABEL_PATTERN =
  /\bweb\.(?:\*|[A-Za-z][\w.-]*)(?=$|[^\w.-])|\b(?:public-web|internal-index)\b/giu
const BARE_NETWORK_HOST_PATTERN =
  /(?<![A-Za-z0-9_/@.-])(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.){2,}[A-Za-z]{2,63}(?![A-Za-z0-9_.-])/giu
const NETWORK_HOST_CONTEXT_PATTERN =
  /\b(?:domain|endpoint|host|index|provider|query|search|service|site|url|web)\b|(?:검색|서비스|엔드포인트|호스트|도메인|색인|인덱스|사이트|접속|조회)/iu
const FOCUSED_SINGLE_PROCESS_OBSERVATION_BOUNDARY_PATTERN =
  /\b(?:exactly\s+once|once|only|single)\b|(?:정확히\s*)?(?:한\s*번|한번|하나만|1회)/iu
const NON_PROCESS_TOOL_REFERENCE_PATTERN =
  /\b(?:apply_patch|webfetch|(?:browser|computer|doc|fs|git|memory|notebook|office|service|terminal)\.[A-Za-z][\w.-]*)\b/iu

const SWARM_GOAL_PREFIX = 'Swarm goal:\n'
const SWARM_OPERATIONAL_CONTRACT_SEPARATOR = '\n\nOperational contract:\n'

const QUOTED_FILE_TARGET_PATTERN =
  /^(?:(?:[A-Za-z]:)?[\\/]|\.{0,2}[\\/]|~[\\/])?[A-Za-z0-9_.@ -]+(?:[\\/][A-Za-z0-9_.@ -]+)*\.[A-Za-z][A-Za-z0-9_-]{0,15}$/u

const FILE_SYSTEM_TARGET_PATTERN =
  /(?:^|[\s`"'([])(?:(?:[A-Za-z]:)?[\\/]|\.{1,2}[\\/]|~[\\/])?[\p{L}\p{N}_.@ -]+(?:[\\/][\p{L}\p{N}_.@ -]+)*\.[A-Za-z][A-Za-z0-9_-]{0,15}(?=$|[^A-Za-z0-9_-])|\b(?:app|application|bug|cli|code|codebase|component|config(?:uration)?|directory|document|feature|file|folder|game|implementation|module|package|program|project|report|repo(?:sitory)?|script|source|test|website|workspace)\b|(?:앱|애플리케이션|버그|코드|코드베이스|컴포넌트|설정|디렉터리|디렉토리|문서|보고서|기능|파일|폴더|게임|구현|모듈|패키지|프로그램|프로젝트|레포|리포지토리|스크립트|소스|테스트|웹사이트|워크스페이스)/iu
const EXPLICIT_FILE_PATH_TARGET_PATTERN =
  /(?:^|[\s`"'([])(?:(?:[A-Za-z]:)?[\\/]|\.{1,2}[\\/]|~[\\/])?[\p{L}\p{N}_.@ -]+(?:[\\/][\p{L}\p{N}_.@ -]+)*\.[A-Za-z][A-Za-z0-9_-]{0,15}(?=$|[^A-Za-z0-9_-])|\b(?:README|LICENSE|Dockerfile|Makefile)\b/iu
const DIRECT_ENGLISH_FILE_READ_TARGET_PATTERN =
  /\b(?:open|read|inspect|review|analy[sz]e|summari[sz]e)\s+(?:(?!\b(?:and|then|before|after|return|provide|write|draft|create)\b)[A-Za-z0-9_-]+\s+){0,4}(?:document|file|report)\b/iu
const DIRECT_KOREAN_FILE_READ_TARGET_PATTERN =
  /(?:문서|파일|보고서)(?:은|는|이|가|을|를|의)?\s*(?:(?:내용|본문|상태|요약|세부|전문)(?:은|는|이|가|을|를|의)?\s*)?(?:읽어|읽고|읽은|확인해|확인하고|살펴|검토해|검토하고|분석해|분석하고|요약해|요약하고|열어)/iu
const EXPLICIT_FILE_READ_ACTION_PATTERN =
  /^(?:please\s+)?(?:open|read|inspect|review|analy[sz]e|summari[sz]e)\b|\b(?:can|could|would)\s+you\s+(?:open|read|inspect|review|analy[sz]e|summari[sz]e)\b|(?:읽어|읽고|읽은|확인해|확인하고|살펴|검토해|검토하고|분석해|분석하고|요약해|요약하고|열어)(?:\s*(?:줘|주세요|봐|보자|줄래))?/iu
const NEGATED_FILE_READ_PATTERN =
  /\b(?:do\s+not|don't|never|not)\s+(?:open|read|inspect|review|analy[sz]e|summari[sz]e)\b|(?:안|못)\s*(?:읽|확인|살펴|검토|분석|요약|열)|(?:읽|확인|살펴|검토|분석|요약|열)(?:으)?지\s*(?:마|말|않)/iu
const FILE_CONTENT_QUESTION_PATTERN =
  /[?？]|\b(?:what|who|when|where|which|why|how)\b|(?:무엇|뭐|누구|언제|어디|어떤|왜|어떻게|인가요|입니까|담당자|상태|내용|제목)/iu
const WORKSPACE_DIRECTORY_TARGET_PATTERN =
  /(?:현재|이|작업)\s*(?:폴더|디렉터리|디렉토리|작업\s*공간|워크스페이스)|\b(?:(?:current|this|working)\s+(?:folder|directory|workspace)|workspace|repository|repo)\b/iu
const LOCAL_DIRECTORY_CONTAINER_PATTERN =
  /(?:폴더|디렉터리|디렉토리)|\b(?:folder|directory)\b/iu
const WORKSPACE_INVENTORY_CONTENT_PATTERN =
  /(?:파일|폴더|항목)|\b(?:files?|folders?|entries|contents?)\b|(?:뭐가|무엇이)\s*(?:있|들어)/iu
const FILE_INVENTORY_REQUEST_PATTERN =
  /(?:파일|폴더|항목)\s*(?:목록|리스트)|(?:목록|리스트)(?:을|를)?(?:\s*(?:보여|알려|나열))?|뭐가\s*(?:있|들어)|무엇이\s*(?:있|들어)|\b(?:list|show)\s+(?:me\s+)?(?:the\s+)?(?:files?|folders?|entries|contents?)\b|\bwhat\s+(?:files?|folders?|entries)\s+(?:are|exist)\b/iu
const WORKSPACE_FILE_COLLECTION_NOUN_PATTERN =
  /(?:파일|문서|보고서|노트)|\b(?:files?|documents?|reports?|notes?)\b/iu
const WORKSPACE_FILE_OWNERSHIP_PATTERN =
  /(?:^|[\s('"‘“])(?:내가|제가|우리가|저희가|내|제|우리|저희)(?=\s|의|가|는|은)|\b(?:my|our)\b|\b(?:did|have)\s+(?:i|we)\b/iu
const COMPLETED_WORKSPACE_FILE_ACTION_PATTERN =
  /(?:작성|기록|저장|만들|생성|수정|추가)한|\b(?:i|we)\s+(?:created|wrote|recorded|saved|modified|added)\b|\b(?:created|written|recorded|saved|modified|added)\s+by\s+(?:me|us)\b/iu
const GROUNDED_WORKSPACE_FILE_LOOKUP_PATTERN =
  /(?:무엇|뭐|어떤)\s*(?:파일|문서|보고서|노트|거|것)|(?:파일|문서|보고서|노트)(?:은|는|이|가|을|를|의)?\s*(?:무엇|뭐|어떤|목록|리스트|제목|담당자|누구|어디|언제)|\bwhich\s+(?:files?|documents?|reports?|notes?)\b|\bwhat\s+(?:files?|documents?|reports?|notes?)\b|\b(?:show|list)\s+(?:me\s+)?(?:the\s+)?(?:files?|documents?|reports?|notes?)\b/iu
const EXPLICIT_FILE_ACTION_PATTERN =
  /^(?:please\s+)?(?:author|build|change|create|delete|develop|draft|edit|fix|generate|implement|improve|make|polish|refactor|remove|rename|save|update|write)\b|\b(?:can|could|would)\s+you\s+(?:author|build|change|create|delete|develop|draft|edit|fix|generate|implement|improve|make|polish|refactor|remove|rename|save|update|write)\b|(?:작성|생성|저장|만들|기록|수정|업데이트|구현|개발|코딩|제작|고쳐|바꿔|변경|리팩터|삭제|지워|개선|다듬)(?:\s*(?:해|하여|해\s*봐|해\s*줘|해주세요|해주십시오|줘|주세요|봐|주십시오))?(?:[.!。！]?\s*)$/iu
const ADDITIONAL_FILE_MUTATION_INTENT_PATTERN =
  /\b(?:change|delete|fix|improve|polish|refactor|remove|rename)\b|(?:고치|바꾸|변경|리팩터|삭제|지우|개선|다듬)/iu
const EXPLICIT_COMMAND_ACTION_PATTERN =
  /^(?:please\s+)?(?:execute|launch|run|start)\b|\b(?:can|could|would)\s+you\s+(?:execute|launch|run|start)\b|(?:실행|구동|돌려|시작)(?:\s*(?:해|하여|시켜|해\s*봐|해\s*줘|해주세요|해주십시오|줘|주세요|봐|주십시오))?(?:[.!。！]?\s*)$/iu
const ACTION_META_QUESTION_PATTERN =
  /\b(?:did|does|how|what|when|where|whether|why)\b|(?:왜|어떻게|뭐|무엇|언제|어디|여부|했어|했나|됐어|됐나|끝났|만들었|작성했|수정했|실행했)/iu
const NEGATED_FILE_ACTION_PATTERN =
  /\b(?:do\s+not|don't|never|not)\s+(?:author|build|change|create|delete|edit|fix|generate|implement|improve|make|polish|refactor|remove|rename|save|update|write)\b|(?:안|못)\s*(?:만들|작성|생성|저장|수정|구현|고치|바꾸|변경|리팩터|삭제|지우|개선|다듬)|(?:만들|작성|생성|저장|수정|구현|개발|고치|바꾸|변경|리팩터|삭제|지우|개선|다듬)(?:하|해)?지\s*(?:마|말|않)/iu
const NEGATED_COMMAND_ACTION_PATTERN =
  /\b(?:do\s+not|don't|never|not)\s+(?:execute|launch|run|start)\b|(?:실행|구동|시작)(?:하|해)?지\s*(?:마|말|않)|(?:안|못)\s*(?:실행|구동|시작)/iu
const UNEXECUTED_TOOL_TEXT_PATTERN =
  /(?:^|\n)\s*(?:tool[_ ]?call|file_write|terminal\.run|process\.(?:follow|list|read|sessions|start|wait)|computer\.(?:launch_app|open_url)|browser\.(?:click|evaluate|extract|navigate|screenshot)|fs\.(?:append|edit|glob|list|read|write)|apply_patch)\s*:/iu
const REPORTED_APPROVAL_DENIAL_PATTERN =
  /^\s*(?:(?:ANSWER|INCOMPLETE|FINAL)\s*:\s*)?\[approval:denied\]/iu
const PERMANENT_ERROR_CODE_PATTERN = /^\[error:\s*([A-Z0-9_]+_PERMANENT)\]/iu
const REPORTED_PERMANENT_ERROR_CODE_PATTERN =
  /(?:^|\r?\n)\s*INCOMPLETE\s*:[^\r\n]{0,512}\[error:\s*([A-Z0-9_]+_PERMANENT)\]/iu
const ACTION_COMPLETION_RECOVERY_MARKER = '[Action completion guard]'
const TERSE_RESULT_FOLLOW_UP_PATTERN =
  /^(?:(?:그래서|그럼)\s*)?(?:확인\s*(?:결과|했어|했나요|했니)?|결과|됐어|되었어|끝났어|완료됐어|찾았어)(?:는|이|가)?(?:\s*(?:뭐야|어때|알려줘))?[?？.!]?\s*$|^(?:so\s+)?(?:what\s+did\s+you\s+find|what(?:'s|\s+is)\s+the\s+result|any\s+results?|did\s+(?:it|you)\s+(?:finish|work)|is\s+it\s+done|are\s+you\s+done|finished|done)[?!.]?\s*$/iu

function currentTurnUserMessageIndex(messages: readonly Message[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (
      message?.role === 'user' &&
      message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true
    ) {
      return index
    }
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') return index
  }
  return -1
}

function directiveText(input: string): string {
  const actionInput = (() => {
    if (!input.startsWith(SWARM_GOAL_PREFIX)) return input
    const contractIndex = input.indexOf(SWARM_OPERATIONAL_CONTRACT_SEPARATOR)
    if (contractIndex < SWARM_GOAL_PREFIX.length) return input
    return input.slice(SWARM_GOAL_PREFIX.length, contractIndex)
  })()
  const preserveQuotedFileTarget = (quoted: string): string => {
    const candidate = quoted.trim()
    if (
      !candidate ||
      !QUOTED_FILE_TARGET_PATTERN.test(candidate) ||
      inputExpressesMutationIntent(candidate)
    ) {
      return ' '
    }
    return ` ${candidate} `
  }

  return actionInput
    .normalize('NFKC')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`\r\n]*)`/g, (_match, quoted: string) => preserveQuotedFileTarget(quoted))
    .replace(/["“]([^"”\r\n]*)["”]/g, (_match, quoted: string) => preserveQuotedFileTarget(quoted))
    .replace(/['‘]([^'’\r\n]*)['’]/g, (_match, quoted: string) => preserveQuotedFileTarget(quoted))
    .trim()
}

function isGroundedWorkspaceFileQuery(directive: string): boolean {
  return (
    (
      GROUNDED_WORKSPACE_FILE_LOOKUP_PATTERN.test(directive)
      || FILE_INVENTORY_REQUEST_PATTERN.test(directive)
    )
    && WORKSPACE_FILE_COLLECTION_NOUN_PATTERN.test(directive)
    && (
      WORKSPACE_FILE_OWNERSHIP_PATTERN.test(directive)
      || COMPLETED_WORKSPACE_FILE_ACTION_PATTERN.test(directive)
    )
  )
}

function isExplicitFileMutationRequest(input: string): boolean {
  const directive = directiveText(input)
  if (!directive || NEGATED_FILE_ACTION_PATTERN.test(directive)) return false
  if (isGroundedWorkspaceFileQuery(directive)) return false
  const hasExplicitPath = EXPLICIT_FILE_PATH_TARGET_PATTERN.test(directive)
  const hasMutationIntent =
    inputExpressesMutationIntent(directive) ||
    ADDITIONAL_FILE_MUTATION_INTENT_PATTERN.test(directive)
  if (!hasMutationIntent || !FILE_SYSTEM_TARGET_PATTERN.test(directive)) {
    return false
  }
  if (EXPLICIT_FILE_ACTION_PATTERN.test(directive)) return true
  // Broad nouns such as "project" and broad intent words such as
  // "development" can coexist in an inspection sentence without authorizing
  // a mutation (for example, a review of development activity). When there is
  // no concrete file path, require an actual imperative/action shape above.
  // A named path retains the wider intent detector for natural requests whose
  // wording does not fit the compact imperative grammar.
  if (!hasExplicitPath) return false
  return !ACTION_META_QUESTION_PATTERN.test(directive) && !/[?？]\s*$/u.test(directive)
}

function isExplicitCommandExecutionRequest(input: string): boolean {
  const directive = directiveText(input)
  if (!directive || NEGATED_COMMAND_ACTION_PATTERN.test(directive)) return false
  if (!EXPLICIT_COMMAND_ACTION_PATTERN.test(directive)) return false
  return (
    !ACTION_META_QUESTION_PATTERN.test(directive) ||
    /\b(?:can|could|would)\s+you\b|(?:해\s*봐|해\s*줘|해주세요|해주십시오|돌려\s*줘|돌려봐)/iu.test(
      directive,
    )
  )
}

/**
 * A single, explicitly named managed-process observation is its own evidence
 * class. It is neither a shell command nor a file read, even when the user
 * also says “do not edit files” and asks to “read the logs”.
 */
export function inputRequestsFocusedSingleProcessObservation(input: string): boolean {
  const matches = input.match(PROCESS_OBSERVATION_TOOL_PATTERN) ?? []
  return matches.length === 1
    && input.length <= 900
    && FOCUSED_SINGLE_PROCESS_OBSERVATION_BOUNDARY_PATTERN.test(input)
    && PROCESS_OBSERVATION_ACTION_PATTERN.test(input)
    && !PROCESS_OBSERVATION_META_PATTERN.test(input)
    && !inputExplicitlyRequestsRenderedUiValidation(input)
    && !NON_PROCESS_TOOL_REFERENCE_PATTERN.test(input)
}

export interface FocusedProcessObservationCall {
  toolName: 'process.follow' | 'process.list' | 'process.read' | 'process.sessions' | 'process.wait'
  arguments: Record<string, unknown>
}

/**
 * Extracts an executable single-process observation only when the request
 * contains every required identity field. This lets provider-neutral runtimes
 * honor exact one-call contracts without depending on a model to reproduce a
 * UUID. Compound workflows and API/how-to prose are rejected by the focused
 * boundary above and remain model-driven.
 */
export function extractFocusedProcessObservationCall(
  input: string,
): FocusedProcessObservationCall | undefined {
  if (!inputRequestsFocusedSingleProcessObservation(input)) return undefined
  const toolName = input.match(PROCESS_OBSERVATION_TOOL_PATTERN)?.[0]?.toLowerCase() as
    | FocusedProcessObservationCall['toolName']
    | undefined
  if (!toolName) return undefined

  const arguments_: Record<string, unknown> = {}
  if (toolName !== 'process.sessions' && toolName !== 'process.list') {
    const id = /\b(?:process\s+)?id\s*(?:=|:)?\s*[`"']?([A-Za-z0-9][A-Za-z0-9._:-]{2,127})/iu.exec(input)?.[1]
      ?? /\bid\s*=\s*[`"']([A-Za-z0-9][A-Za-z0-9._:-]{2,127})[`"']/iu.exec(input)?.[1]
    if (!id) return undefined
    arguments_.id = id
  }

  for (const field of ['timeoutMs', 'stdoutOffset', 'stderrOffset'] as const) {
    const raw = new RegExp(`\\b${field}\\s*=\\s*(\\d{1,10})\\b`, 'iu').exec(input)?.[1]
    if (raw !== undefined) arguments_[field] = Number(raw)
  }
  return { toolName, arguments: arguments_ }
}

function inputRequestsProcessObservation(input: string): boolean {
  return (input.match(PROCESS_OBSERVATION_TOOL_PATTERN)?.length ?? 0) > 0
    && PROCESS_OBSERVATION_ACTION_PATTERN.test(input)
    && !PROCESS_OBSERVATION_META_PATTERN.test(input)
}

function isExplicitFileReadRequest(input: string): boolean {
  const directive = directiveText(input)
  if (!directive || NEGATED_FILE_READ_PATTERN.test(directive)) return false

  const hasExplicitPath = EXPLICIT_FILE_PATH_TARGET_PATTERN.test(directive)
  if (hasExplicitPath && FILE_CONTENT_QUESTION_PATTERN.test(directive)) {
    return true
  }

  if (
    !EXPLICIT_FILE_READ_ACTION_PATTERN.test(directive) ||
    (
      !hasExplicitPath
      && !DIRECT_ENGLISH_FILE_READ_TARGET_PATTERN.test(directive)
      && !DIRECT_KOREAN_FILE_READ_TARGET_PATTERN.test(directive)
    )
  ) {
    return false
  }

  // Keep general how-to questions informational even if they mention reading
  // a file/document. Direct imperatives and polite requests still require
  // grounded workspace evidence.
  if (
    ACTION_META_QUESTION_PATTERN.test(directive) &&
    !/\b(?:can|could|would)\s+you\b|(?:읽어|확인해|살펴|검토해|분석해|요약해|열어)\s*(?:줘|주세요|봐|줄래)?/iu.test(
      directive,
    )
  ) {
    return false
  }
  return true
}

function isExplicitFileInventoryRequest(input: string): boolean {
  const directive = directiveText(input)
  if (inputRequestsNonRenderedSurfaceInventory(directive)) return false
  const hasLocalDirectoryInventoryClause = directive
    .split(/[.!?。！？;；\n]+/u)
    .some((clause) => (
      WORKSPACE_DIRECTORY_TARGET_PATTERN.test(clause)
      && FILE_INVENTORY_REQUEST_PATTERN.test(clause)
      && (
        LOCAL_DIRECTORY_CONTAINER_PATTERN.test(clause)
        || WORKSPACE_INVENTORY_CONTENT_PATTERN.test(clause)
      )
    ))
  return Boolean(
    directive &&
    (
      hasLocalDirectoryInventoryClause
      || isGroundedWorkspaceFileQuery(directive)
    ),
  )
}

/**
 * Tool names are dotted (`system.info`, `memory.search`), which is exactly the
 * shape the file-target pattern reads as `name.ext`. Naming the tool you want
 * used — "system.info 도구로 측정해서 보고해줘" — therefore looked like a
 * request to touch a file called `system.info`, and the completion gate then
 * blocked a pure reporting turn for having produced no file. Blank the tool
 * names out before matching: a token that IS an available tool is not a path.
 */
function withoutToolNames(input: string, availableToolNames: readonly string[] | undefined): string {
  // A user can explicitly prohibit a tool that is intentionally absent from
  // this run's scoped tool list (for example, `fs.*` or `terminal.run` in a
  // process-only workflow). Those references remain tool identifiers, not
  // filenames. Strip the stable built-in tool namespace grammar before the
  // exact available-name pass so capability scoping cannot change intent
  // classification. Ordinary artifacts such as README.md stay untouched.
  let text = input.replace(
    STRUCTURED_TOOL_REFERENCE_PATTERN,
    (reference) => TOOL_SHAPED_ARTIFACT_SUFFIX_PATTERN.test(reference)
      ? reference
      : ' 도구 ',
  )
  if (!availableToolNames?.length) return text
  // Longest first so `memory.documents.search` is not left as `.search`.
  for (const name of [...availableToolNames].sort((a, b) => b.length - a.length)) {
    if (!name.includes('.')) continue
    text = text.split(name).join(' 도구 ')
  }
  return text
}

/**
 * A bare integration hostname has the same dotted shape as a filename. Mask
 * only host-shaped tokens with nearby network/search context, leaving real
 * dotted files (including multi-dot test files) available to the file guard.
 */
function withoutNetworkHostReferences(input: string): string {
  return input.replace(
    BARE_NETWORK_HOST_PATTERN,
    (hostname: string, offset: number, source: string) => {
      const before = source.slice(Math.max(0, offset - 72), offset)
      const after = source.slice(offset + hostname.length, offset + hostname.length + 72)
      return NETWORK_HOST_CONTEXT_PATTERN.test(`${before} ${after}`)
        ? ' network-host '
        : hostname
    },
  )
}

export function requiredActionEvidenceKinds(
  input: string,
  availableToolNames?: readonly string[],
): ActionEvidenceKind[] {
  // Capability scoping already removes bounded file-mutation prohibitions.
  // Applying the older mutation-only stripper first can leave malformed
  // fragments (for example, `process.read ... 사용하지 말고`) that later look
  // like affirmative observation requests.
  const positiveInput = inputPositiveCapabilityScope(input)
  const constrainedProcessStart = extractConstrainedProcessStart(positiveInput)
  // Exact argv/process contracts are command actions. Their structural fields
  // (`network.mode`, `network.ports`, executable paths) look file-shaped to a
  // generic path detector but do not authorize or require file mutation.
  if (
    extractFocusedProcessStart(positiveInput)
    || extractFocusedTerminalCommand(positiveInput)
  ) {
    return ['command']
  }
  if (
    !constrainedProcessStart
    && inputRequestsFocusedSingleProcessObservation(positiveInput)
  ) {
    return ['process-observation']
  }
  const renderedUiInput = positiveInput.replace(
    SEARCH_RETRIEVAL_LABEL_PATTERN,
    ' search-evidence ',
  )
  const browserActionText = withoutToolNames(
    renderedUiInput,
    [...BROWSER_EXECUTION_TOOL_NAMES],
  )
  const requiresBrowserEvidence = inputExplicitlyRequestsRenderedUiValidation(renderedUiInput)
    && EXPLICIT_BROWSER_OBSERVATION_ACTION_PATTERN.test(browserActionText)
  const text = withoutNetworkHostReferences(
    withoutToolNames(positiveInput, availableToolNames),
  )
  const kinds: ActionEvidenceKind[] = []
  if (inputRequestsProcessObservation(positiveInput)) kinds.push('process-observation')
  // A structured process.start embedded in a wider workflow still requires
  // command evidence. It must not disappear merely because a later
  // process.read/browser step prevents the narrower focused extractor from
  // owning the whole turn.
  if (constrainedProcessStart) kinds.push('command')
  const pathlessFileMutationForbidden = inputExplicitlyForbidsFileMutation(input)
    && !EXPLICIT_FILE_PATH_TARGET_PATTERN.test(text)
  if (!pathlessFileMutationForbidden && isExplicitFileMutationRequest(text)) kinds.push('file')
  if (isExplicitFileReadRequest(text)) kinds.push('file-read')
  if (isExplicitFileInventoryRequest(text)) kinds.push('file-inventory')
  if (isExplicitCommandExecutionRequest(text)) kinds.push('command')
  if (requiresBrowserEvidence) kinds.push('browser')
  if (isVisibleUrlOpenRequest(positiveInput)) kinds.push('visible-url')
  return kinds
}

function toolEvidenceKinds(toolName: string | undefined): readonly ActionEvidenceKind[] {
  if (!toolName) return []
  if (FILE_MUTATION_TOOL_NAMES.has(toolName)) return ['file']
  if (FILE_READ_TOOL_NAMES.has(toolName)) return ['file-read']
  if (FILE_INVENTORY_TOOL_NAMES.has(toolName)) return ['file-inventory']
  if (COMMAND_EXECUTION_TOOL_NAMES.has(toolName)) return ['command']
  if (PROCESS_OBSERVATION_TOOL_NAMES.has(toolName)) return ['process-observation']
  if (BROWSER_EXECUTION_TOOL_NAMES.has(toolName)) return ['browser']
  if (VISIBLE_URL_EXECUTION_TOOL_NAMES.has(toolName)) return ['visible-url']
  return []
}

function unresolvedPreviousActionKinds(
  messages: readonly Message[],
  userInput: string,
): ActionEvidenceKind[] {
  if (!TERSE_RESULT_FOLLOW_UP_PATTERN.test(userInput.trim())) return []
  const currentUserIndex = currentTurnUserMessageIndex(messages)
  if (currentUserIndex <= 0) return []

  let previousUserIndex = -1
  for (let index = currentUserIndex - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') {
      previousUserIndex = index
      break
    }
  }
  if (previousUserIndex < 0) return []
  const previousInput = String(messages[previousUserIndex]?.content ?? '')
  const requiredKinds = requiredActionEvidenceKinds(previousInput)
  if (requiredKinds.length === 0) return []

  const priorTurnMessages = messages.slice(previousUserIndex + 1, currentUserIndex)
  const toolNamesById = new Map<string, string>()
  for (const message of priorTurnMessages) {
    for (const toolCall of message.toolCalls ?? []) {
      toolNamesById.set(toolCall.id, toolCall.name)
    }
  }
  const successfulKinds = new Set<ActionEvidenceKind>()
  for (const message of priorTurnMessages) {
    if (
      message.role !== 'tool'
      || message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] !== 'success'
    ) {
      continue
    }
    const kinds = toolEvidenceKinds(
      message.name ?? (message.toolCallId ? toolNamesById.get(message.toolCallId) : undefined),
    )
    for (const kind of kinds) successfulKinds.add(kind)
  }
  return requiredKinds.filter((kind) => !successfulKinds.has(kind))
}

/**
 * Resolves the minimum action evidence required for the current turn. Terse
 * result follow-ups inherit only still-missing kinds from the immediately
 * preceding action request. For owned/recent artifact lookups, file inventory
 * is minimum workspace grounding; it does not by itself prove authorship or
 * recency, which remains the model's responsibility to qualify.
 */
export function requiredActionEvidenceKindsForTurn(options: {
  messages: readonly Message[]
  userInput: string
  availableToolNames?: readonly string[]
  additionalRequiredKinds?: readonly ActionEvidenceKind[]
  /** Semantic contract review owns actions when wording inference is disabled. */
  inferActionsFromWording?: boolean
}): ActionEvidenceKind[] {
  if (options.inferActionsFromWording === false) return [...new Set(options.additionalRequiredKinds ?? [])]
  const directRequiredKinds = requiredActionEvidenceKinds(
    options.userInput,
    options.availableToolNames,
  )
  const inferredKinds = directRequiredKinds.length > 0
    ? directRequiredKinds
    : unresolvedPreviousActionKinds(options.messages, options.userInput)
  return [...new Set([
    ...inferredKinds,
    ...(options.additionalRequiredKinds ?? []),
  ])]
}

export function evaluateActionCompletion(options: {
  messages: readonly Message[]
  content: string
  userInput: string
  availableToolNames?: readonly string[]
  /**
   * Trusted runtime tools eligible for generic command evidence. Callers build
   * this from an operational run contract and/or canonical tool-security
   * metadata, so this guard does not need every plugin or integration name.
   * Specific file/browser/process classes still require their canonical tools.
   */
  commandEvidenceToolNames?: readonly string[]
  additionalRequiredKinds?: readonly ActionEvidenceKind[]
  inferActionsFromWording?: boolean
}): ActionCompletionEvaluation {
  const requiredKinds = requiredActionEvidenceKindsForTurn(options)
  if (requiredKinds.length === 0) {
    return {
      requiredKinds,
      missingKinds: [],
      outcome: 'not_required',
      hasUnexecutedToolText: false,
      hasAvailableEvidenceTool: false,
    }
  }

  const latestUserIndex = currentTurnUserMessageIndex(options.messages)
  const currentTurnMessages =
    latestUserIndex >= 0 ? options.messages.slice(latestUserIndex + 1) : options.messages
  const toolNamesById = new Map<string, string>()
  for (const message of currentTurnMessages) {
    for (const toolCall of message.toolCalls ?? []) {
      toolNamesById.set(toolCall.id, toolCall.name)
    }
  }

  const attemptedKinds = new Set<ActionEvidenceKind>()
  const successfulKinds = new Set<ActionEvidenceKind>()
  const commandEvidenceToolNames = new Set(options.commandEvidenceToolNames ?? [])
  for (const message of currentTurnMessages) {
    if (message.role !== 'tool') continue
    const toolName = message.name
      ?? (message.toolCallId ? toolNamesById.get(message.toolCallId) : undefined)
    const kinds = [
      ...toolEvidenceKinds(toolName),
      ...(toolName && commandEvidenceToolNames.has(toolName) ? ['command' as const] : []),
    ]
    const relevantKinds = kinds.filter((kind) => requiredKinds.includes(kind))
    if (relevantKinds.length === 0) continue
    for (const kind of relevantKinds) attemptedKinds.add(kind)
    if (message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] === 'success') {
      for (const kind of relevantKinds) successfulKinds.add(kind)
    }
  }

  const missingKinds = requiredKinds.filter((kind) => !successfulKinds.has(kind))
  const availableToolNames = new Set(options.availableToolNames ?? [])
  const hasAvailableEvidenceTool = missingKinds.every((kind) =>
    kind === 'file'
      ? [...FILE_MUTATION_TOOL_NAMES].some((name) => availableToolNames.has(name))
      : kind === 'file-read'
        ? [...FILE_READ_TOOL_NAMES].some((name) => availableToolNames.has(name))
        : kind === 'file-inventory'
          ? [...FILE_INVENTORY_TOOL_NAMES].some((name) => availableToolNames.has(name))
          : kind === 'browser'
            ? [...BROWSER_EXECUTION_TOOL_NAMES].some((name) => availableToolNames.has(name))
            : kind === 'process-observation'
              ? [...PROCESS_OBSERVATION_TOOL_NAMES].some((name) => availableToolNames.has(name))
              : kind === 'visible-url'
                ? [...VISIBLE_URL_EXECUTION_TOOL_NAMES].some((name) => availableToolNames.has(name))
              : [...COMMAND_EXECUTION_TOOL_NAMES, ...commandEvidenceToolNames]
                  .some((name) => availableToolNames.has(name)),
  )
  const hasFailedAttempt = missingKinds.some((kind) => attemptedKinds.has(kind))
  const hasUnexecutedToolText =
    UNEXECUTED_TOOL_TEXT_PATTERN.test(options.content) ||
    containsPromptToolCallEnvelope(options.content)
  const latestRelevantToolMessage = [...currentTurnMessages].reverse().find((message) => {
    if (message.role !== 'tool') return false
    const toolName = message.name
      ?? (message.toolCallId ? toolNamesById.get(message.toolCallId) : undefined)
    const kinds = [
      ...toolEvidenceKinds(toolName),
      ...(toolName && commandEvidenceToolNames.has(toolName) ? ['command' as const] : []),
    ]
    return kinds.some((kind) => requiredKinds.includes(kind))
  })
  const hasReportedApprovalDenial =
    latestRelevantToolMessage?.metadata?.[APPROVAL_FAILURE_STATUS_METADATA_KEY] === 'denied' &&
    /^\s*\[approval:denied\]/iu.test(String(latestRelevantToolMessage.content)) &&
    REPORTED_APPROVAL_DENIAL_PATTERN.test(options.content)
  const permanentToolCode = PERMANENT_ERROR_CODE_PATTERN
    .exec(String(latestRelevantToolMessage?.content ?? '').trim())?.[1]?.toUpperCase()
  const reportedPermanentCode = REPORTED_PERMANENT_ERROR_CODE_PATTERN
    .exec(options.content)?.[1]?.toUpperCase()
  const hasReportedPermanentFailure = Boolean(
    permanentToolCode && reportedPermanentCode && permanentToolCode === reportedPermanentCode,
  )
  const latestFailureSummary = latestRelevantToolMessage?.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] === 'error'
    ? String(latestRelevantToolMessage.content ?? '')
        .trim()
        .split(/\r?\n/u, 1)[0]
        ?.slice(0, 360)
    : undefined

  return {
    requiredKinds,
    missingKinds,
    outcome:
      missingKinds.length === 0
        ? 'success'
        : hasReportedApprovalDenial || hasReportedPermanentFailure
          ? 'blocked'
          : hasFailedAttempt
            ? 'failed'
            : 'missing',
    hasUnexecutedToolText,
    hasAvailableEvidenceTool,
    ...(latestFailureSummary ? { lastFailureSummary: latestFailureSummary } : {}),
  }
}

type FailureLanguage = 'ko' | 'en' | 'zh' | 'ja'

function failureLanguage(input: string): FailureLanguage {
  if (/[가-힣]/u.test(input)) return 'ko'
  if (/[ぁ-ゟ゠-ヿ]/u.test(input)) return 'ja'
  if (/\p{Script=Han}/u.test(input)) return 'zh'
  return 'en'
}

export function actionCompletionFailureReason(
  evaluation: ActionCompletionEvaluation,
): string | null {
  if (
    evaluation.outcome === 'not_required' ||
    evaluation.outcome === 'success' ||
    evaluation.outcome === 'blocked'
  )
    return null
  const kinds = evaluation.missingKinds.join(' and ')
  if (evaluation.hasUnexecutedToolText) {
    return `the response contains unexecuted tool-call text and has no successful ${kinds} tool result`
  }
  if (evaluation.outcome === 'failed') {
    return `the requested ${kinds} action has no successful tool result after a failed attempt`
  }
  return `the requested ${kinds} action has no successful tool result`
}

export function countActionCompletionRecoveryPrompts(messages: readonly Message[]): number {
  return messages.filter((message) =>
    message.role === 'system' &&
    typeof message.content === 'string' &&
    message.content.includes(ACTION_COMPLETION_RECOVERY_MARKER),
  ).length
}

export function buildActionCompletionRecoveryMessage(
  evaluation: ActionCompletionEvaluation,
): Message {
  const instructions = evaluation.missingKinds.map((kind) => {
    if (kind === 'file-inventory') {
      return 'Call fs.list exactly once with an empty arguments object so it lists the active session cwd. Do not use fs.glob, fs.search, fs.read, git, terminal.run, or a write/edit probe for this directory-listing request.'
    }
    if (kind === 'file-read') {
      return 'Call an available structured read tool for the concrete target before answering: use office.read_slide for a PowerPoint slide and fs.read for an ordinary text file.'
    }
    if (kind === 'file') {
      return 'Call the appropriate fs.write/fs.append/fs.edit/apply_patch tool and use its actual result.'
    }
    if (kind === 'browser') {
      return 'Call the explicitly requested browser.navigate/browser.extract/browser.click/browser.evaluate/browser.screenshot tool and use its actual current-turn result.'
    }
    if (kind === 'visible-url') {
      return 'Call computer.open_url with the selected HTTP(S) destination and use its actual current-turn result. If approval is required, wait for the user decision; do not claim that a browser was opened before the tool succeeds.'
    }
    if (kind === 'process-observation') {
      return 'Call the explicitly requested process.read/process.follow/process.wait/process.list/process.sessions tool exactly once and use its actual current-turn result.'
    }
    return 'Call the appropriate terminal.run/process.start/process.stop/service.start/service.stop/computer.launch_app tool and use its actual result.'
  })
  return {
    role: 'system',
    content: [
      ACTION_COMPLETION_RECOVERY_MARKER,
      'The previous answer had no successful structured tool result for the requested action.',
      evaluation.hasUnexecutedToolText
        ? 'The prior tool_call:/terminal.run: prose was not executed. Emit a real structured tool call using an exact available tool name.'
        : '',
      ...instructions,
      'Return a normal final answer only after that tool succeeds. Do not print or simulate tool-call text in prose.',
    ].join(' '),
  }
}

export function buildActionCompletionFailureOutput(
  evaluation: ActionCompletionEvaluation,
  userInput: string,
): string {
  const language = failureLanguage(userInput)
  const unavailable = evaluation.outcome === 'missing' && !evaluation.hasAvailableEvidenceTool
  const unsafe = evaluation.hasUnexecutedToolText
  switch (language) {
    case 'ko':
      return [
        'INCOMPLETE: 요청한 파일, 명령 또는 브라우저 작업에 성공한 실제 도구 결과가 없어 완료됐다고 확인할 수 없습니다.',
        unsafe ? '응답에 적힌 도구 호출 문자열은 실행된 호출로 취급하지 않았습니다.' : '',
        evaluation.lastFailureSummary
          ? `마지막 실제 도구 실패: ${evaluation.lastFailureSummary}`
          : '',
        unavailable
          ? '필요한 도구가 현재 실행에 제공되지 않았습니다. 도구 사용이나 실행 모드 변경은 사용자 확인 후 다시 시도해야 합니다.'
          : '',
      ]
        .filter(Boolean)
        .join(' ')
    case 'zh':
      return 'INCOMPLETE: 没有成功的实际工具结果，因此无法确认所请求的文件、命令或浏览器操作已完成。响应中的工具调用文本不视为已执行。'
    case 'ja':
      return 'INCOMPLETE: 成功した実際のツール結果がないため、依頼されたファイル、コマンド、またはブラウザ操作の完了を確認できません。応答内のツール呼び出し文字列は実行済みとは扱いません。'
    default:
      return [
        'INCOMPLETE: The requested file, command, or browser action has no successful, actual tool result, so completion cannot be confirmed.',
        unsafe ? 'Tool-call text in the response was not treated as an executed call.' : '',
        evaluation.lastFailureSummary
          ? `Last actual tool failure: ${evaluation.lastFailureSummary}`
          : '',
        unavailable
          ? 'The required tool was not available in this run; tool access or a mode change must be confirmed by the user before retrying.'
          : '',
      ]
        .filter(Boolean)
        .join(' ')
  }
}
