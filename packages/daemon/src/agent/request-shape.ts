/**
 * Small deterministic request-shape checks shared by routing and contract
 * generation. They describe ordinary user intent, not a graph or benchmark.
 */

import type {
  AgentRequestedProcessStart,
  AgentRequestedTerminalCommand,
} from '@sepilotd/core'

const REPOSITORY_CHANGE_TOPIC_RE =
  /\b(?:git\s+(?:log|status|diff)|(?:latest|last|recent|current)\s+(?:(?:\d+|few)\s+)?(?:commits?|changes?|modifications?|edits?|diffs?|worktree|working\s+tree))\b|(?:(?:가장\s*)?(?:최근|최신|현재)(?:의|에|까지|부터|\s)*(?:(?:\d+|몇)\s*(?:개|건)?\s*)?(?:커밋|수정(?:한|된)?\s*(?:내용|사항|내역)?|변경(?:한|된)?\s*(?:내용|사항|내역)?|차이|diff|작업\s*트리))/iu

const REPOSITORY_SUMMARY_ACTION_RE =
  /\b(?:check|explain|describe|introduce|summari[sz]e|show|list|tell)\b|(?:확인|설명|소개|요약|보여|알려|정리)(?:해|해줘|해주세요|해봐|하자|할래|해라|하세요)?/iu
const REPOSITORY_SUBSTANTIVE_REVIEW_ACTION_RE =
  /\b(?:analy[sz]e|review|inspect|audit|assess|evaluate|investigate)\b|(?:분석|리뷰|검토|점검|감사|평가|조사)(?:해|해줘|해주세요|해봐|하자|할래|해라|하세요)?/iu

// Korean mutation words require an imperative/continuative ending so the noun
// phrase "최근 수정한 내용" is not mistaken for the command "수정해줘".
const EXPLICIT_REPOSITORY_MUTATION_RE =
  /\b(?:apply|cherry-pick|revert|reset|rebase|squash|merge|fix|modify|edit|update|implement|write|create|delete|remove)\b|(?:고쳐|바꿔|적용해|되돌려|삭제해|지워|구현해|작성해|생성해|수정해|변경해|개선해|병합해|리베이스해|스쿼시해)(?:\s*(?:줘|주세요|봐|라|요))?/iu
const ENGLISH_NEGATED_REPOSITORY_MUTATION_CLAUSE_RE =
  /\b(?:do\s+not|don['’]t|dont|never)\s+(?:directly\s+)?(?:apply|cherry-pick|revert|reset|rebase|squash|merge|fix|modify|edit|update|implement|write|create|delete|remove)\b(?:(?!\b(?:but|however|instead)\b)[^.!?;\n]){0,240}/giu
const ENGLISH_WITHOUT_REPOSITORY_MUTATION_CLAUSE_RE =
  /\bwithout\s+(?:directly\s+)?(?:applying|cherry-picking|reverting|resetting|rebasing|squashing|merging|fixing|modifying|editing|updating|implementing|writing|creating|deleting|removing)\b(?:(?!\b(?:but|however|instead)\b)[^.!?;\n]){0,240}/giu
const EXPLICIT_ALTERNATE_EXECUTION_SURFACE_RE =
  /\bterminal\.run\b|\b(?:use|using|via|with|through|in)\s+(?:the\s+)?(?:terminal|shell|bash|powershell|cmd\.exe|curl)\b|(?:터미널|쉘|셸|명령줄)(?:로|에서|을|를)/iu
const EXPLICIT_MANAGED_PROCESS_SURFACE_RE = /\bprocess\.start\b/iu
const EXPLICIT_TOOL_SURFACE_RE =
  /\b(?:process\.[a-z][\w.-]*|terminal\.run|browser\.[a-z][\w.-]*|service\.[a-z][\w.-]*|schedule_[a-z][\w-]*|(?:web|jpad|monitor|notification|fs|memory|system|workspace|git|code|doc|notebook|office|apps|media|computer|market|pages|image_gen|device|usage|swarm|subagent|self|skillhub|external_acp|a2a)\.[a-z][\w.-]*)\b/giu
const REPOSITORY_DIFF_TOPIC_RE =
  /\b(?:git\s+diff|(?:latest|last|recent|current)\s+(?:changes?|modifications?|edits?|diffs?|worktree|working\s+tree))\b|(?:(?:가장\s*)?(?:최근|최신|현재)(?:의|에|까지|부터|\s)*(?:수정|변경|차이|diff|작업\s*트리))/iu
const FILE_PATH_TOKEN_RE = /[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,10}/gu
const ENGLISH_LITERAL_REPLACEMENT_RE =
  /\b(?:replace|change)\s+[^\n]{1,120}?\s+(?:with|to)\s+[^\n]{1,120}/iu
const KOREAN_LITERAL_REPLACEMENT_RE =
  /(?:을|를)\s*[^\n]{1,120}?(?:으?로)\s*(?:바꾸|바꿔|교체|변경)/u
const BROAD_REPLACEMENT_SCOPE_RE =
  /\b(?:all|every)\s+(?:files?|occurrences?|references?|packages?|workspace|repository|codebase)\b|(?:모든|전체|전부)\s*(?:파일|참조|패키지|저장소|코드베이스)/iu
const ADDITIONAL_MUTATION_RE =
  /\band\s+(?:also\s+)?(?:add|delete|fix|implement|refactor|remove|test|update)\b|그리고\s*(?:고쳐|구현|리팩터|삭제|수정|추가|테스트)/iu
const EXPLICIT_NO_TOOL_RE =
  /\b(?:do\s+not|don['’]t|never)\s+(?:use|run|call|invoke|execute)\s+(?:any\s+)?(?:tools?|commands?)\b|\b(?:do\s+not|don['’]t|never)\s+(?:use|run|call|invoke|execute)\s+[^\n]{1,120}?\b(?:and|or)\s+(?:any\s+)?other\s+(?:tools?|commands?)\b|\bwithout\s+(?:using|running|calling|invoking|executing)\s+(?:any\s+)?(?:tools?|commands?)\b|\b(?:tool|command)\s+(?:use|calls?|execution)\s+(?:is|are)\s+(?:forbidden|not\s+allowed)\b|(?:도구|툴|명령)(?:를|은|는|도)?\s*(?:사용|실행|호출)하지\s*(?:마|말)|(?:도구|툴|명령)(?:를|은|는|도)?\s*(?:사용|실행|호출)(?:하)?거나[^\n]{0,80}하지\s*(?:마|말)|(?:도구|툴|명령)\s*(?:없이|금지)/iu
const SCOPED_OTHER_TOOL_PROHIBITION_RE =
  /(?:다른|그\s*외(?:의)?|추가(?:적인)?)\s*(?:도구|툴|명령)(?:를|은|는|도)?\s*(?:사용|실행|호출)하지\s*(?:마|말)(?:아|라|고|아줘|아주세요|세요)?|(?:이|위|앞서|해당)\s*(?:(?:\d+|한|두|세)\s*)?(?:[\p{L}\p{N}._/-]+\s*){0,4}외(?:의)?\s*(?:[\p{L}\p{N}._/-]+\s*){0,4}(?:도구|툴|명령)(?:를|은|는|도)?\s*(?:사용|실행|호출)하지\s*(?:마|말)(?:아|라|고|아줘|아주세요|세요)?|\b(?:do\s+not|don['’]t|never)\s+(?:use|run|call|invoke|execute)\s+(?:any\s+)?(?:other\s+)?(?:tools?|commands?)\s+(?:except|other\s+than)\s+(?:the\s+)?(?:named|specified|requested|above|preceding|these|those)\b[^.!?;\n]*/giu
const NAMED_TERMINAL_COMMAND_RE =
  /(?:`[^`\n]{1,300}`|\b(?:kubectl|helm|docker|podman|git|pnpm|npm|yarn|bun|cargo|go|python\d*|node|curl|systemctl|journalctl)\s+[^\n]{1,300})/iu
const NAMED_TERMINAL_EXECUTABLE_RE =
  /\b(?:kubectl|helm|docker|podman|git|pnpm|npm|yarn|bun|cargo|go|python\d*|node|curl|systemctl|journalctl)(?=\s+\S)/giu
const COMMAND_EXECUTION_ACTION_RE =
  /\b(?:run|execute|invoke)\b|(?:실행|돌려|호출)(?:해|해줘|해주세요|하라|하세요)?/iu
const COMMAND_EXECUTION_USE_ACTION_RE =
  /\b(?:run|execute|invoke|use|call)\b|(?:실행|돌려|호출|사용)(?:해|해줘|해주세요|하라|하세요)?/iu
const DIRECT_EXECUTABLE_CALL_RE =
  /\bdirect\s+[A-Za-z0-9_.-]{0,64}\s*(?:executable|command)(?:\s+calls?)?\b|(?:직접\s*)?(?:실행\s*파일|명령)(?:을|를)?\s*(?:직접\s*)?(?:실행|호출|사용)/iu
const SINGLE_COMMAND_SCOPE_RE =
  /\b(?:only|once|just)\b|(?:한\s*번|한번|결과만|명령만|만\s*실행)/iu
const TERMINAL_COMMAND_TRAILING_SCOPE_RE =
  /\s+(?:명령(?:을|를|만)?|(?:(?:정확히|딱)\s*)?(?:한\s*번|한번)|(?:한\s*번\s*)?실행(?:하고|해서|해|하라|하세요)?|결과(?:만|를|을)?|(?:(?:exactly|just|only)\s+)?once\b|(?:and|then)\s+(?:report|show|return|print|tell)\b)/iu
const RAW_COMMAND_OUTPUT_RE =
  /\b(?:output|stdout)\s+only\b|\b(?:show|report|return|print)\s+(?:only\s+)?(?:the\s+)?(?:actual\s+)?(?:output|stdout)\b|(?:실제\s*)?(?:출력|결과)(?:만|을\s*(?:그대로\s*)?(?:보여|알려|보고|출력))/iu
const EXPLICIT_CURRENT_DIRECTORY_BOUNDARY_RE =
  /\b(?:only\s+)?(?:within|inside|under)\s+(?:the\s+)?(?:current|this)\s+(?:working\s+)?director(?:y|ies)\b|\b(?:do\s+not|don't|never)\s+(?:modify|change|touch|write\s+to|create|delete)\s+(?:anything\s+)?(?:outside|beyond)\s+(?:the\s+)?(?:current|this)\s+(?:working\s+)?director(?:y|ies)\b|(?:현재|이)\s*(?:작업\s*)?디렉터리\s*(?:안|내|에서만)|(?:현재|이)\s*(?:작업\s*)?디렉터리\s*(?:밖|외부)(?:의|를|는|은)?\s*(?:파일이나?\s*디렉터리를?\s*)?(?:변경|수정|생성|삭제|건드리)하지\s*(?:마|말)/iu
const SMALL_SOURCE_AND_TEST_DELIVERABLE_RE =
  /\b(?:source|implementation|function|module|code)\b[^\n]{0,180}\b(?:unit\s+)?tests?\b|\b(?:unit\s+)?tests?\b[^\n]{0,180}\b(?:source|implementation|function|module|code)\b|(?:소스|구현|함수|모듈|코드)(?:와|과|및|,|\s)[^\n]{0,120}테스트|테스트[^\n]{0,120}(?:소스|구현|함수|모듈|코드)/iu
const BROAD_CODING_SCOPE_RE =
  /\b(?:entire|whole|all|every|across)\s+(?:repository|repo|codebase|workspace|project|packages?|services?|modules?|files?)\b|\b(?:architecture|migration|large[- ]scale|cross[- ]package|cross[- ]service|end[- ]to[- ]end|e2e|redesign|overhaul)\b|(?:저장소|코드베이스|워크스페이스|프로젝트)\s*(?:전체|전반)|(?:전체|모든|전부)\s*(?:패키지|서비스|모듈|파일)|(?:아키텍처|마이그레이션|대규모|전면|E2E|엔드투엔드)\s*(?:변경|개편|리팩터링|구현|작업)?/iu
const RENDERED_UI_CODING_SCOPE_RE =
  /\b(?:ui|ux|front[- ]?end|website|web\s+(?:app|page)|responsive|browser\s+ui)\b|(?:UI|UX|프론트엔드|웹\s*(?:앱|페이지)|반응형|브라우저\s*화면)/u

function terminalIntentClauses(message: string): string[] {
  const clauses: string[] = []
  let start = 0
  let inBackticks = false
  for (let index = 0; index < message.length; index += 1) {
    const char = message[index]
    if (char === '`') {
      inBackticks = !inBackticks
      continue
    }
    if (inBackticks) continue
    const previous = index > 0 ? message[index - 1] : ''
    const next = index + 1 < message.length ? message[index + 1] : ''
    const boundary = char === '\n'
      || char === ';'
      || char === '；'
      || char === '!'
      || char === '?'
      || char === '。'
      || char === '！'
      || char === '？'
      || (char === '.' && previous !== '.' && (!next || /\s/u.test(next)))
    if (!boundary) continue
    const clause = message.slice(start, index).trim()
    if (clause) clauses.push(clause)
    start = index + 1
  }
  const tail = message.slice(start).trim()
  if (tail) clauses.push(tail)
  return clauses
}

function focusedUnstructuredTerminalClause(message: string): string | undefined {
  const candidates = terminalIntentClauses(message).filter((clause) =>
    NAMED_TERMINAL_COMMAND_RE.test(clause)
    && namedTerminalCommandCount(clause) === 1
    && COMMAND_EXECUTION_ACTION_RE.test(clause)
    && SINGLE_COMMAND_SCOPE_RE.test(clause),
  )
  return candidates.length === 1 ? candidates[0] : undefined
}

function namedTerminalCommandCount(clause: string): number {
  const inlineCommands = [...clause.matchAll(/`[^`\n]{1,300}`/gu)]
  if (inlineCommands.length > 0) return inlineCommands.length
  return [...clause.matchAll(NAMED_TERMINAL_EXECUTABLE_RE)].length
}

function withoutNegatedRepositoryMutationClauses(message: string): string {
  return message
    .replace(ENGLISH_NEGATED_REPOSITORY_MUTATION_CLAUSE_RE, ' ')
    .replace(ENGLISH_WITHOUT_REPOSITORY_MUTATION_CLAUSE_RE, ' ')
}

/**
 * A bounded, read-only summary of recent repository state. Users naturally
 * say "최근 수정 사항" or "latest changes" when they want commit/diff/status
 * evidence, so this intentionally covers more than the literal word commit.
 * Substantive review verbs are excluded: code review, risk analysis, and audit
 * require source/context inspection beyond a mechanically truncated Git diff.
 */
export function isFocusedRepositoryChangeLookup(message: string): boolean {
  if (!message.trim() || !REPOSITORY_CHANGE_TOPIC_RE.test(message)) return false
  if (EXPLICIT_REPOSITORY_MUTATION_RE.test(
    withoutNegatedRepositoryMutationClauses(message),
  )) return false
  if (EXPLICIT_ALTERNATE_EXECUTION_SURFACE_RE.test(message)) return false
  if (isSubstantiveRepositoryChangeReview(message)) return false
  return REPOSITORY_SUMMARY_ACTION_RE.test(message)
    || /(?:몇\s*개|\b\d+\b|\?|？)/u.test(message)
}

/**
 * A read-only assessment of recent changes that needs both Git evidence and
 * source context. This is deliberately separate from the summary fast path:
 * it enables a compact review-oriented read tool profile without exposing
 * mutation tools or the entire general-purpose catalog.
 */
export function isSubstantiveRepositoryChangeReview(message: string): boolean {
  if (!message.trim() || !REPOSITORY_CHANGE_TOPIC_RE.test(message)) return false
  if (EXPLICIT_REPOSITORY_MUTATION_RE.test(
    withoutNegatedRepositoryMutationClauses(message),
  )) return false
  if (EXPLICIT_ALTERNATE_EXECUTION_SURFACE_RE.test(message)) return false
  return REPOSITORY_SUBSTANTIVE_REVIEW_ACTION_RE.test(message)
}

/** Whether a focused lookup asks about changed content rather than only commit metadata. */
export function focusedRepositoryLookupNeedsDiff(message: string): boolean {
  return isFocusedRepositoryChangeLookup(message) && REPOSITORY_DIFF_TOPIC_RE.test(message)
}

/**
 * An explicit per-turn prohibition on tool use. This is intentionally narrow:
 * "do not modify files" still permits read-only evidence tools, while "do not
 * use tools" removes the tool surface and disables evidence-recovery calls.
 */
export function explicitlyForbidsToolUse(message: string): boolean {
  const normalized = message.trim().replace(SCOPED_OTHER_TOOL_PROHIBITION_RE, '')
  return EXPLICIT_NO_TOOL_RE.test(normalized)
}

/** A user-requested single CLI invocation that needs no unrelated tool catalog. */
export function isFocusedTerminalCommandExecution(message: string): boolean {
  const normalized = message.trim()
  if (!normalized || normalized.length > 700) return false
  // An explicit managed-process surface carries lifecycle semantics that a
  // foreground terminal call cannot preserve.
  if (EXPLICIT_MANAGED_PROCESS_SURFACE_RE.test(normalized)) return false
  const structuredCommand = extractStructuredProcessCommand(normalized)
  const explicitToolSurfaces = normalized.match(EXPLICIT_TOOL_SURFACE_RE) ?? []
  const hasAdditionalToolSurface = explicitToolSurfaces.some(
    (surface) => surface.toLowerCase() !== 'terminal.run',
  )
  const structuredSingleTerminalCall = Boolean(structuredCommand)
    && /\bterminal\.run\b/iu.test(normalized)
    && !hasAdditionalToolSurface
    && COMMAND_EXECUTION_ACTION_RE.test(normalized)
    && SINGLE_COMMAND_SCOPE_RE.test(normalized)
  return structuredSingleTerminalCall
    || (
      !hasAdditionalToolSurface
      && focusedUnstructuredTerminalClause(normalized) !== undefined
    )
}

/**
 * Whether a request positively asks for terminal command execution, including
 * a multi-observation workflow that cannot be reduced to one exact argv.
 *
 * This is deliberately only a capability-shape signal. It does not extract an
 * executable, choose a tool trajectory, or grant mutation authority. Callers
 * must independently establish the workspace-mutation boundary.
 */
export function inputRequestsTerminalCommandExecution(message: string): boolean {
  const normalized = message.trim()
  if (!normalized || explicitlyForbidsToolUse(normalized)) return false
  return terminalIntentClauses(normalized).some((clause) => {
    const explicitTerminalTool = /\bterminal\.run\b/iu.test(clause)
      && COMMAND_EXECUTION_USE_ACTION_RE.test(clause)
    const namedExecutable = NAMED_TERMINAL_COMMAND_RE.test(clause)
      && (
        COMMAND_EXECUTION_USE_ACTION_RE.test(clause)
        || DIRECT_EXECUTABLE_CALL_RE.test(clause)
      )
    if (!explicitTerminalTool && !namedExecutable) return false
    return !/\b(?:do\s+not|don['’]t|dont|never|without)\b[^.!?;\n]{0,80}\b(?:run|execute|invoke|use|call)\b/iu.test(clause)
      && !/(?:실행|호출|사용)(?:은|는|을|를)?\s*(?:하지\s*(?:마|말)|제외|없이)/u.test(clause)
  })
}

function isFocusedManagedProcessStart(message: string): boolean {
  const normalized = message.trim()
  if (!normalized || normalized.length > 700) return false
  const structuredCommand = extractStructuredProcessCommand(normalized)
  const explicitToolSurfaces = normalized.match(EXPLICIT_TOOL_SURFACE_RE) ?? []
  const hasAdditionalToolStep = explicitToolSurfaces.some(
    (surface) => surface.toLowerCase() !== 'process.start',
  )
  const positiveMutationScope = normalized
    .replace(
      /\b(?:do\s+not|don't|dont|never)\s+(?:directly\s+)?(?:modify|edit|touch|change|write\s+to|delete|remove|mutate)\b/giu,
      ' ',
    )
    .replace(
      /(?:수정|편집|변경|건드리|고치|삭제|지우)(?:은|는|을|를)?\s*(?:하지\s*(?:마|말|않)|말고|없이|않(?:고|도록|는다|습니다))/giu,
      ' ',
    )
  return EXPLICIT_MANAGED_PROCESS_SURFACE_RE.test(normalized)
    && !hasAdditionalToolStep
    && !EXPLICIT_REPOSITORY_MUTATION_RE.test(positiveMutationScope)
    && (
      // Supplying process.start's executable + JSON argv is itself an exact,
      // single-tool authority grant. Requiring the user to additionally say
      // “once” made ordinary structured invocations fall through to semantic
      // planning, even though there was nothing for a model to infer. An
      // explicitly named second tool keeps multi-step workflows on the normal
      // graph instead of silently dropping later steps.
      Boolean(structuredCommand)
      || (
        SINGLE_COMMAND_SCOPE_RE.test(normalized)
        &&
        NAMED_TERMINAL_COMMAND_RE.test(normalized)
        && COMMAND_EXECUTION_ACTION_RE.test(normalized)
      )
    )
}

function splitLiteralTerminalArgv(command: string): string[] | undefined {
  const words: string[] = []
  let current = ''
  let quote: '"' | "'" | undefined
  let escaped = false
  let tokenStarted = false

  for (const char of command.trim()) {
    if (escaped) {
      current += char
      escaped = false
      tokenStarted = true
      continue
    }
    if (char === '\\' && quote !== "'") {
      escaped = true
      tokenStarted = true
      continue
    }
    if (quote) {
      if (char === quote) quote = undefined
      else current += char
      tokenStarted = true
      continue
    }
    if (char === '"' || char === "'") {
      quote = char
      tokenStarted = true
      continue
    }
    // terminal.run is argv based. Shell operators or substitutions cannot be
    // represented without changing their meaning, so leave those requests on
    // the normal model-driven path instead of pretending they are exact.
    if ('|;&><`$(){}'.includes(char)) return undefined
    if (/\s/u.test(char)) {
      if (tokenStarted) {
        words.push(current)
        current = ''
        tokenStarted = false
      }
      continue
    }
    current += char
    tokenStarted = true
  }
  if (escaped || quote) return undefined
  if (tokenStarted) words.push(current)
  return words.length > 0 ? words : undefined
}

/**
 * Extract an argv-safe command only from the conservative focused-command
 * shape above. Inline code supports arbitrary executable names; unquoted
 * prose keeps the existing common-CLI boundary so phrases such as "run tests
 * once" are not misread as an executable named `tests`.
 */
function extractFocusedCommandArgv(message: string): AgentRequestedTerminalCommand | undefined {
  const inline = message.match(/`([^`\n]{1,300})`/u)?.[1]?.trim()
  const named = inline ? undefined : message.match(NAMED_TERMINAL_COMMAND_RE)?.[0]?.trim()
  let literal = inline ?? named
  if (!literal) return undefined

  if (!inline) {
    // The command matcher intentionally accepts ordinary prose so users do
    // not have to use backticks. A punctuation mark followed by whitespace is
    // nevertheless a structural sentence boundary; version numbers, URLs,
    // dotted filenames, and decimal values have no whitespace after the dot.
    const sentenceBoundary = literal.search(/[.!?。！？](?=\s+\S)/u)
    if (sentenceBoundary >= 0) literal = literal.slice(0, sentenceBoundary).trim()
    const trailingScope = literal.search(TERMINAL_COMMAND_TRAILING_SCOPE_RE)
    if (trailingScope >= 0) literal = literal.slice(0, trailingScope).trim()
    // Korean focus particles can attach directly to the final argv token:
    // "npm test만 한 번 실행" means the literal command is `npm test`.
    literal = literal.replace(/([^\s])(?:만|을|를)$/u, '$1')
  }

  const argv = splitLiteralTerminalArgv(literal)
  if (!argv || argv.length === 0) return undefined
  const [executable, ...args] = argv
  if (!executable || executable.startsWith('-') || /[\/\\]$/u.test(executable)) return undefined
  return { executable, args }
}

function extractJsonStringArrayAfter(
  message: string,
  fieldPattern: RegExp,
): string[] | undefined {
  const match = fieldPattern.exec(message)
  if (!match) return undefined
  const start = match.index + match[0].length
  if (message[start] !== '[') return undefined

  let quote = false
  let escaped = false
  for (let index = start + 1; index < message.length; index += 1) {
    const char = message[index]
    if (escaped) {
      escaped = false
      continue
    }
    if (char === '\\' && quote) {
      escaped = true
      continue
    }
    if (char === '"') {
      quote = !quote
      continue
    }
    if (char !== ']' || quote) continue

    try {
      const parsed: unknown = JSON.parse(message.slice(start, index + 1))
      if (
        !Array.isArray(parsed)
        || parsed.length > 256
        || parsed.some((entry) => typeof entry !== 'string' || entry.length > 1_024)
      ) {
        return undefined
      }
      return parsed as string[]
    } catch {
      return undefined
    }
  }
  return undefined
}

/**
 * Parse the explicit structured shape shown by the process.start tool itself.
 * This avoids asking a model to reconstruct argv when the user already supplied
 * executable and args fields, while deliberately accepting only JSON strings
 * (not shell expressions or JavaScript-like values).
 */
function extractStructuredProcessCommand(
  message: string,
): AgentRequestedTerminalCommand | undefined {
  const executableMatch = message.match(
    /\bexecutable\s*(?:=|:)\s*(?:`([^`\r\n]+)`|"([^"\r\n]+)"|'([^'\r\n]+)'|([^\s,;]+))/iu,
  )
  const executable = executableMatch
    ? (executableMatch[1] ?? executableMatch[2] ?? executableMatch[3] ?? executableMatch[4])?.trim()
    : undefined
  if (
    !executable
    || executable.startsWith('-')
    || /[\/\\]$/u.test(executable)
    || /[|;&><`$(){}\s]/u.test(executable)
  ) {
    return undefined
  }

  const args = extractJsonStringArrayAfter(message, /\bargs\s*(?:=|:)\s*/iu)
  if (!args) return undefined
  return { executable, args }
}

export function extractFocusedTerminalCommand(
  message: string,
): AgentRequestedTerminalCommand | undefined {
  if (!isFocusedTerminalCommandExecution(message)) return undefined
  const structured = extractStructuredProcessCommand(message)
  if (structured) return structured
  const focusedClause = focusedUnstructuredTerminalClause(message)
  return focusedClause ? extractFocusedCommandArgv(focusedClause) : undefined
}

function extractProcessCwd(message: string): string | undefined {
  const match = message.match(/\bcwd\s*(?:=|:|\bis\b|는|은)?\s*(`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|[^\s,;]+)/iu)
  const raw = match?.[1]?.trim().replace(/^[`'"]|[`'"]$/gu, '')
  const cwd = raw?.replace(/[.;]$/u, '').replace(/(?:으?로)$/u, '')
  return cwd || undefined
}

function extractJsonObjectAfter(
  message: string,
  prefix: RegExp,
): Record<string, unknown> | undefined {
  const match = prefix.exec(message)
  if (!match || match.index === undefined) return undefined
  const start = match.index + match[0].length
  if (message[start] !== '{') return undefined

  let depth = 0
  let quote = false
  let escaped = false
  for (let index = start; index < message.length; index += 1) {
    const char = message[index]
    if (escaped) {
      escaped = false
      continue
    }
    if (char === '\\' && quote) {
      escaped = true
      continue
    }
    if (char === '"') {
      quote = !quote
      continue
    }
    if (quote) continue
    if (char === '{') depth += 1
    if (char !== '}') continue
    depth -= 1
    if (depth !== 0) continue

    try {
      const parsed: unknown = JSON.parse(message.slice(start, index + 1))
      return parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)
        ? parsed as Record<string, unknown>
        : undefined
    } catch {
      return undefined
    }
  }
  return undefined
}

function extractProcessNetwork(
  message: string,
): AgentRequestedProcessStart['network'] | undefined {
  const structured = extractJsonObjectAfter(message, /\bnetwork\s*(?:=|:)\s*/iu)
  if (structured) {
    const mode = structured.mode
    if (mode !== 'none' && mode !== 'loopback') return undefined
    if (mode === 'none') return 'none'
    const rawPorts = structured.ports
    if (rawPorts !== undefined && !Array.isArray(rawPorts)) return undefined
    const ports = Array.isArray(rawPorts)
      ? [...new Set(rawPorts.filter(
          (port): port is number => Number.isInteger(port) && Number(port) >= 1 && Number(port) <= 65_535,
        ))]
      : undefined
    if (Array.isArray(rawPorts) && ports?.length !== rawPorts.length) return undefined
    return { mode, ...(ports?.length ? { ports } : {}) }
  }

  const mode = message.match(/\bnetwork(?:\.mode)?\s*(?:=|:|\bis\b|는|은)?\s*(none|loopback)\b/iu)?.[1]?.toLowerCase()
  const portList = message.match(/\bports?\s*(?:=|:|\bis\b|는|은)?\s*\[([\d,\s]+)\]/iu)?.[1]
  const ports = portList
    ? [...new Set(portList.split(',')
        .map((entry) => Number.parseInt(entry.trim(), 10))
        .filter((port) => Number.isInteger(port) && port >= 1 && port <= 65_535))]
    : undefined
  return mode === 'none'
    ? 'none'
    : mode === 'loopback'
      ? { mode, ...(ports?.length ? { ports } : {}) }
      : undefined
}

function extractStructuredProcessStart(
  message: string,
): AgentRequestedProcessStart | undefined {
  const command = extractStructuredProcessCommand(message) ?? extractFocusedCommandArgv(message)
  if (!command) return undefined

  const cwd = extractProcessCwd(message)
  const ttlText = message.match(/\bttlMs\s*(?:=|:)\s*(\d+)\b/iu)?.[1]
  const ttlMs = ttlText === undefined ? undefined : Number(ttlText)
  const lifetime = message.match(/\blifetime\s*(?:=|:|\bis\b|는|은)?\s*(bounded|session)\b/iu)?.[1]?.toLowerCase()
  const network = extractProcessNetwork(message)

  return {
    ...command,
    ...(cwd ? { cwd } : {}),
    ...(Number.isSafeInteger(ttlMs) ? { ttlMs } : {}),
    ...(lifetime === 'bounded' || lifetime === 'session' ? { lifetime } : {}),
    ...(network ? { network } : {}),
  }
}

/** Extract a structurally bounded process.start request without model inference. */
export function extractFocusedProcessStart(
  message: string,
): AgentRequestedProcessStart | undefined {
  if (!isFocusedManagedProcessStart(message)) return undefined
  return extractStructuredProcessStart(message)
}

/**
 * Preserve exact structured launch options inside a broader process workflow.
 * The caller may still need process.read, browser validation, or another
 * explicitly requested step, so this is a constraint rather than permission
 * to collapse the whole turn to one deterministic process.start call.
 */
export function extractConstrainedProcessStart(
  message: string,
): AgentRequestedProcessStart | undefined {
  const normalized = message.trim()
  if (
    !normalized
    || normalized.length > 3_000
    || !EXPLICIT_MANAGED_PROCESS_SURFACE_RE.test(normalized)
    || !extractStructuredProcessCommand(normalized)
  ) {
    return undefined
  }
  return extractStructuredProcessStart(normalized)
}

/** Whether a focused command asks for the observed stdout/stderr without interpretation. */
export function requestsRawTerminalCommandOutput(message: string): boolean {
  return isFocusedTerminalCommandExecution(message) && RAW_COMMAND_OUTPUT_RE.test(message)
}

/**
 * A small code-and-test task with an explicit filesystem boundary. This is
 * intentionally narrower than generic mutation intent: broad repository work,
 * UI work, migrations, and refactors still need the multi-phase coder graph.
 * The direct loop is materially cheaper for a self-contained source/test pair
 * and still retains normal write approvals and terminal policy enforcement.
 */
export function isFocusedBoundedCodingTask(message: string): boolean {
  const normalized = message.trim()
  if (!normalized || normalized.length > 800) return false
  if (!EXPLICIT_CURRENT_DIRECTORY_BOUNDARY_RE.test(normalized)) return false
  if (!SMALL_SOURCE_AND_TEST_DELIVERABLE_RE.test(normalized)) return false
  if (BROAD_CODING_SCOPE_RE.test(normalized)) return false
  if (RENDERED_UI_CODING_SCOPE_RE.test(normalized)) return false
  return EXPLICIT_REPOSITORY_MUTATION_RE.test(normalized)
}

/**
 * A tightly scoped, single-file literal substitution. These edits fit the
 * ordinary tool loop: read once, show an approval diff, edit, read back, and
 * answer. Sending them through the multi-phase coder graph invents contracts,
 * compiler checks, and recovery loops that outweigh the requested change.
 */
export function isFocusedSingleFileReplacement(message: string): boolean {
  const normalized = message.trim()
  if (!normalized || normalized.length > 500) return false
  if (EXPLICIT_ALTERNATE_EXECUTION_SURFACE_RE.test(normalized)) return false
  if (BROAD_REPLACEMENT_SCOPE_RE.test(normalized)) return false
  if (ADDITIONAL_MUTATION_RE.test(normalized)) return false

  const paths = new Set(normalized.match(FILE_PATH_TOKEN_RE) ?? [])
  if (paths.size !== 1) return false

  return ENGLISH_LITERAL_REPLACEMENT_RE.test(normalized)
    || KOREAN_LITERAL_REPLACEMENT_RE.test(normalized)
}
