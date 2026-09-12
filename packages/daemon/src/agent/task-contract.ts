import {
  ThinkingLevel,
  type AgentArtifactSection,
  type AgentExecutionCapability,
  type AgentExecutionIntent,
  type AgentEvidenceRequirement,
  type AgentRequiredArtifact,
  type AgentRunContract,
  type ChatRequest,
} from '@sepilotd/core'
/*
 * Keep the contract planner on the same provider-neutral function-calling
 * protocol as the execution graph. Free-form JSON is a fragile transport for
 * reasoning models and used to silently collapse precise user requirements
 * into the generic fallback contract.
 */
export const DURABLE_RUN_CONTRACT_TOOL_NAME = 'durable_run_contract'

/*
 * The schema describes semantic completion data, including only user-imposed
 * tool-order and target-authority boundaries. It never asks the planner to
 * invent an implementation trajectory. Provider adapters translate the
 * required contract tool choice to their native dialect.
 */
function durableRunContractTool() {
  return {
    name: DURABLE_RUN_CONTRACT_TOOL_NAME,
    description: 'Return the durable semantic completion contract for this user request.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      required: ['summary', 'acceptanceCriteria', 'constraints', 'outOfScope', 'executionIntent'],
      properties: {
        summary: { type: 'string' },
        acceptanceCriteria: {
          type: 'array',
          minItems: 2,
          maxItems: 7,
          items: { type: 'string' },
        },
        constraints: {
          type: 'array',
          maxItems: 6,
          items: { type: 'string' },
        },
        outOfScope: {
          type: 'array',
          maxItems: 4,
          items: { type: 'string' },
        },
        executionIntent: {
          type: 'object',
          additionalProperties: false,
          required: ['kind', 'workspaceMutation', 'capabilities'],
          properties: {
            kind: {
              type: 'string',
              enum: ['operational-action', 'workspace-change', 'inspection', 'artifact-production', 'conversation'],
            },
            workspaceMutation: {
              type: 'string',
              enum: ['forbidden', 'allowed', 'required'],
            },
            capabilities: {
              type: 'array',
              items: {
                type: 'string',
                enum: ['process', 'service', 'terminal', 'browser', 'filesystem-read', 'filesystem-write', 'network', 'application-state'],
              },
            },
            allowedTools: {
              type: 'array',
              maxItems: 32,
              items: {
                type: 'string',
                pattern: '^[A-Za-z0-9_.:-]+$',
              },
            },
            toolSequence: {
              type: 'array',
              maxItems: 32,
              uniqueItems: true,
              items: {
                type: 'string',
                pattern: '^[A-Za-z0-9_.:-]+$',
              },
            },
            retryPolicy: {
              type: 'string',
              enum: ['forbidden'],
            },
            authorizedWriteTargets: {
              type: 'array',
              maxItems: 32,
              items: { type: 'string', maxLength: 160 },
            },
            protectedWriteTargets: {
              type: 'array',
              maxItems: 32,
              items: { type: 'string', maxLength: 160 },
            },
          },
        },
        requiredArtifacts: { type: 'array', items: { type: 'object' } },
        evidenceRequirements: { type: 'array', items: { type: 'object' } },
        artifactSections: { type: 'array', items: { type: 'object' } },
      },
    },
  }
}

import {
  extractConstrainedProcessStart,
  extractFocusedProcessStart,
  extractFocusedTerminalCommand,
  isFocusedTerminalCommandExecution,
  isFocusedRepositoryChangeLookup,
  isSubstantiveRepositoryChangeReview,
  inputRequestsTerminalCommandExecution,
} from './request-shape.js'

const MAX_SUMMARY_CHARS = 280
const MAX_CRITERION_CHARS = 360
const MAX_FALLBACK_REQUEST_CHARS = 1_024
const MAX_REQUEST_GIST_CHARS = 600
const MAX_LIST_ITEM_CHARS = 260
const MAX_ARTIFACT_PATH_CHARS = 260
const MAX_ARTIFACTS = 8
const MAX_EVIDENCE_REQUIREMENTS = 6
const MAX_ARTIFACT_SECTIONS = 12
const DEFAULT_RUN_CONTINUATION_POLICY =
  'If context, time, tool, or iteration budget is insufficient, preserve resumable state and report the next concrete step.'
const DEFAULT_BROAD_COVERAGE_POLICY =
  'For broad or exhaustive analysis/report work, keep an explicit evidence or coverage map of inspected areas, or clearly mark remaining scope as incomplete.'

function isDefaultContingentRunPolicy(value: string): boolean {
  const normalized = normalizeText(value).toLowerCase()
  return normalized === DEFAULT_RUN_CONTINUATION_POLICY.toLowerCase()
    || normalized === DEFAULT_BROAD_COVERAGE_POLICY.toLowerCase()
}
const MAX_EXPLICIT_NO_TOUCH_TARGETS = 4

/**
 * A name-only sequence can enforce transitions between distinct tools, but it
 * cannot identify two semantic actions performed through the same tool. Treat
 * any duplicate name as an unrepresentable sequence and fail open to ordinary
 * tool policy instead of turning unrelated preparatory calls into completed
 * workflow steps. Exact argv/process contracts remain separate authorities.
 */
export function normalizeRepresentableToolSequence(
  sequence: readonly string[],
): string[] {
  const seen = new Set<string>()
  for (const toolName of sequence) {
    if (seen.has(toolName)) return []
    seen.add(toolName)
  }
  return [...sequence]
}

const EXPLICIT_NO_RETRY_PATTERN =
  /\b(?:(?:do\s+not|don['’]t|dont|never)\s+(?:retry|repeat|rerun|re-run)|without\s+(?:a\s+)?retr(?:y|ies)|no\s+retr(?:y|ies)|(?:each|every)\b[^.!?;\n]{0,100}\b(?:at\s+most|only|exactly)\s+(?:once|one\s+time))\b|(?:재시도|재호출|재실행)(?:는|은|을|를|도)?\s*(?:하지\s*(?:마|말)|금지|없이)|(?:다시|반복해서?)\s*(?:실행|호출|시도)하지\s*(?:마|말)|(?:각각|각\s*(?:작업|동작|검증|단계))[^.!?。！？;；\n]{0,100}(?:정확히\s*)?(?:한\s*번|한번|1\s*회)만?/iu

/** Explicit user authority for a run-level no-retry failure boundary. */
export function inputExplicitlyForbidsRetries(input: string): boolean {
  return EXPLICIT_NO_RETRY_PATTERN.test(input)
}
export const RUN_CONTRACT_PLANNER_MAX_TOKENS = 4096

const FALLBACK_DURABLE_ARTIFACT_EXTENSIONS = new Set([
  'csv',
  'doc',
  'docx',
  'htm',
  'html',
  'json',
  'md',
  'mdx',
  'pdf',
  'ppt',
  'pptx',
  'svg',
  'tsv',
  'txt',
  'xml',
  'yaml',
  'yml',
])
const DOCUMENT_ARTIFACT_EXTENSIONS = new Set([
  'adoc',
  'doc',
  'docx',
  'md',
  'mdx',
  'pdf',
  'rst',
  'txt',
])
const FALLBACK_ARTIFACT_PATH_PATTERN =
  /(?:^|[\s`"'([])((?:\.{1,2}\/|\/|~\/)?[A-Za-z0-9_.@-]+(?:\/[A-Za-z0-9_.@-]+)*\.[A-Za-z0-9][A-Za-z0-9_-]{0,15})(?=$|[\s`"',.)\]\uAC00-\uD7AF])/g
const FALLBACK_ARTIFACT_INTENT_PATTERN =
  /\b(?:author|build|create|develop|draft|edit|emit|export|fix|generate|implement|make|migrate|modify|output|patch|produce|refactor|save|scaffold|ship|update|write)\b|(?:작성|생성|저장|만들|기록|출력|수정|업데이트|구현|개발|코딩|제작|고치|마이그레이션|패치|리팩터|스캐폴드|써\s*줘|써줘)/giu
// `output` is both an authoring verb ("output results.json") and the noun for
// observed command/tool results ("report the command output").  The broad
// artifact lexicon intentionally includes the verb, so mask only structurally
// observational noun phrases before applying it.  A named-path/output action
// remains visible through the surrounding authoring verb or artifact path.
const OBSERVED_OUTPUT_NOUN_PATTERN =
  /\b(?:report|show|display|return|summari[sz]e|inspect|verify|check|observe|include)\b[^.!?\n]{0,80}\b(?:(?:command|tool|runtime|test|build|terminal)\s+)?outputs?\b|\b(?:command|tool|runtime|test|build|terminal)\s+outputs?\b[^.!?\n]{0,48}\b(?:is|are|was|were|as|for|from|shows?|contains?|evidence)\b/giu
// Authoring vocabulary can name the subject of a read-only investigation
// rather than an action: “policy authoring principles”, “how to write a
// plugin”, and “정책 작성 방법을 조사해” do not authorize workspace edits.
// Strip only a mutation term structurally attached to a concept noun or
// how-to phrase; a later independent “save/write/구현해” action stays visible.
const REFERENCED_AUTHORING_CONCEPT_PATTERN =
  /\b(?:how|ways?|methods?)\s+to\s+(?:author|build|create|develop|draft|implement|make|modify|write)\b|\b(?:authoring|building|creation|development|drafting|implementation|recording|writing)\s+(?:best\s+practices|candidates?|destinations?|guidelines|locations?|methods?|pages?|patterns?|principles|process|rules?|targets?|workspaces?)\b|(?:작성|생성|구현|개발|제작|수정|기록)\s*(?:(?:대상|후보)(?:\s*후보)?|모범\s*사례|가이드(?:라인)?|규칙|방법|방식|원칙|절차|지침)(?:을|를|은|는)?|(?:작성|생성|구현|개발|제작|수정|기록|저장)할\s*수\s*있는(?=[^.!?。！？\n]{0,80}(?:대상|목적지|위치|후보|페이지|워크스페이스|destination|location|page|target|workspace))/giu
// A completed authoring verb can describe an object that already exists
// rather than grant authority to author it now: "saved schedules", "generated
// jobs", "저장된 스케줄", and "생성된 작업" are observation targets. Mask
// only the structurally adjectival/passive participle. A real predicate in the
// same clause remains visible ("update the saved report", "저장된 스케줄을
// 수정해줘"), so existing-state references cannot hide a requested mutation.
const EXISTING_STATE_AUTHORING_PARTICIPLE_PATTERN =
  /\b(?:authored|built|created|developed|drafted|edited|exported|generated|implemented|modified|patched|produced|recorded|saved|scaffolded|updated|written)(?=\s+(?:(?:already|currently|previously)\s+)?[\p{L}\p{N}_-])|(?:작성|생성|저장|기록|출력|수정|업데이트|구현|개발|제작|패치|리팩터)(?:해\s*둔|해둔|되어\s*있는|되어있는|된)(?=\s*[\p{Script=Hangul}\p{L}\p{N}_-])/giu
// "Development" can identify an environment being inspected rather than an
// action to create software. Strip only the bounded environment noun phrase;
// an adjacent positive verb such as implement/fix/구현/수정 remains visible.
const REFERENCED_DEVELOPMENT_ENVIRONMENT_PATTERN =
  /\bdevelopment\s+(?:cluster|environment|infrastructure|namespace|server)\b|개발\s*(?:클러스터|환경|인프라|네임스페이스|서버)/giu
// "Development" can also qualify the environment of an arbitrary operational
// surface (for example, "development CI status" or "개발 Foo 배포 현황").
// Mask only the noun modifier when the same bounded clause contains both a
// state noun and an observation action. Conjugated Korean authoring predicates
// remain visible, so "앱을 개발하고 상태를 확인해" still carries mutation
// intent.
const REFERENCED_DEVELOPMENT_OPERATIONAL_STATUS_PATTERN =
  /\bdevelopment\b(?=[^.!?\n]{0,96}\b(?:status|state|health)\b[^.!?\n]{0,64}\b(?:check|inspect|observe|report|review|show|summari[sz]e|validate|verify)\b)|개발(?!하|해|했|될|된|할|을|를|자)(?=[^.!?。！？\n]{0,96}(?:상태|현황|건강도)(?:은|는|을|를)?[^.!?。！？\n]{0,64}(?:점검|조회|확인|관찰|보고|검토|검증|요약))/giu
// A CLI output-format option describes serialization, not authority to create
// a workspace artifact. Keep path-like values visible: `--output report.json`
// may name a durable write target, while `--output json` is only a format.
const CLI_OUTPUT_FORMAT_OPTION_PATTERN =
  /--output(?:=|\s+)(?:csv|json|jsonl|ndjson|table|text|tsv|yaml|yml)\b/giu
// "Build" is both an authoring verb and the name of a validation command.
// A clause that explicitly runs/re-runs/checks a build is operational
// evidence, not permission to edit source.  Remove only that bounded command
// phrase before applying the broader authoring lexicon; positive mutation in
// another clause remains visible.
const BUILD_VALIDATION_ACTION_PATTERN =
  /\b(?:run|execute|re[-\s]?run|check|verify)\s+(?:the\s+)?(?:(?:existing|affected|unit|integration|production)\s+)*(?:(?:tests?|test\s+suite)\s+(?:and|then)\s+)?(?:production\s+)?build\b|(?:(?:프로덕션|운영)\s*)?빌드(?:를|을)?\s*(?:실행|재실행|돌려|확인|검증)/giu
const TEST_VALIDATION_ACTION_PATTERN =
  /\b(?:run|execute|re[-\s]?run|check|verify)\s+(?:the\s+)?(?:(?:existing|affected|unit|integration|end[- ]to[- ]end|e2e)\s+)*(?:tests?|test\s+suite)\b|\b(?:tests?|test\s+suite)\b[^.!?\n]{0,36}\b(?:run|executed?|re[-\s]?run|checked?|verified?)\b|(?:(?:기존|영향받는|단위|통합|e2e|엔드투엔드)\s*)?(?:테스트|테스트\s*스위트)(?:를|을)?[^.!?。！？\n]{0,24}(?:실행|재실행|돌려|확인|검증)/giu
const RENDERED_UI_PREIMPLEMENTATION_AUTHORING_PATTERN =
  /\b(?:author|build|create|develop|design|implement|make)\b|(?:작성|생성|구현|개발|코딩|제작|디자인|만들)(?:하|해|어|기|자|라|어라|어줘|해주세요)?/iu
// Browser QA commonly persists screenshots as evidence.  That is an output
// of the validation tool, not authority to modify the product under test.
// Strip only screenshot/capture persistence phrases when the same request has
// the structural browser-validation signals below; an independent "save the
// image" request remains ordinary authoring/mutation intent.
const RENDERED_UI_EVIDENCE_ARTIFACT_ACTION_PATTERN =
  /\b(?:capture|take|save|export|record|write)\s+(?:(?:only|the|explicitly|requested|attached|browser|desktop|mobile|responsive|page|visual|and|&|\+|\/)\s+)*(?:screenshots?|screen\s*shots?|screen\s+captures?|shots?|visual\s+evidence|screenshot\s+artifacts?)\b(?:\s+(?:to|under|in|into|at)\s+[^\s,;.\n]+)?|\b(?:(?:browser\.)?screenshots?|screen\s*shots?|screen\s+captures?|visual\s+evidence|screenshot\s+artifacts?)\b[^.!?。！？\n]{0,120}\b(?:captured|taken|saved?|exported?|recorded?|written?)\b(?:\s+(?:to|under|in|into|at)\s+[^\s,;.\n]+)?|(?:스크린샷|화면\s*캡처|시각\s*증거)(?:을|를)?(?:\s+[^\s,;.。！？\n]+\s*(?:아래|경로에|에|로))?\s*(?:캡처|촬영|저장|기록|남겨)(?:하|해|하고|해서|하여|해줘|해주세요|기)/giu
// Negative scope clauses are capability boundaries, not action requests.
// Strip only the negated mutation verb phrase (not the surrounding request)
// before applying broad mutation/follow-up lexicons. This keeps commands such
// as “do not edit files; run the tests” from inheriting an earlier edit loop,
// while a later positive clause such as “then fix failures” remains visible.
const EXPLICITLY_NEGATED_MUTATION_ACTION_PATTERN =
  /\b(?:do\s+not|don't|dont|never)\s+(?:directly\s+)?(?:modify|edit|touch|change|write\s+to|delete|remove|mutate)\b|\bwithout\s+(?:modifying|editing|touching|changing|writing\s+to|deleting|removing|mutating)\b|\bno\s+(?:direct\s+)?(?:file(?:system)?\s+)?(?:write|writes|writing|edit|edits|editing|modification|modifications|mutation|mutations)\b|\b(?:file(?:system)?\s+)?(?:writes?|writing|edits?|editing|modifications?|mutations?|changes?)\s+(?:is|are)\s+(?:forbidden|prohibited|disallowed)\b|(?:수정|편집|변경|건드리|고치|삭제|지우)(?:은|는|을|를)?\s*(?:하지\s*(?:마|말|않)|말고|없이|않(?:고|도록|는다|습니다)|금지(?:한다|합니다|됨|입니다)?)/giu
const FOCUSED_SINGLE_BROWSER_OBSERVATION_PATTERN =
  /\bbrowser\.(?:navigate|extract)\b/giu
// A focused browser observation may be completed directly from one tool
// result only when the affirmative request names no second tool surface. Keep
// this boundary structural: it recognizes the runtime's canonical built-in
// namespaces after negative capability clauses have been removed, rather
// than trying to infer workflow meaning from task-specific prose.
const STRUCTURED_BUILTIN_TOOL_REFERENCE_PATTERN =
  /\b(?:browser|computer|doc|fs|git|memory|notebook|office|process|service|system|terminal|web)\.(?:\*|[a-z][\w.-]*)(?=$|[^\w.-])|\bapply_patch\b/giu
const FOCUSED_SINGLE_OPERATION_BOUNDARY_PATTERN =
  /\b(?:exactly\s+once|once\s+only|one\s+time|single\s+(?:call|operation|navigation|extraction))\b|(?:정확히\s*)?(?:한\s*번|한번|1\s*회)(?:만)?/iu
const FOCUSED_BROWSER_OBSERVATION_TARGET_PATTERN =
  /https?:\/\/[^\s<>()]+|\b(?:current|currently\s+open|running|active)\s+(?:page|screen|site|app|url)\b|(?:현재|지금)\s*(?:실행\s*중인|열린|표시된|활성)?\s*(?:페이지|화면|사이트|웹\s*앱|URL)/iu
const DURABLE_DOCUMENT_AUTHORING_PATTERN =
  /\b(?:author|create|draft|generate|produce|save|write)\b|(?:작성|생성|저장|만들|기록|써\s*줘|써줘)/iu
const DURABLE_DOCUMENT_DELIVERABLE_PATTERN =
  /\b(?:architecture\s+(?:documentation|plan)|design\s+(?:doc(?:ument)?|plan)|(?:technical|implementation|written)\s+plan|documents?|documentation|wiki|runbook|whitepaper|specification|reports?|written\s+(?:analysis|audit|report)|(?:analysis|audit|research|status)\s+report)\b|(?:아키텍처\s*문서|설계서|설계\s*문서|문서|문서화|위키|런북|백서|명세서|분석서|감사\s*보고서|조사\s*보고서|결과\s*보고서|보고서)/iu
const NEGATED_DURABLE_DOCUMENT_AUTHORING_PATTERN =
  /\b(?:do\s+not|don't|dont|never|without)\b[^.!?;\n]{0,100}\b(?:author|create|draft|extend|expand|generate|produce|save|update|write)\b|(?:작성|생성|저장|만들|기록|확장|업데이트|갱신)[^.!?。！？;；\n]{0,60}?(?:하지\s*(?:마|말|않)|말고|없이|않(?:고|도록|는다|습니다))/iu
const EXPLICIT_DOCUMENTATION_ACTION_PATTERN =
  /\bdocument(?:ing)?\s+(?:the\s+|this\s+|our\s+|current\s+)?(?:api|architecture|behavior|codebase|design|feature|repository|system|workflow)\b|(?:API|아키텍처|동작|코드베이스|설계|기능|저장소|시스템|워크플로)(?:를|을)?\s*문서화/iu
const EXPLICIT_NO_TOUCH_ENGLISH_PATTERN =
  /(?:do\s+not|don't|never)\s+(?:directly\s+)?(?:modify|edit|touch|change|update|overwrite|write\s+to|delete|remove|mutate)\s+(?:the\s+)?(?<target>`[^`]+`|"[^"]+"|'[^']+'|[./~A-Za-z0-9_@-][^\s,;:)\]\n]*)/giu
const EXPLICIT_NO_TOUCH_KOREAN_PATTERN =
  /(?:절대로|절대)?\s*(?:직접\s*)?(?<target>`[^`]+`|"[^"]+"|'[^']+'|[./~A-Za-z0-9_@-][^\s,;:)\]\n]*)\s*(?:은|는|을|를)?\s*(?:직접\s*)?(?:수정|편집|변경|업데이트|갱신|덮어쓰|건드리|고치|삭제|지우)[^\n.?!。！？]{0,24}?(?:하지\s*마|하지\s*말|말고|마라|않)/giu
const AMBIGUOUS_NO_TOUCH_TARGETS = new Set([
  'it',
  'this',
  'that',
  'result',
  'artifact',
  'output',
  'project',
  'repo',
  'repository',
  '이거',
  '그거',
  '저거',
  '결과',
  '산출물',
  '프로젝트',
])
const NO_TOUCH_CONSTRAINT_PREFIX = 'Do not directly modify target '
const NO_TOUCH_OUT_OF_SCOPE_PREFIX = 'Direct file edits, deletion, or mutation of '
const CURRENT_DOCUMENT_PHASE_CONSTRAINT_PREFIX = 'Limit this run to the explicitly requested current document/planning phase'
const DEFERRED_IMPLEMENTATION_OUT_OF_SCOPE_PREFIX = 'Downstream implementation after the explicitly requested document/planning phase'
const RENDERED_UI_REQUEST_PATTERN =
  /\b(?:front[- ]?end|web\s*(?:app|site|page|ui|games?)?|website|webpage|static\s+site|browser\s+(?:app|games?)|spa|single[- ]page\s+app|dashboard|landing\s+(?:page|screen)|component\s+library|html|css|ui|ux|layout|responsive|games?|mini[- ]?games?|canvas)\b|\b(?:react|vue|svelte(?:kit)?|next(?:\.js)?|nuxt|astro|solid(?:js)?|angular|tailwind)\b.{0,48}\b(?:app|page|route|screen|component|view|dashboard|site|ui|form|button|landing)\b|\b(?:app|page|route|screen|component|view|dashboard|site|ui|form|button|landing)\b.{0,48}\b(?:react|vue|svelte(?:kit)?|next(?:\.js)?|nuxt|astro|solid(?:js)?|angular|tailwind)\b|(?:웹|프론트|사이트|페이지|화면|디자인|게임|대시보드|레이아웃|반응형)/iu
const DIRECT_RENDERED_UI_AUTHORING_PATTERN =
  /\b(?:author|build|clone|code|create|design|develop|implement|make|polish|redesign|refine|restyle|ship|update)\b(?:(?!\b(?:analysis|audit|document|report|summary)\b)[^.!?;\n]){0,96}\b(?:front[- ]?end|web\s*(?:app|site|page|ui|games?)?|website|webpage|browser\s+(?:app|games?)|spa|single[- ]page\s+app|dashboard|landing\s+(?:page|screen)|component\s+library|html|css|ui|ux|layout|responsive|games?|canvas)\b|(?:웹\s*(?:앱|사이트|페이지|UI)|프론트(?:엔드)?|사이트|페이지|화면|UI|UX|게임|대시보드|레이아웃|반응형|컴포넌트)(?:을|를|은|는|이|가)?\s*(?:직접\s*)?(?:작성|생성|구현|개발|코딩|제작|디자인|수정|편집|변경|개선|리디자인|만들|고치|다듬)/iu
const INTERACTIVE_RENDERED_UI_REQUEST_PATTERN =
  /\b(?:games?|mini[- ]?games?|web\s*app|browser\s*(?:app|games?)|spa|single[- ]page\s+app|dashboard|tool|calculator|editor|simulator|player|quiz|tab|menu|button|form|drag|drop|interactive|interaction|click|canvas)\b|\b(?:react|vue|svelte(?:kit)?|next(?:\.js)?|nuxt|astro|solid(?:js)?|angular)\b.{0,48}\b(?:app|route|dashboard|form|button|editor|tool|calculator|quiz|menu|tab)\b|\b(?:app|route|dashboard|form|button|editor|tool|calculator|quiz|menu|tab)\b.{0,48}\b(?:react|vue|svelte(?:kit)?|next(?:\.js)?|nuxt|astro|solid(?:js)?|angular)\b|(?:게임|웹\s*앱|앱|대시보드|도구|계산기|에디터|시뮬레이터|퀴즈|탭|메뉴|버튼|폼|드래그|상호작용|인터랙션|클릭|캔버스)/iu
const COMMON_INTERACTIVE_UI_TOOL_PATTERN =
  /\b(?:(?:to[-\s]?do|todo|task)\s+(?:app|tool|board|list)|kanban\s+board|pomodoro\s+timer|stopwatch|(?:timer|countdown)\s+(?:app|tool|ui)|calculator(?:\s+(?:app|tool|ui))?|weather\s+app|notes?\s+app|chat\s+app|drawing\s+(?:app|tool)|paint\s+(?:app|tool)|markdown\s+editor|text\s+editor|calendar\s+app|habit\s+tracker|expense\s+tracker|budget\s+tracker)\b|(?:할\s*일|투두|태스크)\s*(?:앱|도구|보드|목록)|칸반\s*보드|포모도로\s*타이머|스톱워치|계산기(?:\s*(?:앱|도구|ui))?|날씨\s*앱|메모\s*앱|채팅\s*앱|그림\s*(?:앱|도구)|마크다운\s*에디터|텍스트\s*에디터|캘린더\s*앱|습관\s*추적(?:기|앱)?/iu
const COMMON_INTERACTIVE_UI_TOOL_EXPLICIT_SURFACE_PATTERN =
  /\b(?:app|web|browser|front[- ]?end|ui|screen|page|canvas|editor|board|interactive|playable)\b|(?:앱|웹|브라우저|프론트|화면|페이지|캔버스|에디터|보드|상호작용|인터랙션)/iu
const COMMON_INTERACTIVE_UI_TOOL_COMMAND_LINE_PATTERN =
  /\b(?:cli|tui|command[-\s]?line|terminal|shell)\b|(?:터미널|셸|쉘|명령줄|커맨드라인)/iu
const POSITIVE_COMMAND_LINE_INTERFACE_PATTERN =
  /\b(?:build|create|develop|design|implement|make|run|use)\b[^.!?\n]{0,64}\b(?:cli|tui|command[-\s]?line|terminal|shell)\b|\b(?:cli|tui|command[-\s]?line|terminal|shell)\b[^.!?\n]{0,48}\b(?:app|game|interface|tool|version|only|based|implementation)\b|(?:cli|tui|터미널|셸|쉘|명령줄|커맨드라인)(?:로|에서|용|기반(?:으로)?|만)[^.!?。！？\n]{0,48}(?:구현|개발|디자인|실행|동작|사용|만들)|(?:cli|tui|터미널|명령줄)[^.!?。！？\n]{0,32}(?:앱|게임|도구|인터페이스|버전)/iu
const EXPLICIT_BROWSER_RENDERED_SURFACE_PATTERN =
  /\b(?:browser|web(?:site|page)?|front[- ]?end|html|css|canvas)\b|(?:브라우저|웹|프론트|사이트|웹\s*페이지|캔버스)/iu
// Validation-only follow-ups are a first-class browser capability request even
// when the UI was authored in an earlier turn. Keep the signal structural:
// there must be an operational validation verb plus concrete rendered-page
// evidence (a browser tool, screenshot/viewport evidence, or a browser and a
// rendered target). Merely discussing browser APIs or screenshot docs does not
// satisfy both sides of that boundary.
const EXPLICIT_RENDERED_UI_VALIDATION_ACTION_PATTERN =
  /\b(?:open|navigate|visit|inspect|validate|verify|test|audit|check|capture|take|record|extract|exercise|interact|click|evaluate)\b|(?:열어|접속|탐색|검사|검수|확인|테스트|검증|감사|캡처|촬영|추출|상호작용|클릭|평가)/iu
const EXPLICIT_RENDERED_UI_VALIDATION_EVIDENCE_PATTERN =
  /\bbrowser\.(?:navigate|screenshot|click|evaluate|extract)\b|\b(?:screenshots?|screen\s*shots?|viewport|visual\s+qa|layout\s+audit|rendered\s+(?:ui|page)|(?:real|headless)\s+browser|desktop\s*(?:and|&|\/)\s*mobile|mobile\s*(?:and|&|\/)\s*desktop)\b|(?:스크린샷|화면\s*캡처|뷰포트|시각\s*검수|레이아웃\s*검수|렌더링된\s*(?:ui|화면|페이지)|실제\s*브라우저|헤드리스\s*브라우저|데스크톱\s*(?:과|와|및|\/)\s*모바일|모바일\s*(?:과|와|및|\/)\s*데스크톱)/iu
const EXPLICIT_RENDERED_UI_SCREENSHOT_REQUEST_PATTERN =
  /\bbrowser\.screenshot\b|\b(?:screenshots?|screen\s*shots?|viewport|desktop\s*(?:and|&|\/)\s*mobile|mobile\s*(?:and|&|\/)\s*desktop)\b|(?:스크린샷|화면\s*캡처|뷰포트|데스크톱\s*(?:과|와|및|\/)\s*모바일|모바일\s*(?:과|와|및|\/)\s*데스크톱)/iu
const EXPLICIT_RENDERED_UI_VISUAL_QA_REQUEST_PATTERN =
  /\b(?:visual\s+(?:qa|review|inspection)|layout\s+audit|spacing|overflow|contrast|overlap|collision|touch\s+targets?|asset\s+completeness)\b|(?:시각\s*(?:검수|리뷰)|레이아웃\s*(?:검수|감사)|간격|오버플로|넘침|대비|겹침|충돌|터치\s*타깃|에셋\s*누락)/iu
const EXPLICIT_RENDERED_UI_SMOKE_REQUEST_PATTERN =
  /\bbrowser\.(?:click|evaluate)\b|\b(?:console|page\s+errors?|browser\s+smoke|interaction|interact|click|dynamic\s+state)\b|(?:콘솔|페이지\s*오류|브라우저\s*스모크|상호작용|인터랙션|클릭|동적\s*상태)/iu
const EXPLICIT_RENDERED_UI_INTERACTION_REQUEST_PATTERN =
  /\bbrowser\.(?:click|evaluate)\b|\b(?:interaction|interact|click|dynamic\s+state)\b|(?:상호작용|인터랙션|클릭|동적\s*상태)/iu
const EXPLICIT_RENDERED_UI_VALIDATION_TARGET_PATTERN =
  /\b(?:page|site|website|webpage|web\s+app|ui|screen|view|dashboard|game|canvas|localhost|local\s+(?:server|service)|running\s+(?:server|service|app|page))\b|(?:페이지|사이트|웹\s*앱|ui|화면|뷰|대시보드|게임|캔버스|로컬호스트|로컬\s*(?:서버|서비스)|실행\s*중인\s*(?:서버|서비스|앱|페이지))/iu
const NON_RENDERED_SURFACE_INVENTORY_PATTERN =
  /(?:\b(?:app|pages?|site|workspace)\b|(?:앱|페이지|사이트|워크스페이스))(?:(?:\s*(?:and|&|\/|,|및|와|과)\s*)(?:\b(?:app|pages?|site|workspace)\b|(?:앱|페이지|사이트|워크스페이스)))*(?:의|에\s*있는)?\s*(?:\b(?:configuration|inventory|list|metadata|records?)\b|(?:구성|목록|메타데이터|레코드))|(?:\b(?:configuration|inventory|list|metadata|records?)\b|(?:구성|목록|메타데이터|레코드))\s+(?:of|for)\s+(?:the\s+)?(?:\b(?:app|pages?|site|workspace)\b|(?:앱|페이지|사이트|워크스페이스))(?:(?:\s*(?:and|&|\/|,|및|와|과)\s*)(?:\b(?:app|pages?|site|workspace)\b|(?:앱|페이지|사이트|워크스페이스)))*/iu
const EXPLICIT_RENDERED_UI_MUTATION_ACTION_PATTERN =
  /\b(?:author|build|create|develop|design|edit|fix|implement|make|modify|polish|refine|restyle|update)\b|(?:(?:작성|생성|구현|개발|코딩|제작|디자인|수정|편집|변경|개선|리디자인)\s*(?:하|해|해야|해줘|해주세요|하고|해서|하여|할|하자|해라|해봐|해주|되|시켜)|(?:만들|고치|다듬)(?:어|어줘|어주세요|고|어서|기|자|라|어라|어봐|어주))/iu
const STAGED_DOCUMENT_FIRST_PATTERN =
  /\b(?:first(?:ly)?|for\s+now|for\s+(?:this|the)\s+(?:phase|step|turn)|this\s+turn|start\s+by|begin\s+by|initially)\b(?:[^.!?\n]|\.(?=[\p{L}\p{N}_/-])){0,140}\b(?:design\s+(?:doc(?:ument)?|brief|plan)|architecture\s+(?:doc(?:ument)?|plan)|spec(?:ification)?|planning\s+document|written\s+plan)\b|(?:먼저|우선|일단|이번\s*(?:에는|턴에는)?)(?:[^.!?。！？\n]|\.(?=[\p{L}\p{N}_/-])){0,120}(?:설계\s*문서|설계서|기획\s*문서|계획서|명세서|문서)/iu
const CURRENT_PHASE_DOCUMENT_ONLY_PATTERN =
  /\b(?:for\s+(?:this|the)\s+(?:phase|step|turn)|this\s+(?:phase|step|turn)|for\s+now|at\s+this\s+stage)\b(?:[^.!?\n]|\.(?=[\p{L}\p{N}_/-])){0,160}\b(?:design|planning|specification|documentation)\b(?:[^.!?\n]|\.(?=[\p{L}\p{N}_/-])){0,80}\bonly\b|(?:지금\s*)?(?:(?:이번|이|현재)\s*(?:턴|단계|페이즈))(?:[^.!?。！？\n]|\.(?=[\p{L}\p{N}_/-])){0,120}(?:설계|기획|계획|명세|문서)(?:[^.!?。！？\n]|\.(?=[\p{L}\p{N}_/-])){0,40}(?:만\s*(?:수행|진행|작성|만들)|작성\s*만)/iu
const SAME_TURN_IMPLEMENTATION_AFTER_DOCUMENT_PATTERN =
  /\b(?:then|and\s+then|after(?:wards)?|next)\b[^.!?\n]{0,100}\b(?:build|code|develop|implement|ship)\b|(?:그리고|그\s*다음|이후|뒤이어|작성\s*후|작성하고|만들고|한\s*뒤)[^.!?。！？\n]{0,100}(?:구현|개발|코딩|제작)/iu
const NEGATED_DOWNSTREAM_IMPLEMENTATION_PATTERN =
  /\b(?:do\s+not|don't|dont|never)\s+(?:create|write|implement|build|develop|code)\b[^.!?;\n]{0,100}|(?:구현|개발|코딩|제작|소스\s*코드|소스|구현\s*파일|코드\s*파일)[^.!?。！？;；\n]{0,100}?(?:하지\s*(?:마|말(?:아|라)?|마라|않)|만들지\s*(?:마|말(?:아|라)?|마라)|작성하지\s*(?:마|말(?:아|라)?|마라)|없이|제외)/giu
const CURRENT_IMPLEMENTATION_DOCUMENT_PATTERN =
  /\b(?:analy[sz]e|audit|inspect|review|reconstruct|reverse[-\s]?engineer|extract)\b[^.!?\n]{0,120}\b(?:repository|repo|codebase|source|implementation|architecture|system)\b|\b(?:analysis|assessment|audit|documentation|inventory|report|review)\b(?:[^.!?\n]|\.(?=[\p{L}\p{N}_/-])){0,80}\b(?:for|of|on|about)\s+(?:(?:the|this|current|existing)\s+)?(?:repository|repo|codebase|source|implementation|architecture|system)\b|\b(?:current|existing|as[-\s]?is|implemented|observed)\s+(?:repository|repo|codebase|source|implementation|architecture|system)\b|(?:현재|기존|구현된|관찰된|실제)(?:의|\s)*(?:저장소|리포지토리|코드베이스|소스|구현|아키텍처|시스템)|(?:저장소|리포지토리|코드베이스|소스|구현|아키텍처|시스템)[^.!?。！？\n]{0,100}(?:분석|평가|감사|검토|리뷰|현황|인벤토리|보고서|문서화|역설계|추출)/iu
const COMMON_INTERACTIVE_UI_TOOL_NON_UI_PATTERN =
  /\b(?:api|client|server|backend|service|endpoint|function|method|class|library|package|sdk|module|utility|util|cli|command[-\s]?line|terminal|shell|parser|algorithm|solver|model|dataset|test|unit\s+test|headless|no\s+ui|without\s+(?:a\s+)?ui)\b|(?:백엔드|서버|엔드포인트|함수|메서드|클래스|라이브러리|패키지|모듈|유틸|터미널|셸|파서|알고리즘|솔버|모델|데이터셋|테스트|ui\s*없이|화면\s*없이)/iu
const COMMON_INTERACTIVE_UI_CONTROL_PATTERN =
  /\b(?:accordion|carousel|slider|range\s+slider|dropdown|drop[-\s]?down|combo\s*box|combobox|select\s+menu|menu|tabs?|toggle|switch|stepper|filter(?:able)?|sort(?:able)?|draggable|drag[-\s]?and[-\s]?drop|file\s+upload|upload\s+(?:widget|control|component)|date\s*picker|color\s*picker)\b|(?:아코디언|캐러셀|카루셀|슬라이더|드롭다운|콤보박스|선택\s*메뉴|탭|메뉴|토글|스위치|스테퍼|필터|정렬|드래그\s*(?:앤드|&)?\s*드롭|파일\s*업로드|날짜\s*선택|색상\s*선택)/iu
const COMMON_INTERACTIVE_UI_CONTROL_SURFACE_PATTERN =
  /\b(?:app|web|browser|front[- ]?end|ui|ux|interface|screen|page|component|widget|control|form|panel|modal|dashboard|react|vue|svelte(?:kit)?|next(?:\.js)?|nuxt|astro|solid(?:js)?|angular|tailwind)\b|(?:앱|웹|브라우저|프론트|ui|ux|인터페이스|화면|페이지|컴포넌트|위젯|컨트롤|폼|양식|패널|모달|대시보드)/iu
const COMMON_RENDERED_UI_SURFACE_PATTERN =
  /\b(?:(?:login|log[-\s]?in|sign[-\s]?in|signup|sign[-\s]?up|register|registration|auth|checkout|payment|cart|pricing|portfolio|profile|settings|preferences|onboarding|contact|feedback|search|booking|reservation|product|account|admin)\s+(?:page|screen|view|form|flow|wizard|modal|panel|card|table)|(?:page|screen|view|form|flow|wizard|modal|panel|card|table)\s+(?:for|to)\s+(?:login|log[-\s]?in|sign[-\s]?in|signup|sign[-\s]?up|register|registration|auth|checkout|payment|cart|pricing|portfolio|profile|settings|preferences|onboarding|contact|feedback|search|booking|reservation|product|account|admin)|admin\s+panel|control\s+panel|hero\s+section|navigation\s+bar|navbar|pricing\s+table|product\s+card|profile\s+card|contact\s+form|login\s+form|signup\s+form)\b|(?:로그인|회원\s*가입|가입|인증|체크아웃|결제|장바구니|가격|요금|포트폴리오|프로필|설정|온보딩|문의|연락처|피드백|검색|예약|상품|제품|계정|관리자|어드민)\s*(?:페이지|화면|뷰|폼|양식|플로우|모달|패널|카드|표)|(?:관리자|어드민)\s*(?:패널|화면|페이지)/iu
const COMMON_RENDERED_UI_SURFACE_INTERACTIVE_PATTERN =
  /\b(?:login|log[-\s]?in|sign[-\s]?in|signup|sign[-\s]?up|register|registration|auth|checkout|payment|cart|settings|preferences|onboarding|contact|feedback|search|booking|reservation|admin|form|flow|wizard|modal|panel|navbar|navigation\s+bar)\b|(?:로그인|회원\s*가입|가입|인증|체크아웃|결제|장바구니|설정|온보딩|문의|연락처|피드백|검색|예약|관리자|어드민|폼|양식|플로우|모달|패널)/iu
const COMMON_RENDERED_UI_SURFACE_NON_UI_PATTERN =
  /\b(?:parser|scraper|crawler|extractor|function|method|class|library|package|sdk|module|utility|util|cli|command[-\s]?line|terminal|shell|data\s+model|schema|database|migration|api\s+client|backend|endpoint|algorithm|solver|model|dataset|headless|no\s+ui|without\s+(?:a\s+)?ui)\b|(?:파서|스크래퍼|크롤러|추출기|함수|메서드|클래스|라이브러리|패키지|모듈|유틸|터미널|셸|데이터\s*모델|스키마|데이터베이스|마이그레이션|api\s*클라이언트|백엔드|엔드포인트|알고리즘|솔버|모델|데이터셋|ui\s*없이|화면\s*없이)/iu
const VISUAL_UI_REFINEMENT_ACTION_PATTERN =
  /\b(?:redesign|re[-\s]?design|restyle|re[-\s]?style|polish|improve|refine|refresh|moderni[sz]e|beautify)\b|(?:리디자인|다시\s*디자인|재\s*디자인|개선|다듬|정리|예쁘|멋지|꾸며|스타일\s*개선)/iu
const VISUAL_UI_REFINEMENT_SUBJECT_PATTERN =
  /\b(?:home\s*page|homepage|landing\s+page|site|website|web\s*(?:app|page|site)|app|ui|ux|interface|screen|page|view|layout|visual\s+design|design|style|styling|product\s+card|profile\s+screen|card|modal|panel|form|dashboard|navbar|navigation\s+bar|hero\s+section)\b|(?:홈\s*페이지|랜딩\s*페이지|사이트|웹\s*(?:앱|페이지|사이트)|앱|ui|ux|인터페이스|화면|페이지|뷰|레이아웃|시각\s*디자인|디자인|스타일|카드|모달|패널|폼|양식|대시보드|내비|네비|히어로)/iu
const VISUAL_UI_REFINEMENT_NON_UI_PATTERN =
  /\b(?:copy|article|post|blog\s+post|readme|documentation|docs?|architecture|database|schema|api|backend|server|function|parser|data\s+model|essay|summary|text\s+only|no\s+ui|without\s+(?:a\s+)?ui)\b|(?:문구|카피|글|아티클|게시글|문서|아키텍처|데이터베이스|스키마|api|백엔드|서버|함수|파서|데이터\s*모델|요약|ui\s*없이|화면\s*없이)/iu
const VISUAL_UI_FEEDBACK_FOLLOWUP_PATTERN =
  /\b(?:make\s+it\s+look\s+better|looks?\s+(?:bad|broken|ugly|off|wrong|messy|cramped|cluttered|unfinished|unpolished)|look\s+and\s+feel|visual(?:ly)?\s+(?:bad|broken|off|wrong)|overlapping\s+(?:text|controls?|buttons?|elements?)|(?:text|controls?|buttons?|elements?)\s+(?:overlap|overlaps|are\s+overlapping|is\s+overlapping)|(?:buttons?|controls?|text|content|elements?)\s+(?:are\s+)?(?:clipped|cut\s+off|overflowing)|spacing\s+(?:feels\s+)?(?:bad|off|wrong|cramped|tight)|contrast\s+(?:is\s+)?(?:bad|low|poor)|layout\s+(?:is\s+|looks?\s+)?(?:broken|bad|off|wrong|cramped|cluttered))\b|(?:보기\s*(?:안\s*)?좋|(?<![\p{L}\p{N}_])별로|엉망|구려|깨져|겹쳐|겹침|잘려|삐져\s*나|넘쳐|간격이?\s*(?:좁|이상)|대비가?\s*(?:낮|나쁘)|텍스트가?\s*겹|버튼이?\s*잘|(?:ui|ux|디자인|화면|레이아웃)\s*(?:가|이|을|를|은|는)?\s*(?:너무\s*)?(?:고려\s*(?:안|않)|신경\s*(?:안|않)|엉성|허술|아쉽))/iu
const VISUAL_UI_VIEWPORT_FEEDBACK_PATTERN =
  /\b(?:(?:on\s+)?(?:mobile|phone|tablet|handset|small\s+screen|narrow\s+viewport)\b.{0,64}\b(?:broken|bad|off|wrong|cut\s+off|clipped|overflow(?:s|ing)?|cramped|too\s+(?:wide|narrow|small|large)|not\s+(?:work|working|fit|fitting))|(?:mobile|phone|tablet|handset|small\s+screen|narrow\s+viewport)\s+(?:view|layout|viewport|screen)\b.{0,64}\b(?:broken|bad|off|wrong|cut\s+off|clipped|overflow(?:s|ing)?|cramped)|(?:works?|looks?)\s+on\s+desktop\b.{0,64}\bnot\s+on\s+(?:mobile|phone|tablet)|responsive\s+(?:layout|breakpoints?|view|screen)\b.{0,64}\b(?:broken|bad|off|wrong|not\s+(?:work|working|fit|fitting)))\b|(?:(?:모바일|폰|휴대폰|태블릿|작은\s*화면|좁은\s*화면).{0,32}(?:깨져|잘려|넘쳐|안\s*맞|이상|망가|삐져)|반응형.{0,32}(?:깨져|안\s*맞|이상|망가))/iu
const VISUAL_UI_FEEDBACK_FOLLOWUP_NON_UI_PATTERN =
  /\b(?:api|backend|server|endpoint|database|schema|query|cron|job|worker|queue|code|function|method|class|module|package|library|sdk|parser|algorithm|solver|model|dataset|readme|documentation|docs?|copy|article|post|text\s+only|no\s+ui|without\s+(?:a\s+)?ui)\b|(?:백엔드|서버|엔드포인트|데이터베이스|스키마|쿼리|크론|잡|워커|큐|코드|함수|메서드|클래스|모듈|패키지|라이브러리|파서|알고리즘|솔버|모델|데이터셋|문서|카피|문구|글|ui\s*없이|화면\s*없이)/iu
const NAMED_BROWSER_GAME_PATTERN =
  /\b(?:2048|wordle|tetris|pong|breakout|minesweeper|tic[-\s]?tac[-\s]?toe|sudoku|flappy\s+bird|space\s+invaders?|memory\s+(?:match|cards?))\b|\b(?:snake)\b.{0,24}\b(?:clone|game)\b|\b(?:clone|game)\b.{0,24}\b(?:snake)\b|(?:테트리스|워들|퐁|벽돌\s*깨기|지뢰\s*찾기|스도쿠|틱택토|플래피\s*버드|스페이스\s*인베이더|스네이크\s*게임)/iu
const NAMED_BROWSER_GAME_UI_HINT_PATTERN =
  /\b(?:clone|game|app|playable|interactive|keyboard|controls?|browser|web|canvas|ui|responsive)\b|(?:게임|앱|클론|복제|상호작용|인터랙션|키보드|컨트롤|브라우저|웹|캔버스|반응형)/iu
const NAMED_BROWSER_GAME_NON_UI_PATTERN =
  /\b(?:strategy|opening|solver|algorithm|rules?|guide|analysis|essay|article|history|bot|ai|model|dataset|puzzle\s+generator|random\s+puzzle|opener)\b|(?:전략|공략|규칙|분석|역사|알고리즘|솔버|모델|데이터셋)/iu
const CONTEXTUAL_FOLLOWUP_REQUEST_PATTERN =
  /^\s*(?:(?:fix\s+(?:it|this|that)|redo\s+(?:it|this|that)|retry|try\s+again|do\s+(?:it|this|that)\s+again|make\s+(?:it|this|that)\s+better|clean\s+(?:it|this|that)\s+up|polish\s+(?:it|this|that)|improve\s+(?:it|this|that)|update\s+(?:it|this|that)|continue|keep\s+going|same\s+(?:issue|thing)|again|did\s+you\s+(?:actually\s+)?(?:check|inspect|review|validate|test)(?:\s+(?:it|this|that|the\s+(?:result|output|artifact)))?|(?:was|is)\s+(?:it|this|that|the\s+(?:result|output|artifact))\s+(?:actually\s+)?(?:checked|inspected|reviewed|validated|tested))\b|(?:이거|그거|저거|다시|고쳐|수정|개선|계속|(?:산출물|결과물|결과|화면|ui)?\s*(?:검사|검수|확인|테스트|검증)\s*한\s*거\s*맞|(?:제대로|진짜)\s*(?:검사|검수|확인|테스트|검증)(?:했|한)))/iu
const CONTEXTUAL_REMAINING_WORK_PATTERN =
  /^\s*(?:remaining|unfinished|leftover)\s+(?:work|issues?|improvements?|fixes?)\b|^\s*남은\s*(?:작업|문제|개선점?|수정|보완|할\s*일)/iu
const CONTEXTUAL_SCOPED_REPAIR_PATTERN =
  /\b(?:failing\s+tests?|test\s+failures?|observed\s+errors?|reported\s+defects?)\b.{0,48}\b(?:fix|repair|correct)\b|(?:테스트\s*실패|실패\s*원인|관찰된\s*오류|보고된\s*결함)(?:을|를|은|는|이|가)?[^.!?。！？\n]{0,40}(?:수정|고쳐|바로잡)/iu
const CONTEXTUAL_REVIEW_FOLLOWUP_PATTERN =
  /\b(?:(?:did\s+you\s+)?(?:only\s+)?(?:run|do)\s+(?:one|1|a\s+single|single)\s+iteration|(?:one|1|a\s+single|single)\s+iteration\s+(?:only|pass)|review\s+(?:the\s+)?(?:process|progress|steps?|iterations?|result|output|artifact|deliverable)(?:\s+(?:and|\/|&)\s+(?:process|progress|steps?|iterations?|result|output|artifact|deliverable))?|(?:process|progress|steps?|iterations?|result|output|artifact|deliverable)(?:\s+(?:and|\/|&)\s+(?:process|progress|steps?|iterations?|result|output|artifact|deliverable))?\s+review)\b|(?:한\s*번만\s*(?:iteration|이터레이션|반복)\s*(?:돌|했|수행)|한번만\s*(?:iteration|이터레이션|반복)\s*(?:돌|했|수행)|(?:iteration|이터레이션|반복)\s*(?:한\s*번|한번|1\s*번)\s*만|(?:진행\s*과정|진행|과정|프로세스|절차|결과|산출물|결과물|작업\s*흐름)(?:\s*(?:과|와|랑|및|\/|&)\s*(?:진행\s*과정|진행|과정|프로세스|절차|결과|산출물|결과물|작업\s*흐름)){0,2}\s*(?:을|를|도)?\s*(?:리뷰|검토|점검)\s*(?:해|해봐|했|한)?)/iu
const CONTEXTUAL_AUDIT_FOLLOWUP_PATTERN =
  /\b(?:did\s+you\s+(?:actually\s+)?(?:check|inspect|review|validate|test)(?:\s+(?:it|this|that|the\s+(?:result|output|artifact|deliverable)))?|(?:was|is)\s+(?:it|this|that|the\s+(?:result|output|artifact|deliverable))\s+(?:actually\s+)?(?:checked|inspected|reviewed|validated|tested))\b|(?:(?:산출물|결과물|결과|화면|ui)?\s*(?:검사|검수|확인|테스트|검증)\s*한\s*거\s*맞|(?:제대로|진짜)\s*(?:검사|검수|확인|테스트|검증)(?:했|한)|(?:산출물|결과물|결과|화면|ui)\s*(?:을|를)?\s*(?:검토|리뷰|점검)\s*(?:해|해봐|했|한)?)/iu
const CONTEXTUAL_EXECUTION_MODE_FOLLOWUP_PATTERN =
  /(?:(?:do\s+not|don't|dont|stop|avoid)\s+(?:use|using|route(?:\s+through)?|run(?:\s+in)?)\s+(?:ask|chat|question|read[-\s]?only)\s+mode|(?:ask|chat|question|read[-\s]?only)\s+mode\s+instead\s+of\s+(?:interactive|agent|execution|coder|graph)|(?:ask|질문|물어보기)\s*모드(?:로)?\s*(?:하지\s*마|하지\s*말|말고|마라|쓰지\s*마|사용하지\s*마))[\s\S]{0,180}(?:interactive|agent|execution|coder|graph|cli|인터랙티브|인터렉티브|에이전트|실행|수행|코더|그래프)(?:\s*(?:mode|모드))?|(?:interactive|agent|execution|coder|graph|cli|인터랙티브|인터렉티브|에이전트|실행|수행|코더|그래프)\s*(?:mode|모드)(?:로)?\s*(?:run|use|switch|continue|do|go|해|해줘|돌려|진행|사용|전환)/iu
const CONTEXTUAL_FOLLOWUP_REVIEW_CRITERION_PREFIX =
  'The current review/audit follow-up is directly answered'
const CONTEXTUAL_FOLLOWUP_ACTION_CRITERION_PREFIX =
  'The current contextual follow-up is directly handled'
const CONTEXTUAL_FOLLOWUP_CRITERION_PREFIXES = [
  CONTEXTUAL_FOLLOWUP_REVIEW_CRITERION_PREFIX,
  CONTEXTUAL_FOLLOWUP_ACTION_CRITERION_PREFIX,
]
const RENDERED_UI_DESIGN_PLAN_CRITERION =
  'Before implementation, the UI work is grounded in a brief, concrete design plan recorded as a completed todowrite item that names the actual target user/workflow, primary screens and non-empty states, responsive layout approach, visual style direction, expected controls/interactions, and required assets or media; placeholder labels such as "target user/workflow, primary screens, responsive layout, visual style, assets" without task-specific details do not satisfy this; the first rendered screen is the usable product/tool/game rather than a generic marketing shell unless the user specifically requested a landing page.'
const RENDERED_UI_SCREENSHOT_CRITERION =
  'Rendered UI is inspected in a real browser with screenshot images that are actually attached for the next model turn at desktop and mobile viewports, including the primary screen and representative interactive/active states at desktop and mobile sizes when the artifact is interactive (use browser.screenshot waitFor/waitAfterMs for async SPAs, games, media, or animations so the captured frame is ready, use browser.click/browser.evaluate with saved screenshot/layout audits whose output says `Screenshot image attachment: attached` when available, and for games or animated canvas/WebGL work use browser.evaluate to verify frame, pixel, position, or game-state changes over time at desktop and mobile sizes); layout, spacing, text wrapping, color contrast, text/control overlap, overflow, blank-band/cutoff audit warnings, and obvious visual polish issues are fixed or explicitly reported.'
const RENDERED_UI_VISUAL_QA_CRITERION =
  'After screenshots are captured, a visual QA pass compares the rendered result against the design plan at desktop and mobile sizes, explicitly checks concrete visual quality aspects such as layout/spacing, text wrapping/overflow, contrast/readability, overlap/collision, controls/touch targets, and assets/media completeness, records a completed todowrite after the latest browser audit with the concrete issues found or explicitly says none were found, fixes any issue that makes the UI feel unfinished or hard to use, and captures a fresh screenshot after fixes before reporting completion.'
const RENDERED_UI_VALIDATION_ONLY_VISUAL_QA_CRITERION =
  'After screenshots are captured, a visual QA pass assesses the existing rendered product against the requested audit dimensions at desktop and mobile sizes, explicitly checks layout/spacing, text wrapping/overflow, contrast/readability, overlap/collision, controls/touch targets, and assets/media completeness, records the concrete issues found or explicitly says none were found, and captures a fresh screenshot only after an observed visual defect is fixed.'
const RENDERED_UI_SMOKE_CRITERION =
  'Validation includes browser smoke runs of the primary UI, including console/page errors and representative interaction at desktop and mobile viewports when interaction exists; prefer browser.click/browser.evaluate with saved screenshot/layout audits and attached screenshot images for active-state checks instead of changing the app only to make validation convenient; any visible defect or browser.screenshot layout-audit warning triggers a fix and another screenshot before completion, or the result is reported as UNVERIFIED.'
const RENDERED_UI_VALIDATION_ONLY_SMOKE_CRITERION =
  'Validation includes browser smoke runs of the existing primary UI, including console/page errors and representative interaction at desktop and mobile viewports when interaction is requested; observed defects and browser.screenshot layout-audit warnings are explicitly reported without modifying the product, and any evidence that could not be gathered is reported as UNVERIFIED.'
const RENDERED_UI_EVIDENCE_DESCRIPTION =
  'Rendered UI validation must include browser screenshot evidence with attached images whose browser output says `Screenshot image attachment: attached`, layout-audit results at desktop and mobile viewport, readiness waits via browser.screenshot waitFor/waitAfterMs when the UI renders asynchronously or animates into its useful state, representative interactive states at desktop and mobile viewports when applicable (browser.click/browser.evaluate with saved screenshot/layout audits whose output says `Screenshot image attachment: attached`), dynamic browser.evaluate evidence for games or animated canvas/WebGL work showing frame, pixel, position, or game-state changes over time at desktop and mobile viewport, console/page-error smoke results, a completed visual QA todowrite after the latest browser audit comparing the result to the design plan with issues found or explicitly none while checking layout/spacing, text wrapping/overflow, contrast/readability, overlap/collision, controls/touch targets, and assets/media completeness, and a final screenshot after any visual fixes.'
const RENDERED_UI_VALIDATION_ONLY_EVIDENCE_DESCRIPTION =
  'Rendered UI validation must include browser screenshot evidence with attached images whose browser output says `Screenshot image attachment: attached`, layout-audit results at desktop and mobile viewports, readiness waits when the UI renders asynchronously, representative browser.click/browser.evaluate interaction and active-state evidence, dynamic state-change evidence for games or animations, console/page-error results, and a visual QA summary that checks layout/spacing, text wrapping/overflow, contrast/readability, overlap/collision, controls/touch targets, and assets/media completeness; capture a fresh screenshot only after an observed defect is fixed.'

export type RenderedUiRequestedScope = {
  bounded: boolean
  viewportLabel: string
  viewportDimensions: string[]
  excludesMobile: boolean
  excludesDesktop: boolean
  excludesInteraction: boolean
  excludesSemanticVisualInspection: boolean
}

const EXPLICIT_DESKTOP_ONLY_VALIDATION_PATTERN =
  /\b(?:desktop(?:\s+(?:viewport|screen|size))?\s+only|only\s+(?:the\s+)?desktop(?:\s+(?:viewport|screen|size))?|no\s+(?:mobile|phone|tablet)(?:\s+(?:viewport|screen|size|qa|validation|test(?:ing)?))?|without\s+(?:a\s+)?(?:mobile|phone|tablet)(?:\s+(?:viewport|screen|size|qa|validation|test(?:ing)?))?)\b|(?:데스크톱|데스크탑)(?:\s*(?:뷰포트|화면|크기))?\s*만|모바일(?:\s*(?:뷰포트|화면|크기|검수|검증|테스트))?(?:은|는|을|를)?\s*(?:제외|하지\s*마|하지\s*말|없이)/iu
const EXPLICIT_MOBILE_ONLY_VALIDATION_PATTERN =
  /\b(?:mobile(?:\s+(?:viewport|screen|size))?\s+only|only\s+(?:the\s+)?mobile(?:\s+(?:viewport|screen|size))?|no\s+desktop(?:\s+(?:viewport|screen|size|qa|validation|test(?:ing)?))?|without\s+(?:a\s+)?desktop(?:\s+(?:viewport|screen|size|qa|validation|test(?:ing)?))?)\b|모바일(?:\s*(?:뷰포트|화면|크기))?\s*만|(?:데스크톱|데스크탑)(?:\s*(?:뷰포트|화면|크기|검수|검증|테스트))?(?:은|는|을|를)?\s*(?:제외|하지\s*마|하지\s*말|없이)/iu
const EXPLICIT_SINGLE_SCREENSHOT_PATTERN =
  /\b(?:exactly\s+)?(?:one|1|single)\s+(?:browser\s+)?screenshots?\b|\bscreenshots?\s+(?:exactly\s+)?(?:once|one\s+time|1\s+time)\b|(?:browser\.screenshot|스크린샷|화면\s*캡처)(?:은|는|을|를)?\s*(?:정확히\s*)?(?:한\s*번|한번|1\s*회)(?:만)?/iu
const EXPLICIT_SINGLE_SCREENSHOT_INTERACTION_PATTERN =
  /(?:\bbrowser\.(?:click|evaluate)\b(?:\s+(?:exactly\s+)?(?:once|one\s+time|1\s+time))?|\bbrowser\.(?:click|evaluate)\b(?:은|는|을|를)?\s*(?:정확히|딱)?\s*(?:한\s*번|한번|1\s*회)(?:만)?)[\s\S]{0,240}\b(?:screenshotPath|screenshots?|capture)\b|(?:\bbrowser\.(?:click|evaluate)\b(?:은|는|을|를)?\s*(?:정확히|딱)?\s*(?:한\s*번|한번|1\s*회)(?:만)?)[\s\S]{0,240}(?:스크린샷|화면\s*캡처|캡처)/iu
const EXPLICIT_NO_RENDERED_INTERACTION_PATTERN =
  /\b(?:no|without)\s+(?:extra\s+|additional\s+|any\s+)?(?:interaction|click(?:s|ing)?|browser\.(?:click|evaluate)|dynamic\s+state(?:\s+checks?)?)\b|\b(?:do\s+not|don't|dont|never)\s+(?:interact|click|use\s+browser\.(?:click|evaluate)|check\s+dynamic\s+state)\b|(?:상호작용|인터랙션|클릭|동적\s*상태\s*검증)(?:은|는|을|를)?\s*(?:하지\s*마|하지\s*말|제외|없이)/iu
const EXPLICIT_VIEWPORT_DIMENSION_PATTERN = /\b(\d{3,5})\s*[x×]\s*(\d{3,5})\b/giu
const EXPLICIT_STRUCTURED_VIEWPORT_DIMENSION_PATTERN =
  /\bviewportWidth\b\s*(?:=|:)\s*["']?(\d{3,5})["']?[^.!?。！？\n]{0,160}?\bviewportHeight\b\s*(?:=|:)\s*["']?(\d{3,5})["']?/giu
const EXPLICIT_REVERSED_STRUCTURED_VIEWPORT_DIMENSION_PATTERN =
  /\bviewportHeight\b\s*(?:=|:)\s*["']?(\d{3,5})["']?[^.!?。！？\n]{0,160}?\bviewportWidth\b\s*(?:=|:)\s*["']?(\d{3,5})["']?/giu
const EXPLICIT_NO_ADDITIONAL_TOOL_CALL_PATTERN =
  /\b(?:do\s+not|don't|dont|never)\s+(?:call|use|invoke|run|execute)\s+(?:any\s+)?(?:other|additional|extra)\s+(?:tools?|commands?)\b|(?:다른|그\s*외(?:의)?|추가(?:적인)?)\s*(?:도구|툴|명령)(?:를|은|는|도)?\s*(?:사용|실행|호출)하지\s*(?:마|말)/iu
const EXPLICIT_NO_FILESYSTEM_READ_PATTERN =
  /\b(?:no|without)\s+(?:any\s+|extra\s+|additional\s+)?(?:file(?:system)?\s+reads?|reading|listing|searching|globbing)\b|\b(?:do\s+not|don't|dont|never)\s+(?:use\s+)?(?:fs\.(?:read|list|glob|search)|read|list|search|glob)(?:\s+(?:any\s+)?(?:workspace\s+|repository\s+)?files?)?\b|(?:파일|소스)(?:을|를|은|는)?\s*(?:읽|조회|검색|탐색|목록화)(?:지\s*마|지\s*말|하지\s*마|하지\s*말|없이)|(?:파일|소스)\s*(?:읽기|조회|검색|탐색)(?:는|은|을|를)?\s*(?:제외|하지\s*마|하지\s*말|없이)/iu
const EXPLICIT_NO_TERMINAL_PATTERN =
  /\b(?:no|without)\s+(?:any\s+|extra\s+|additional\s+)?(?:terminal|shell|command[- ]line)(?:\s+(?:commands?|calls?|execution))?\b|\b(?:do\s+not|don't|dont|never)\s+(?:use|run|invoke)\s+(?:the\s+)?(?:terminal|shell|terminal\.run)\b|(?:터미널|셸|쉘|명령줄)(?:\s*(?:명령|호출|실행))?(?:은|는|을|를)?\s*(?:사용하지\s*마|실행하지\s*마|제외|없이)|(?:터미널|셸|쉘|명령줄)(?:\s*(?:명령|호출|실행))?\s*(?:과|와|및|,)[^.!?。！？\n]{0,40}(?:은|는|을|를)?\s*하지\s*(?:마|말)/iu
const EXPLICIT_NO_BROWSER_PATTERN =
  /\b(?:no|without)\s+(?:any\s+)?browser(?:\s+(?:tools?|calls?|validation))?\b|\b(?:do\s+not|don't|dont|never)\s+(?:use|open|invoke)\s+(?:the\s+)?browser\b|브라우저(?:\s*(?:도구|호출|검증))?(?:은|는|을|를)?\s*(?:사용하지\s*마|열지\s*마|제외|없이)/iu
const EXPLICIT_NO_PROCESS_TOOLS_PATTERN =
  /\b(?:no|without)\s+(?:any\s+)?process(?:\s+(?:tools?|calls?|operations?))\b|\b(?:do\s+not|don't|dont|never)\s+use\s+process\.\*?\b|프로세스(?:\s*(?:도구|호출|조작))?(?:은|는|을|를)?\s*(?:사용하지\s*마|제외|없이)/iu

function applyExplicitCapabilityBoundaries(
  capabilities: AgentExecutionCapability[],
  input: string,
): AgentExecutionCapability[] {
  const excluded = new Set<AgentExecutionCapability>()
  const explicitScopes = explicitNegativeClauseScopes(input)
  let inlinePeriodSentinel = '\uE100'
  while (input.includes(inlinePeriodSentinel)) inlinePeriodSentinel += '\uE101'
  const protectedPositiveScope = inputPositiveActionScope(input).replace(
    /(?<=[\p{L}\p{N}_-])\.(?=[\p{L}\p{N}_-])/gu,
    inlinePeriodSentinel,
  )
  // Capability prohibitions are broader than mutation prohibitions. Remove
  // each bounded negative clause while preserving any affirmative prefix in
  // the same clause ("navigate once, but do not make another call").
  const positiveScope = protectedPositiveScope
    .replace(
      /(^|[.!?。！？,，;；\n]+\s*)([^.!?。！？,，;；\n]*?)(\b(?:do\s+not|don't|dont|never|no|without)\b[^.!?。！？,，;；\n]*)/giu,
      (_whole, boundary: string, prefix: string) => `${boundary}${prefix.trimEnd()}`,
    )
    .replace(
      /(^|[.!?。！？,，;；\n]+\s*)([^.!?。！？,，;；\n]*?(?:하지\s*(?:마|말(?:아|라)?|마라)|하지\s*않(?:아|도록|는다|습니다)|제외|없이))\s*(?=$|[.!?。！？,，;；\n])/gu,
      (_whole, boundary: string, clause: string) => {
        const conjunction = clause.lastIndexOf('하고')
        const affirmativePrefix = conjunction >= 0 ? clause.slice(0, conjunction) : ''
        const restoredPrefix = affirmativePrefix.split(inlinePeriodSentinel).join('.')
        return /\b(?:process|browser|terminal|fs)\.[a-z][\w.-]*\b/iu.test(restoredPrefix)
          ? `${boundary}${affirmativePrefix.trimEnd()}`
          : boundary
      },
    )
    .split(inlinePeriodSentinel).join('.')
  const mentionsFilesystemRead = (value: string) =>
    /\bfs\.(?:glob|list|read|search)\b|\bfs\.\*(?=$|[^\w])|\b(?:read|list|search|glob)(?:ing)?\s+(?:the\s+)?(?:workspace\s+|repository\s+)?files?\b|(?:파일|소스)\s*(?:읽기|조회|검색|탐색|목록)/iu.test(value)
  const mentionsFilesystemWrite = (value: string) =>
    /\b(?:apply_patch|fs\.(?:append|edit|write))\b|\bfs\.\*(?=$|[^\w])|\b(?:modify|edit|write|create|delete|remove)\s+(?:the\s+)?files?\b|(?:파일|소스)\s*(?:수정|편집|작성|생성|삭제)/iu.test(value)
  const mentionsTerminal = (value: string) =>
    /\bterminal\.run\b|\b(?:terminal|shell|command[- ]line)\b|(?:터미널|셸|쉘|명령줄)/iu.test(value)
  const mentionsBrowser = (value: string) =>
    /\bbrowser\.[a-z][\w.-]*\b|\bbrowser\b|브라우저/iu.test(value)
  const mentionsProcess = (value: string) =>
    /\bprocess\.[a-z][\w.-]*\b|\bprocess\s+(?:tool|call|operation|session|lifecycle)\b|프로세스/iu.test(value)

  // A shared trailing prohibition ("files, terminal.run, ... do not use")
  // is a capability boundary even when no one direct-regex variant happens to
  // enumerate that exact multilingual list. Conversely, "call browser.navigate
  // once; no additional browser calls" retains the explicitly requested call.
  if (
    (mentionsFilesystemRead(input) && !mentionsFilesystemRead(positiveScope))
    || (
      mentionsFilesystemRead(explicitScopes.negativeText)
      && !mentionsFilesystemRead(explicitScopes.positiveText)
    )
    || (EXPLICIT_NO_FILESYSTEM_READ_PATTERN.test(input) && !mentionsFilesystemRead(positiveScope))
  ) excluded.add('filesystem-read')
  if (
    (mentionsFilesystemWrite(input) && !mentionsFilesystemWrite(positiveScope))
    || (
      mentionsFilesystemWrite(explicitScopes.negativeText)
      && !mentionsFilesystemWrite(explicitScopes.positiveText)
    )
  ) {
    excluded.add('filesystem-write')
  }
  if (
    (mentionsTerminal(input) && !mentionsTerminal(positiveScope))
    || (
      mentionsTerminal(explicitScopes.negativeText)
      && !mentionsTerminal(explicitScopes.positiveText)
    )
    || (EXPLICIT_NO_TERMINAL_PATTERN.test(input) && !mentionsTerminal(positiveScope))
  ) excluded.add('terminal')
  if (
    (mentionsBrowser(input) && !mentionsBrowser(positiveScope))
    || (
      mentionsBrowser(explicitScopes.negativeText)
      && !mentionsBrowser(explicitScopes.positiveText)
    )
    || (EXPLICIT_NO_BROWSER_PATTERN.test(input) && !mentionsBrowser(positiveScope))
  ) excluded.add('browser')
  if (
    (mentionsProcess(input) && !mentionsProcess(positiveScope))
    || (
      mentionsProcess(explicitScopes.negativeText)
      && !mentionsProcess(explicitScopes.positiveText)
    )
    || (EXPLICIT_NO_PROCESS_TOOLS_PATTERN.test(input) && !mentionsProcess(positiveScope))
  ) excluded.add('process')
  return capabilities.filter((capability) => !excluded.has(capability))
}

function explicitWorkflowCapabilities(input: string): AgentExecutionCapability[] {
  const capabilities = new Set<AgentExecutionCapability>(['process'])
  if (/\bbrowser\.[a-z][\w.-]*\b/iu.test(input)) capabilities.add('browser')
  if (/\bterminal\.run\b/iu.test(input)) capabilities.add('terminal')
  if (/\bservice\.[a-z][\w.-]*\b/iu.test(input)) capabilities.add('service')
  if (/\bfs\.(?:glob|list|read|search)\b/iu.test(input)) capabilities.add('filesystem-read')
  if (/\b(?:apply_patch|fs\.(?:append|edit|write))\b/iu.test(input)) {
    capabilities.add('filesystem-write')
  }
  // A structured process.start option such as `network.mode=loopback` is not
  // authority for an independent internet/network tool. Only an actual fetch
  // tool or an explicit network-access action grants that capability.
  if (
    /\b(?:webfetch|web\.fetch)\b/iu.test(input)
    || /\b(?:use|allow|perform|make|send)\s+(?:the\s+)?(?:internet|network)(?:\s+access|\s+request)/iu.test(input)
    || /(?:인터넷|네트워크)\s*(?:접근|요청|조회|검색)(?:을|를)?\s*(?:사용|수행|허용|해|하)/u.test(input)
  ) capabilities.add('network')
  return applyExplicitCapabilityBoundaries([...capabilities], input)
}

function explicitReadOnlyInspectionCapabilities(input: string): AgentExecutionCapability[] {
  const positive = inputPositiveCapabilityScope(input)
  const capabilities = new Set<AgentExecutionCapability>()
  if (/\bprocess\.[a-z][\w.-]*\b/iu.test(positive)) capabilities.add('process')
  if (/\bservice\.[a-z][\w.-]*\b/iu.test(positive)) capabilities.add('service')
  if (/\bterminal\.run\b|\b(?:terminal|shell|command[- ]line)\b|(?:터미널|셸|쉘|명령줄)/iu.test(positive)) {
    capabilities.add('terminal')
  }
  if (/\bbrowser\.[a-z][\w.-]*\b|\bbrowser\b|브라우저/iu.test(positive)) {
    capabilities.add('browser')
  }
  if (
    /\bfs\.(?:glob|list|read|search)\b|\b(?:read|list|search|glob)(?:ing)?\s+(?:the\s+)?(?:workspace\s+|repository\s+)?files?\b|(?:파일|소스)\s*(?:읽기|조회|검색|탐색|목록)/iu.test(positive)
  ) {
    capabilities.add('filesystem-read')
  }
  if (
    /\b(?:web\.search|webfetch|jpad(?:\.[a-z][\w.-]*)?)\b|https?:\/\/|\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b|\b(?:web|internet|network)\s+(?:search|lookup|request|access)\b|\b(?:search|look\s*up|query)\s+(?:the\s+)?web\b|(?:웹|인터넷|네트워크|내부)\s*(?:검색|조회|요청|접근)/iu.test(positive)
    || /\b(?:official|public|external|online)\s+(?:docs?|documentation|sources?)\b|(?:공식|공개|외부|온라인)\s*(?:문서|자료|출처)|(?:출처|원문)\s*URL/iu.test(positive)
  ) {
    capabilities.add('network')
  }
  return applyExplicitCapabilityBoundaries([...capabilities], input)
}

function explicitNegativeClauseScopes(input: string): {
  negativeText: string
  positiveText: string
} {
  let inlinePeriodSentinel = '\uE210'
  while (input.includes(inlinePeriodSentinel)) inlinePeriodSentinel += '\uE211'
  const protectedInput = input
    .replace(/\bfs\.(?=\*)/giu, `fs${inlinePeriodSentinel}`)
    .replace(
      /(?<=[\p{L}\p{N}_-])\.(?=[\p{L}\p{N}_-])/gu,
      inlinePeriodSentinel,
    )
  const negativeClauses: string[] = []
  const positiveProtected = protectedInput.replace(
    /(^|[.!?。！？;；\n]+\s*)([^.!?。！？;；\n]+)/gu,
    (whole, boundary: string, clause: string) => {
      const englishNegative = /\b(?:do\s+not|don't|dont|never|no|without)\b/iu.exec(clause)
      const englishDeclarativeNegative =
        /\b(?:is|are|was|were)\s+not\s+(?:requested|required|needed|included|in\s+scope)\b/iu.exec(clause)
      const koreanDeclarativeNegative =
        /(?:은|는|이|가)?\s*아니(?:다|야|에요|예요|고|며|라)|아님/u.exec(clause.trim())
      const koreanNegative = /(?:[\p{Script=Hangul}]+지\s*(?:마|말(?:아|라|고)?|마라|않(?:아|고|도록|는다|습니다))|제외(?:하고)?|없이)(?=\s|,|$)/u
        .exec(clause.trim())
      const englishProhibition = /\b(?:is|are)\s+(?:forbidden|prohibited|disallowed)\b/iu
        .exec(clause.trim())
      const koreanProhibition = /(?:금지(?:한다|합니다|해|하(?:라|세요|도록)?|됨|입니다)?|불가(?:해|함|입니다)?)(?=\s|,|$)/u
        .exec(clause.trim())
      if (
        !englishNegative
        && !englishDeclarativeNegative
        && !koreanDeclarativeNegative
        && !koreanNegative
        && !englishProhibition
        && !koreanProhibition
      ) return whole

      negativeClauses.push(clause.split(inlinePeriodSentinel).join('.').trim())
      // In “X is not requested”, X itself is the negated subject. Keeping the
      // prefix would turn that subject back into a positive requirement in
      // downstream capability/evidence matchers.
      if (englishDeclarativeNegative || englishProhibition) return boundary
      if (koreanDeclarativeNegative) {
        const negativeEnd = koreanDeclarativeNegative.index
          + koreanDeclarativeNegative[0].length
        const trailingPositive = clause.slice(negativeEnd)
          .replace(/^\s*(?:,|그리고|대신|그\s*뒤|이후)?\s*/u, '')
        return trailingPositive.trim() ? `${boundary}${trailingPositive.trimStart()}` : boundary
      }
      if (englishNegative) {
        const beforeNegative = clause.slice(0, englishNegative.index).trimEnd()
        const afterNegative = clause.slice(englishNegative.index + englishNegative[0].length)
        const explicitPositiveTransition = /(?:^|,|\s)\s*(?:but|however|instead|then|after(?:ward)?s?|subsequently)\b\s*/iu
          .exec(afterNegative)
        // A leading `without <negative gerund>, <positive imperative>` is a
        // bounded adjunct rather than a prohibition over the whole sentence.
        // Other leading negative imperatives keep comma-coordinated actions
        // in the same negative scope unless an explicit contrast/sequence
        // marker opens a new affirmative scope.
        const withoutComma = englishNegative.index === 0
          && /^without$/iu.test(englishNegative[0])
          ? afterNegative.indexOf(',')
          : -1
        const trailingCandidate = explicitPositiveTransition
          ? afterNegative.slice(
              explicitPositiveTransition.index + explicitPositiveTransition[0].length,
            )
          : withoutComma >= 0
            ? afterNegative.slice(withoutComma + 1).replace(/^\s*(?:and\s+)?/iu, '')
            : ''
        const trailingPositive = trailingCandidate.trim()
          ? explicitNegativeClauseScopes(trailingCandidate).positiveText.trim()
          : ''
        const retained = [beforeNegative, trailingPositive].filter(Boolean).join(' ')
        return retained ? `${boundary}${retained}` : boundary
      }
      if (koreanNegative) {
        const negativeStart = koreanNegative.index
        const negativeEnd = negativeStart + koreanNegative[0].length
        const beforeNegative = clause.slice(0, negativeStart)
        const afterNegative = clause.slice(negativeEnd)
          .replace(/^\s*(?:,|그리고|대신|그\s*뒤|이후)?\s*/u, '')
        // A comma-delimited prefix normally belongs to the shared negative
        // subject list: "screenshots, file edits, other tools ... 하지 마".
        // Preserve a prefix only when it ends in an affirmative Korean action
        // connective followed by one bounded negated target, as in "테스트를
        // 실행하고 파일은 수정하지 말고 결과만 보고해".
        const positivePrefix = beforeNegative.match(
          /^(.*?(?:호출|확인|실행|검사|검증|열|읽|작성|시작|수정|편집|변경|삭제|생성|배포|전송|게시|발행|중지|종료|보내)(?:하고|해서|한\s*뒤|후|고))\s*(?:,\s*)?[^,]{1,80}$/iu,
        )?.[1]?.trimEnd() ?? ''
        if (afterNegative.trim()) {
          return `${boundary}${[positivePrefix, afterNegative.trimStart()].filter(Boolean).join(' ')}`
        }
        if (positivePrefix) {
          return `${boundary}${positivePrefix}`
        }
      }
      if (koreanProhibition) return boundary
      return boundary
    },
  )
  return {
    negativeText: negativeClauses.join('\n'),
    positiveText: positiveProtected.split(inlinePeriodSentinel).join('.'),
  }
}

/**
 * Return the affirmative capability/tool scope of a request. Unlike
 * inputPositiveActionScope (which is intentionally mutation-focused), this
 * removes whole bounded clauses such as "no terminal.run or fs.*" and
 * "process.read는 사용하지 마" so downstream evidence guards do not turn a
 * prohibited tool mention into a required action.
 */
export function inputPositiveCapabilityScope(input: string): string {
  return explicitNegativeClauseScopes(input).positiveText
}

/**
 * Tool ceilings are an authority boundary, so planner prose such as
 * "search" or "read" cannot be treated as a registry identity. Preserve
 * only canonical dotted names (plus apply_patch's established identifier)
 * that the user actually wrote in the affirmative scope of the request.
 */
export function explicitCanonicalToolNames(
  input: string,
  availableToolNames?: readonly string[],
): ReadonlySet<string> {
  const positive = inputPositiveCapabilityScope(input)
  if (availableToolNames) {
    const available = new Map(
      availableToolNames
        .map((name) => name.trim().toLowerCase())
        .filter(Boolean)
        .map((name) => [name, name]),
    )
    const mentioned = new Set<string>()
    for (const match of positive.matchAll(/[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*/giu)) {
      const normalized = match[0].toLowerCase()
      if (available.has(normalized)) mentioned.add(normalized)
    }
    return mentioned
  }
  return new Set(
    [...positive.matchAll(/\b(?:apply_patch|[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)+)\b/giu)]
      .map((match) => match[0].toLowerCase()),
  )
}

/**
 * Rendered-UI QA has useful broad defaults, but those defaults are subordinate
 * to an explicit validation boundary. A user asking for one named viewport or
 * excluding interaction has already made the product decision; the contract
 * may not silently turn that request into a desktop+mobile interaction suite.
 */
export function renderedUiRequestedScope(input: string): RenderedUiRequestedScope {
  const { negativeText, positiveText } = explicitNegativeClauseScopes(input)
  const excludesMobile = EXPLICIT_DESKTOP_ONLY_VALIDATION_PATTERN.test(input)
    || /\b(?:mobile|phone|tablet)\b|모바일|휴대폰|태블릿/iu.test(negativeText)
  const excludesDesktop = EXPLICIT_MOBILE_ONLY_VALIDATION_PATTERN.test(input)
    || /\bdesktop\b|데스크톱|데스크탑/iu.test(negativeText)
  const dimensions = [
    ...[...input.matchAll(EXPLICIT_VIEWPORT_DIMENSION_PATTERN)]
      .map((match) => `${match[1]}×${match[2]}`),
    ...[...input.matchAll(EXPLICIT_STRUCTURED_VIEWPORT_DIMENSION_PATTERN)]
      .map((match) => `${match[1]}×${match[2]}`),
    ...[...input.matchAll(EXPLICIT_REVERSED_STRUCTURED_VIEWPORT_DIMENSION_PATTERN)]
      .map((match) => `${match[2]}×${match[1]}`),
  ]
  const uniqueDimensions = [...new Set(dimensions)]
  const singleNamedViewport = uniqueDimensions.length === 1
    && (
      EXPLICIT_SINGLE_SCREENSHOT_PATTERN.test(input)
      || EXPLICIT_SINGLE_SCREENSHOT_INTERACTION_PATTERN.test(input)
    )
  const bounded = excludesMobile || excludesDesktop || singleNamedViewport
  const viewportLabel = singleNamedViewport
    ? `the explicitly requested ${uniqueDimensions[0]} viewport`
    : excludesMobile
      ? 'the explicitly requested desktop viewport(s)'
      : excludesDesktop
        ? 'the explicitly requested mobile viewport(s)'
        : 'desktop and mobile viewports'
  return {
    bounded,
    viewportLabel,
    viewportDimensions: uniqueDimensions,
    excludesMobile,
    excludesDesktop,
    excludesInteraction: EXPLICIT_NO_RENDERED_INTERACTION_PATTERN.test(input)
      || /\b(?:interaction|click|hover|browser\.(?:click|evaluate)|dynamic\s+state)\b|상호작용|인터랙션|클릭|호버|동적\s*상태/iu.test(negativeText)
      || (
        EXPLICIT_NO_ADDITIONAL_TOOL_CALL_PATTERN.test(input)
        && !EXPLICIT_RENDERED_UI_INTERACTION_REQUEST_PATTERN.test(positiveText)
      ),
    // A capture can be requested purely as a durable browser artifact while
    // the user intentionally limits validation to tool-returned DOM/canvas and
    // console metadata. Preserve that distinction: the screenshot request is
    // still positive, but pixel/semantic visual QA remains out of scope.
    excludesSemanticVisualInspection:
      /\b(?:semantic|pixel(?:[- ]level)?)\s+(?:visual\s+)?(?:inspection|review|qa|analysis)\b|\bvisual\s+(?:inspection|review|qa)\b|(?:의미론적|픽셀(?:\s*수준)?)\s*(?:시각\s*)?(?:검사|검수|리뷰|분석)/iu.test(negativeText),
  }
}

function renderedUiCriterionExceedsRequestedScope(
  text: string,
  scope: RenderedUiRequestedScope,
): boolean {
  if (!scope.bounded && !scope.excludesInteraction) return false
  if (
    scope.viewportDimensions.length > 0
    && (
      /\bdesktop\b[^.!?。！？\n]{0,120}\bmobile\b|\bmobile\b[^.!?。！？\n]{0,120}\bdesktop\b|(?:데스크톱|데스크탑)[^.!?。！？\n]{0,120}모바일|모바일[^.!?。！？\n]{0,120}(?:데스크톱|데스크탑)/iu.test(text)
      || [...text.matchAll(EXPLICIT_VIEWPORT_DIMENSION_PATTERN)]
        .some((match) => !scope.viewportDimensions.includes(`${match[1]}×${match[2]}`))
    )
  ) {
    return true
  }
  if (scope.excludesMobile && /\b(?:mobile|phone|tablet|touch\s+targets?)\b|(?:모바일|휴대폰|태블릿|터치\s*타깃)/iu.test(text)) {
    return true
  }
  if (scope.excludesDesktop && /\bdesktop\b|(?:데스크톱|데스크탑)/iu.test(text)) {
    return true
  }
  if (
    scope.excludesSemanticVisualInspection
    && /\b(?:semantic|pixel(?:[- ]level)?)\s+(?:visual\s+)?(?:inspection|review|qa|analysis)\b|\bvisual\s+(?:inspection|review|qa|summary)\b|\b(?:contrast|readability|assets?\/media completeness)\b|(?:의미론적|픽셀(?:\s*수준)?)\s*(?:시각\s*)?(?:검사|검수|리뷰|분석)|시각\s*(?:검수|리뷰)|대비|가독성/iu.test(text)
  ) {
    return true
  }
  return scope.excludesInteraction
    && /\b(?:interaction|interact|click|dynamic\s+state|browser\.(?:click|evaluate)|game-state\s+changes?)\b|(?:상호작용|인터랙션|클릭|동적\s*상태)/iu.test(text)
}

function scopedRenderedUiScreenshotCriterion(
  scope: RenderedUiRequestedScope,
  captureOnly: boolean,
): string {
  if (captureOnly) {
    return `The requested browser screenshot is saved with its image attachment at ${scope.viewportLabel}, and the browser-returned DOM/canvas layout plus console/page metadata is reported without adding unrequested viewports, interactions, or semantic pixel inspection.`
  }
  if (!scope.bounded && !scope.excludesInteraction) return RENDERED_UI_SCREENSHOT_CRITERION
  const activeState = scope.excludesInteraction
    ? ''
    : ' and representative interactive/active state evidence when the request includes interaction'
  return `Rendered UI is inspected in a real browser with screenshot images actually attached for the next model turn at ${scope.viewportLabel}${activeState}; use readiness waits for asynchronous rendering, audit layout, spacing, text wrapping, color contrast, overlap, overflow, blank bands, and cutoff, and fix or explicitly report observed defects without adding unrequested viewports or interactions.`
}

function scopedRenderedUiVisualQaCriterion(
  scope: RenderedUiRequestedScope,
  validationOnly: boolean,
): string {
  if (!scope.bounded && !scope.excludesInteraction) {
    return validationOnly
      ? RENDERED_UI_VALIDATION_ONLY_VISUAL_QA_CRITERION
      : RENDERED_UI_VISUAL_QA_CRITERION
  }
  return `After screenshots are captured, a visual QA pass assesses the rendered result at ${scope.viewportLabel}, explicitly checking layout/spacing, text wrapping/overflow, contrast/readability, overlap/collision, and assets/media completeness; record concrete issues or explicitly say none were found, and capture a fresh screenshot only after an observed defect is fixed. Do not expand validation to an excluded viewport or interaction state.`
}

function scopedRenderedUiSmokeCriterion(
  scope: RenderedUiRequestedScope,
  validationOnly: boolean,
  interactionRequested: boolean,
): string {
  if (!scope.bounded && !scope.excludesInteraction) {
    return validationOnly
      ? RENDERED_UI_VALIDATION_ONLY_SMOKE_CRITERION
      : RENDERED_UI_SMOKE_CRITERION
  }
  const interaction = interactionRequested && !scope.excludesInteraction
    ? ' and requested representative interaction'
    : ''
  const defectHandling = validationOnly
    ? 'observed defects and screenshot layout-audit warnings are explicitly reported without modifying the product'
    : 'any visible defect or screenshot layout-audit warning triggers a fix and another screenshot before completion'
  return `Validation includes a browser smoke run at ${scope.viewportLabel}, including console/page errors${interaction}; ${defectHandling}, and any evidence that could not be gathered is reported as UNVERIFIED. Do not add excluded viewports or interaction work.`
}

function scopedRenderedUiEvidenceDescription(
  scope: RenderedUiRequestedScope,
  validationOnly: boolean,
): string {
  if (scope.excludesSemanticVisualInspection) {
    return `Capture-only rendered UI evidence must include the requested attached browser screenshot at ${scope.viewportLabel} plus the browser-returned DOM/canvas layout, viewport, page title, and console/page-error metadata. Semantic or pixel-level visual inspection is excluded, and evidence must not add another viewport, interaction, or tool call.`
  }
  if (!scope.bounded && !scope.excludesInteraction) {
    return validationOnly
      ? RENDERED_UI_VALIDATION_ONLY_EVIDENCE_DESCRIPTION
      : RENDERED_UI_EVIDENCE_DESCRIPTION
  }
  const interaction = scope.excludesInteraction
    ? ''
    : ', requested interaction/active-state evidence when applicable'
  return `Rendered UI validation must include an attached browser screenshot whose output says \`Screenshot image attachment: attached\`, layout-audit results at ${scope.viewportLabel}, readiness waits when rendering is asynchronous${interaction}, console/page-error results, and a visual QA summary covering layout/spacing, text wrapping/overflow, contrast/readability, overlap/collision, and assets/media completeness. Evidence must remain within the explicitly requested viewport and interaction scope.`
}

const RESPONSE_ONLY_MUTATION_ACTION_PATTERN = new RegExp([
  // A bounded verb/object/destination phrase names the response channel.
  // Never cross a conjunction or another mutation action: a separate save
  // request must retain its authority, and paths are preserved below.
  String.raw`\b(?:write|draft|provide|include|show|return|put|output|emit)\s+(?:(?!(?:and|or|then|after|before|create|update|save|publish|post|send|write|edit|modify|delete|remove)\b)[\p{L}\p{N}_-]+\s+){0,6}(?:in|as|within|to)\s+(?:(?:the|your|this)\s+)?(?:answer|response|reply|chat|conversation)\b`,
  String.raw`\b(?:answer|respond|reply)\s+(?:only\s+)?with\b`,
  // A response deliverable can carry descriptive modifiers (for example,
  // "return a complete Korean Markdown draft"). Keep the span bounded and
  // refuse to cross conjunctions or durable mutation verbs, so a later
  // "and save/create ..." remains visible as real write authority.
  String.raw`\b(?:return(?:\s+only)?|provide\s+only|show\s+only)\s+(?:(?:a|an|the)\s+)?(?:(?!(?:and|or|then|after(?:ward)?s?|before|create|update|save|publish|post|send|write|edit|modify|delete|remove)\b)[\p{L}\p{N}_-]+\s+){0,6}(?:draft|analysis|summary|report|markdown|code|content)\b`,
  String.raw`(?:답변|응답|채팅|대화)(?:에|으로|에서|로)\s*(?:만\s*)?(?:작성|제시|포함|출력|보여\s*주|써)(?:해|하|해줘|해주세요|한다|하세요|줘)?`,
  String.raw`(?:(?:마지막|최종)\s*)?(?:답변|응답|회신)(?:은|는|을|를)?\s*(?:작성|제시|출력|보여\s*주|써)(?:해|하|해줘|해주세요|한다|하세요|줘)?`,
  String.raw`(?:(?:마지막|최종)\s*)?(?:답변|응답)(?:은|는|을|를)?[^.!?。！？\n]{0,80}?(?:bullet|불릿|글머리표|\d+\s*(?:개|줄|문장)|이내)[^.!?。！？\n]{0,32}?(?:작성|제시|포함|출력|보여\s*주|써)(?:해|하|해줘|해주세요|한다|하세요|줘)?`,
  String.raw`(?:초안|분석|요약|보고서|마크다운|코드|내용)(?:을|를)?[^.!?。！？\n]{0,48}?(?:(?:최종|마지막)\s*)?(?:답변|응답|회신|채팅)(?:으로|에|에서)\s*(?:만\s*)?(?:반환|작성|제시|포함|출력|보여\s*주|써)(?:해|하|해줘|해주세요|한다|하세요|줘)?`,
].join('|'), 'giu')

function stripResponseOnlyMutationActions(input: string): string {
  RESPONSE_ONLY_MUTATION_ACTION_PATTERN.lastIndex = 0
  return input.replace(RESPONSE_ONLY_MUTATION_ACTION_PATTERN, (match) => {
    // Response formatting is presentation, but a path inside the same phrase
    // (for example, "write the answer to report.md") still grants durable
    // file authority and must remain visible to artifact intent detection.
    const durablePath = Array.from(match.matchAll(fallbackArtifactPathMatcher()))
      .map((candidate) => normalizeArtifactPath(candidate[1] ?? ''))
      .some((path) => Boolean(path) && hasFallbackDurableArtifactExtension(path))
    return durablePath ? match : ' '
  })
}

/** A response body is presentation, not authority to create a durable file. */
export function inputRequestsResponseOnlyOutput(input: string): boolean {
  RESPONSE_ONLY_MUTATION_ACTION_PATTERN.lastIndex = 0
  return RESPONSE_ONLY_MUTATION_ACTION_PATTERN.test(inputPositiveActionScope(input))
}

/**
 * Whether the user request expresses a mutation/authoring intent (edit, fix,
 * create, update, write, …), reusing the SAME contract-planner intent lexicon
 * as fallback artifact detection — this is not a new heuristic, it is the shared
 * structural signal that already governs artifact intent. Used by the mode
 * router to promote a read-only graph to an artifact writer when the request
 * clearly asks for a change but names no output file (so no durable contract
 * exists to force the reroute). Built with a fresh non-global RegExp so the
 * shared pattern's `g`/`lastIndex` state cannot leak into a boolean test.
 */
export function inputExpressesMutationIntent(input: string): boolean {
  if (!input) return false
  if (
    isFocusedRepositoryChangeLookup(input)
    || isSubstantiveRepositoryChangeReview(input)
  ) return false
  let positiveScope = stripResponseOnlyMutationActions(inputPositiveActionScope(input))
    .replace(BUILD_VALIDATION_ACTION_PATTERN, ' ')
    .replace(CLI_OUTPUT_FORMAT_OPTION_PATTERN, ' ')
    .replace(OBSERVED_OUTPUT_NOUN_PATTERN, ' ')
    .replace(REFERENCED_AUTHORING_CONCEPT_PATTERN, ' ')
    .replace(EXISTING_STATE_AUTHORING_PARTICIPLE_PATTERN, ' ')
    .replace(REFERENCED_DEVELOPMENT_ENVIRONMENT_PATTERN, ' ')
    .replace(REFERENCED_DEVELOPMENT_OPERATIONAL_STATUS_PATTERN, ' ')
  const renderedValidationRequest =
    EXPLICIT_RENDERED_UI_VALIDATION_ACTION_PATTERN.test(input)
    && (
      EXPLICIT_RENDERED_UI_VALIDATION_EVIDENCE_PATTERN.test(input)
      || (
        EXPLICIT_BROWSER_RENDERED_SURFACE_PATTERN.test(input)
        && EXPLICIT_RENDERED_UI_VALIDATION_TARGET_PATTERN.test(input)
      )
    )
  if (renderedValidationRequest) {
    positiveScope = positiveScope.replace(RENDERED_UI_EVIDENCE_ARTIFACT_ACTION_PATTERN, ' ')
  }
  return new RegExp(FALLBACK_ARTIFACT_INTENT_PATTERN.source, 'iu').test(positiveScope)
}

const WORKSPACE_ENGINEERING_ACTION_PATTERN =
  /\b(?:debug|fix|implement|migrate|patch|refactor|scaffold)\b|(?:디버그|고쳐|구현|마이그레이션|패치|리팩터|스캐폴드)(?:링|해|하|를|링을|링해)?/iu
const WORKSPACE_ENGINEERING_TARGET_PATTERN =
  /\b(?:api|application|app|backend|build|cli|code|compiler|dependency|frontend|function|library|module|package|provider|repository|runtime|server|service|source|test|tui|ui|website|workspace)\b|\.(?:c|cc|cpp|cs|go|java|js|jsx|kt|kts|php|py|rb|rs|swift|ts|tsx)\b|(?:API|애플리케이션|앱|백엔드|빌드|CLI|코드|컴파일러|의존성|프론트엔드|함수|라이브러리|모듈|패키지|프로바이더|저장소|런타임|서버|서비스|소스|테스트|TUI|UI|웹사이트|워크스페이스)/u

/**
 * A conservative deterministic signal for source-producing engineering work.
 *
 * This is used only when an LLM router cannot return a usable decision. It
 * deliberately requires the shared positive mutation signal plus either an
 * implementation/debugging action or a concrete software target. Generic
 * writing and creative requests therefore remain on the general fallback,
 * while coding requests do not enter another LLM-backed routing layer.
 */
export function inputExpressesWorkspaceEngineeringIntent(input: string): boolean {
  if (!inputExpressesMutationIntent(input)) return false
  const positiveScope = inputPositiveActionScope(input)
  if (WORKSPACE_ENGINEERING_ACTION_PATTERN.test(positiveScope)) return true
  return WORKSPACE_ENGINEERING_TARGET_PATTERN.test(positiveScope)
    && !inputRequestsDurableDocument(input)
}

function stripSharedNegatedActionListClauses(input: string): string {
  return input.replace(
    /(^|[.!?。！？;；\n]+\s*)([^.!?。！？;；\n]+)/gu,
    (whole, boundary: string, clause: string) => {
      const trimmed = clause.trim()
      const englishNegative = /\b(?:do\s+not|don't|dont|never)\b/iu.exec(trimmed)
      const trailingKoreanNegative = /(?:[\p{Script=Hangul}]+지\s*(?:마(?:라|세요)?|말(?:아|라|아\s*주세요|아주세요|아줘)?|않(?:아|도록|는다|습니다))|금지(?:한다|합니다|해|하(?:라|세요|도록)?|됨|입니다)?|불가(?:해|함|입니다)?)\s*$/u.test(trimmed)
      const negativeScope = englishNegative
        ? trimmed.slice(englishNegative.index)
        : trimmed
      const sharedList = /[,，;；]|\b(?:and|or)\b|(?:\s및\s|\s또는\s|(?:와|과|이나|나|거나|든지)\s)/iu.test(negativeScope)
      const mutationAction = new RegExp(FALLBACK_ARTIFACT_INTENT_PATTERN.source, 'iu').test(negativeScope)
        || /\b(?:create|update|publish|post|send|write|edit|modify|delete|remove)\b|(?:쓰기|작성|생성|수정|변경|삭제|발행|게시|전송)/iu.test(negativeScope)
      const positiveSequence = /\b(?:but|then|after(?:ward)?s?)\b|(?:하지만|그러나|그\s*다음|이후|(?:수정|편집|변경|고치)(?:하|해|한)?고|(?:수정|편집|변경|고치)(?:하|해|한)?서)/iu.test(negativeScope)

      // Negative lists can share a final predicate: “do not take screenshots,
      // edit files, or run QA” and “스크린샷, 파일 수정, QA는 하지 마”.
      // Treat that bounded clause as a capability boundary. A positive
      // sequence/contrast keeps the clause intact so “edit, then do not test”
      // still authorizes the edit.
      if (
        sharedList
        && mutationAction
        && !positiveSequence
        && (englishNegative || trailingKoreanNegative)
      ) {
        if (englishNegative && englishNegative.index > 0) {
          const originalIndex = clause.toLowerCase().indexOf(englishNegative[0].toLowerCase())
          return `${boundary}${clause.slice(0, Math.max(0, originalIndex)).trimEnd()}`
        }
        return boundary
      }
      return whole
    },
  )
}

function stripDirectNegatedMutationSegments(input: string): string {
  const korean = input.replace(
    /(^|[.!?。！？,，;；\n]+\s*)([^.!?。！？,，;；\n]{0,120}?(?:수정|편집|변경|건드리|고치|삭제|지우)(?:은|는|을|를)?\s*(?:하지\s*(?:마|말고|말(?:아|라)?|않)|말고|없이|않(?:고|도록|는다|습니다)))\s*(?=$|[,，;；.!?。！？\n])/giu,
    (_whole, boundary: string) => boundary,
  )
  return korean.replace(
    /(^|[.!?;,\n]+\s*)((?:do\s+not|don't|dont|never)\s+(?:directly\s+)?(?:modify|edit|touch|change|write\s+to|delete|remove|mutate)\b[^.!?;,\n]*)(?=$|[.!?;,\n])/giu,
    (_whole, boundary: string) => boundary,
  )
}

function stripExplicitlyNegatedMutationActions(input: string): string {
  // A period inside a dotted tool/API name, domain, or filename is not a
  // sentence boundary. Protect those inline periods while the bounded-clause
  // scanners run; otherwise a negative list such as
  // "files, terminal.run, browser tools ... do not use" is split at
  // `terminal.run` and its earlier file nouns leak back into positive scope.
  // Use a per-call sentinel so user text is round-tripped without reserving a
  // character globally.
  let inlinePeriodSentinel = '\uE000'
  while (input.includes(inlinePeriodSentinel)) inlinePeriodSentinel += '\uE001'
  const protectedInput = input.replace(
    /(?<=[\p{L}\p{N}_-])\.(?=[\p{L}\p{N}_-])/gu,
    inlinePeriodSentinel,
  )
  EXPLICITLY_NEGATED_MUTATION_ACTION_PATTERN.lastIndex = 0
  const stripped = stripDirectNegatedMutationSegments(
    stripSharedNegatedActionListClauses(protectedInput),
  )
    .replace(EXPLICITLY_NEGATED_MUTATION_ACTION_PATTERN, ' ')
  return stripped.split(inlinePeriodSentinel).join('.')
}

/**
 * Return only positive current-turn action scope after removing structurally
 * bounded negative mutation lists. Other capability guards reuse this so a
 * later “do not edit files” clause cannot become a separate file action.
 */
export function inputPositiveActionScope(input: string): string {
  return stripExplicitlyNegatedMutationActions(input)
}

export function inputExplicitlyForbidsFileMutation(input: string): boolean {
  EXPLICITLY_NEGATED_MUTATION_ACTION_PATTERN.lastIndex = 0
  return EXPLICITLY_NEGATED_MUTATION_ACTION_PATTERN.test(input)
    || inputPositiveActionScope(input) !== input
}

export function inputRequestsFocusedSingleBrowserObservation(input: string): boolean {
  // Tool/capability exclusions such as "do not take screenshots or edit
  // files" are broader than mutation-only exclusions.  Classifying against
  // the capability-positive scope prevents an exact one-tool browser request
  // from becoming a multi-tool workflow merely because the user named tools
  // they explicitly prohibited.
  const positiveInput = inputPositiveCapabilityScope(input)
  const toolMentions = positiveInput.match(FOCUSED_SINGLE_BROWSER_OBSERVATION_PATTERN) ?? []
  const allToolMentions = positiveInput.match(STRUCTURED_BUILTIN_TOOL_REFERENCE_PATTERN) ?? []
  return toolMentions.length === 1
    && allToolMentions.length === 1
    && FOCUSED_SINGLE_OPERATION_BOUNDARY_PATTERN.test(positiveInput)
    && FOCUSED_BROWSER_OBSERVATION_TARGET_PATTERN.test(positiveInput)
    && inputExplicitlyRequestsRenderedUiValidation(positiveInput)
    && !EXPLICIT_RENDERED_UI_SCREENSHOT_REQUEST_PATTERN.test(positiveInput)
    && !EXPLICIT_RENDERED_UI_VISUAL_QA_REQUEST_PATTERN.test(positiveInput)
    && !EXPLICIT_RENDERED_UI_SMOKE_REQUEST_PATTERN.test(positiveInput)
    && !inputExpressesMutationIntent(positiveInput)
}

/**
 * A conversational analysis/review request does not authorize creating a file.
 * When no path is named, require both an authoring action and a document-shaped
 * deliverable before a planner may propose a persistent document artifact.
 */
export function inputRequestsDurableDocument(input: string): boolean {
  if (documentArtifacts(extractPathLikeArtifacts(input)).length > 0) {
    return true
  }
  return splitArtifactClauses(input).some((clause) =>
    !NEGATED_DURABLE_DOCUMENT_AUTHORING_PATTERN.test(clause)
    && (
      EXPLICIT_DOCUMENTATION_ACTION_PATTERN.test(clause)
      || (
        DURABLE_DOCUMENT_AUTHORING_PATTERN.test(clause)
        && DURABLE_DOCUMENT_DELIVERABLE_PATTERN.test(clause)
      )
    ),
  )
}

function inputHasDocumentPhaseClause(input: string): boolean {
  if (
    EXPLICIT_DOCUMENTATION_ACTION_PATTERN.test(input)
    || documentArtifacts(extractPathLikeArtifacts(input)).length > 0
  ) {
    return true
  }
  return splitArtifactClauses(input).some((clause) => {
    const documentSubject = DURABLE_DOCUMENT_DELIVERABLE_PATTERN.test(clause)
    const positiveAuthoring = DURABLE_DOCUMENT_AUTHORING_PATTERN.test(clause)
      && !/\b(?:do\s+not|don't|dont|never)\b[^;]{0,80}\b(?:author|create|draft|generate|produce|save|write)\b|(?:작성|생성|저장|만들|기록)[^;；]{0,40}?(?:하지\s*(?:마|말|않)|말고|없이)/iu.test(clause)
    const positiveObservation = /\b(?:check|consult|inspect|open|read|review)\b|(?:확인|검토|리뷰|읽|열어|조회)/iu.test(clause)
      && !/\b(?:do\s+not|don't|dont|never)\b[^;]{0,80}\b(?:check|consult|inspect|open|read|review)\b|(?:확인|검토|리뷰|읽|열어|조회)[^;；]{0,40}?(?:하지\s*(?:마|말|않)|말고|없이)/iu.test(clause)
    return documentSubject && (positiveAuthoring || positiveObservation)
  })
}

function inputExpressesRenderedUiAuthoringIntent(input: string): boolean {
  return inputExpressesMutationIntent(input)
    || /\b(?:build|design|implement|develop|code|clone|port|ship)\b|(?:디자인|설계|구현|개발|코딩|클론|복제)/iu.test(input)
}

function clauseExpressesRenderedUiAuthoring(
  clause: string,
  subjectPattern: RegExp,
  nonUiPattern?: RegExp,
): boolean {
  if (!inputExpressesRenderedUiAuthoringIntent(clause)) return false
  if (!subjectPattern.test(clause) || nonUiPattern?.test(clause)) return false
  // A page/app name can be the object being inspected while the only authored
  // object is a report: "inspect the JPAD page list and write an audit report".
  // Do not let those two roles combine into UI implementation authority. A
  // direct UI-authoring phrase still wins for composite requests that really
  // ask to build a dashboard and also write documentation.
  return !DURABLE_DOCUMENT_DELIVERABLE_PATTERN.test(clause)
    || DIRECT_RENDERED_UI_AUTHORING_PATTERN.test(clause)
}

function inputHasRenderedUiAuthoringClause(
  input: string,
  subjectPattern: RegExp,
  nonUiPattern?: RegExp,
): boolean {
  return splitArtifactClauses(inputPositiveActionScope(input)).some((clause) =>
    clauseExpressesRenderedUiAuthoring(clause, subjectPattern, nonUiPattern),
  )
}

/**
 * A project can eventually contain a UI while the current turn is explicitly
 * limited to its first document/planning phase. Downstream implementation and
 * runtime validation are not current-turn completion requirements in that
 * case. The continuation guard keeps combined requests such as "first write
 * the design, then implement it" on the full implementation path.
 */
export function inputLimitsCurrentTurnToDocumentArtifact(input: string): boolean {
  const stagedFirst = STAGED_DOCUMENT_FIRST_PATTERN.exec(input)
  const currentPhaseOnly = CURRENT_PHASE_DOCUMENT_ONLY_PATTERN.exec(input)
  const staged = stagedFirst ?? currentPhaseOnly
  // A short follow-up such as "also write TRD.md" has no reason to repeat the
  // earlier phase-boundary wording: the single named document mutation is the
  // whole current request. Treat that structural shape like an explicitly
  // staged document phase. The predicate below remains conservative and
  // rejects repository analysis, implementation, validation, and UI work.
  if (!staged || staged.index === undefined) {
    return inputLooksLikeSimpleWorkspaceFileOperation(input)
  }
  // A phase boundary must authorize a document output, not merely place a
  // sequencing word near a document reference. For example, "code is the
  // priority; do not repeatedly reread the spec document" can otherwise look
  // like "first ... document" in languages where the sequencing word also
  // means "priority". Requiring the shared durable-document authoring signal
  // keeps referenced requirements as inputs to implementation.
  if (stagedFirst && !currentPhaseOnly && !inputHasDocumentPhaseClause(input)) return false
  const trailing = input.slice(staged.index + staged[0].length)
  // A negative downstream boundary ("write DESIGN.md, but do not create
  // source files") reinforces the staged phase. Do not let the mere presence
  // of an implementation noun after "then/작성하고" invert that boundary.
  // Remove only the bounded negative implementation clause; a later positive
  // "then implement it" remains visible and keeps the full implementation
  // path.
  const positiveTrailing = trailing.replace(NEGATED_DOWNSTREAM_IMPLEMENTATION_PATTERN, ' ')
  return !SAME_TURN_IMPLEMENTATION_AFTER_DOCUMENT_PATTERN.test(positiveTrailing)
}

/**
 * A staged design/spec for a future implementation is authored material, not
 * an as-is repository audit. Requiring an evidence map and acceptance ledger
 * for it makes small planning turns balloon into audit reports. Current-state
 * architecture/repository documents retain the heavier evidence contract.
 */
export function inputRequestsProspectiveDocumentPhase(input: string): boolean {
  return inputLimitsCurrentTurnToDocumentArtifact(input)
    && !CURRENT_IMPLEMENTATION_DOCUMENT_PATTERN.test(input)
}

/**
 * Persisted/semantic run contracts retain explicit user phase boundaries even
 * when the graph's working message window no longer contains the original
 * user message. Capability guards should consult this structured boundary in
 * addition to re-parsing transient chat messages.
 */
export function contractLimitsCurrentTurnToDocumentArtifact(
  contract: Pick<AgentRunContract, 'constraints' | 'outOfScope'> | undefined | null,
): boolean {
  if (!contract) return false
  return contract.constraints.some((entry) =>
    entry.startsWith(CURRENT_DOCUMENT_PHASE_CONSTRAINT_PREFIX)
  ) || contract.outOfScope.some((entry) =>
    entry.startsWith(DEFERRED_IMPLEMENTATION_OUT_OF_SCOPE_PREFIX)
  )
}

function inputExplicitlyUsesCommandLineInterfaceOnly(input: string): boolean {
  // A negative capability boundary such as "no terminal commands" is not a
  // request for a CLI/TUI artifact. Remove that bounded prohibition before
  // deciding that command-line is the positive target surface.
  const positiveInterfaceScope = inputPositiveCapabilityScope(input)
    .replace(EXPLICIT_NO_TERMINAL_PATTERN, ' ')
  return COMMON_INTERACTIVE_UI_TOOL_COMMAND_LINE_PATTERN.test(positiveInterfaceScope)
    && POSITIVE_COMMAND_LINE_INTERFACE_PATTERN.test(positiveInterfaceScope)
    && !EXPLICIT_BROWSER_RENDERED_SURFACE_PATTERN.test(positiveInterfaceScope)
}

function inputRequiresNamedBrowserGameUi(input: string): boolean {
  if (!inputExpressesRenderedUiAuthoringIntent(input)) return false
  if (!NAMED_BROWSER_GAME_PATTERN.test(input)) return false
  if (NAMED_BROWSER_GAME_NON_UI_PATTERN.test(input)) return false
  if (NAMED_BROWSER_GAME_UI_HINT_PATTERN.test(input)) return true
  const normalized = input
    .trim()
    .replace(/[.!?。！？]+$/u, '')
    .replace(/\s+/g, ' ')
  return /^(?:build|make|create|implement|develop|code|ship)\s+(?:a\s+|an\s+|the\s+)?(?:2048|wordle|tetris|pong|breakout|minesweeper|tic[-\s]?tac[-\s]?toe|sudoku|flappy\s+bird|space\s+invaders?|memory\s+(?:match|cards?))$/iu.test(normalized)
    || /^(?:만들어\s*줘|구현해\s*줘|개발해\s*줘|코딩해\s*줘)\s*(?:테트리스|워들|퐁|벽돌\s*깨기|지뢰\s*찾기|스도쿠|틱택토|플래피\s*버드|스페이스\s*인베이더|스네이크\s*게임)$/iu.test(normalized)
}

function inputRequiresCommonInteractiveUiTool(input: string): boolean {
  return splitArtifactClauses(inputPositiveActionScope(input)).some((clause) => {
    if (!clauseExpressesRenderedUiAuthoring(clause, COMMON_INTERACTIVE_UI_TOOL_PATTERN)) {
      return false
    }
    if (!COMMON_INTERACTIVE_UI_TOOL_NON_UI_PATTERN.test(clause)) return true
    return COMMON_INTERACTIVE_UI_TOOL_EXPLICIT_SURFACE_PATTERN.test(clause)
      && !COMMON_INTERACTIVE_UI_TOOL_COMMAND_LINE_PATTERN.test(clause)
  })
}

function inputRequiresCommonInteractiveUiControl(input: string): boolean {
  return splitArtifactClauses(inputPositiveActionScope(input)).some((clause) => {
    if (!clauseExpressesRenderedUiAuthoring(
      clause,
      COMMON_INTERACTIVE_UI_CONTROL_PATTERN,
      COMMON_INTERACTIVE_UI_TOOL_NON_UI_PATTERN,
    )) return false
    return RENDERED_UI_REQUEST_PATTERN.test(clause)
      || COMMON_INTERACTIVE_UI_CONTROL_SURFACE_PATTERN.test(clause)
  })
}

function inputRequiresCommonRenderedUiSurface(input: string): boolean {
  return inputHasRenderedUiAuthoringClause(
    input,
    COMMON_RENDERED_UI_SURFACE_PATTERN,
    COMMON_RENDERED_UI_SURFACE_NON_UI_PATTERN,
  )
}

function inputRequiresCommonRenderedUiSurfaceInteractionSmoke(input: string): boolean {
  return inputRequiresCommonRenderedUiSurface(input)
    && COMMON_RENDERED_UI_SURFACE_INTERACTIVE_PATTERN.test(input)
}

function inputRequiresVisualUiRefinement(input: string): boolean {
  return VISUAL_UI_REFINEMENT_ACTION_PATTERN.test(input)
    && VISUAL_UI_REFINEMENT_SUBJECT_PATTERN.test(input)
    && !VISUAL_UI_REFINEMENT_NON_UI_PATTERN.test(input)
}

function inputRequiresVisualUiFeedbackFollowup(input: string): boolean {
  return (
    VISUAL_UI_FEEDBACK_FOLLOWUP_PATTERN.test(input)
    || VISUAL_UI_VIEWPORT_FEEDBACK_PATTERN.test(input)
  )
    && !VISUAL_UI_FEEDBACK_FOLLOWUP_NON_UI_PATTERN.test(input)
}

function inputLooksLikeDocumentTextMutation(input: string): boolean {
  const paths = Array.from(input.matchAll(fallbackArtifactPathMatcher()))
    .map((match) => normalizeArtifactPath(match[1] ?? ''))
    .filter((path) => Boolean(path) && hasFallbackDurableArtifactExtension(path))
  if (paths.length === 0 || paths.some((path) => !isDocumentArtifactPath(path))) {
    return false
  }
  return /\b(?:append|prepend|replace|preserve|insert)\b|(?:추가|삽입|교체|바꿔|보존|유지)/iu.test(input)
}

export function inputLooksLikeSimpleWorkspaceFileOperation(input: string): boolean {
  const paths = [...new Set(
    Array.from(input.matchAll(fallbackArtifactPathMatcher()))
      .map((match) => normalizeArtifactPath(match[1] ?? '').toLowerCase())
      .filter((path) => Boolean(path) && hasFallbackDurableArtifactExtension(path)),
  )]
  if (paths.length !== 1 || !isDocumentArtifactPath(paths[0]!)) return false
  // A mentioned document can be an input ("read README.md and implement the
  // service") or a negated target. Require the artifact extractor to classify
  // the same path as an intended durable output before granting the lightweight
  // mutation path. Literal append/replace operations retain their established
  // structural signal because words such as "append"/"추가" are deliberately
  // narrower than the broad artifact-authoring lexicon.
  const outputArtifacts = extractPathLikeArtifacts(input)
  const explicitTextMutation = inputLooksLikeDocumentTextMutation(input)
  if (
    !(
      outputArtifacts.length === 1
      && outputArtifacts[0]!.path.toLowerCase() === paths[0]
      && inputExpressesMutationIntent(input)
    )
    && !explicitTextMutation
  ) return false

  // Mask the path before looking for additional work so filenames such as
  // TEST.md or BUILD.md do not look like test/build actions.
  const actionScope = input.replace(fallbackArtifactPathMatcher(), ' ')
  if (
    /\b(?:analy[sz]e|audit|build|code|develop|implement|investigate|refactor|research|test)\b|(?:분석|감사|구현|개발|코딩|리팩터|조사|연구|테스트)/iu.test(actionScope)
  ) {
    return false
  }
  return !inputRequiresNamedBrowserGameUi(input)
    && !inputRequiresCommonInteractiveUiTool(input)
    && !inputRequiresCommonInteractiveUiControl(input)
    && !inputRequiresCommonRenderedUiSurface(input)
    && !inputRequiresVisualUiRefinement(input)
    && !inputRequiresVisualUiFeedbackFollowup(input)
}

export function inputExplicitlyRequestsRenderedUiValidation(input: string): boolean {
  if (
    inputLimitsCurrentTurnToDocumentArtifact(input)
    || inputExplicitlyUsesCommandLineInterfaceOnly(input)
    || !EXPLICIT_RENDERED_UI_VALIDATION_ACTION_PATTERN.test(input)
  ) {
    return false
  }
  if (
    inputRequestsNonRenderedSurfaceInventory(input)
    && !EXPLICIT_RENDERED_UI_VALIDATION_EVIDENCE_PATTERN.test(input)
  ) return false
  if (EXPLICIT_RENDERED_UI_VALIDATION_EVIDENCE_PATTERN.test(input)) return true
  return EXPLICIT_BROWSER_RENDERED_SURFACE_PATTERN.test(input)
    && EXPLICIT_RENDERED_UI_VALIDATION_TARGET_PATTERN.test(input)
}

/**
 * App/site/workspace/page catalogs are structured record inventories, not a
 * local directory listing or rendered-page validation request. Coordinated
 * targets such as "workspace and page list" stay in the same data surface.
 */
export function inputRequestsNonRenderedSurfaceInventory(input: string): boolean {
  return NON_RENDERED_SURFACE_INVENTORY_PATTERN.test(input)
}

function inputExplicitlyRequestsRenderedUiMutation(input: string): boolean {
  const positiveScope = stripExplicitlyNegatedMutationActions(input)
    .replace(BUILD_VALIDATION_ACTION_PATTERN, ' ')
    .replace(TEST_VALIDATION_ACTION_PATTERN, ' ')
  return EXPLICIT_RENDERED_UI_MUTATION_ACTION_PATTERN.test(positiveScope)
}

function inputRequiresRenderedUiAuthoring(input: string): boolean {
  // The surface pattern alone matches any mention of a web surface, including
  // the bare words "web"/"웹" — so "web.search 로 검색해줘" or "이 사이트
  // 요약해줘" collected the four UI authoring criteria, which then require
  // browser screenshot evidence the request never had anything to do with.
  // Rendered-UI validation belongs to requests that actually author UI, which
  // is what every other matcher below already checks for.
  if (
    inputLimitsCurrentTurnToDocumentArtifact(input)
    || inputExplicitlyUsesCommandLineInterfaceOnly(input)
  ) {
    return false
  }
  const specificUiRequest = inputRequiresNamedBrowserGameUi(input)
    || inputRequiresCommonInteractiveUiTool(input)
    || inputRequiresCommonInteractiveUiControl(input)
    || inputRequiresCommonRenderedUiSurface(input)
    || inputRequiresVisualUiRefinement(input)
    || inputRequiresVisualUiFeedbackFollowup(input)
  if (specificUiRequest) return true

  // Literal text being inserted into a document may itself contain words such
  // as "Desktop UI E2E". That payload does not turn a Markdown append into UI
  // authoring and must not attach browser/mobile screenshot requirements.
  if (inputLooksLikeDocumentTextMutation(input)) return false

  return inputHasRenderedUiAuthoringClause(input, RENDERED_UI_REQUEST_PATTERN)
}

export function inputRequiresRenderedUiValidation(input: string): boolean {
  return inputRequiresRenderedUiAuthoring(input)
    || inputExplicitlyRequestsRenderedUiValidation(input)
}

export function inputRequiresRenderedUiInteractionSmoke(input: string): boolean {
  if (
    inputLimitsCurrentTurnToDocumentArtifact(input)
    || inputExplicitlyUsesCommandLineInterfaceOnly(input)
  ) {
    return false
  }
  // Validation of an existing rendered surface inherits only the interaction
  // work positively requested in this turn. A game/dashboard noun describes
  // the target, not permission to click it or synthesize dynamic-state QA.
  if (
    inputExplicitlyRequestsRenderedUiValidation(input)
    && !inputExplicitlyRequestsRenderedUiMutation(input)
  ) {
    return EXPLICIT_RENDERED_UI_INTERACTION_REQUEST_PATTERN.test(
      inputPositiveCapabilityScope(input),
    )
  }
  return INTERACTIVE_RENDERED_UI_REQUEST_PATTERN.test(input)
    || inputRequiresNamedBrowserGameUi(input)
    || inputRequiresCommonInteractiveUiTool(input)
    || inputRequiresCommonInteractiveUiControl(input)
    || inputRequiresCommonRenderedUiSurfaceInteractionSmoke(input)
}

export function inputLooksLikeContextualFollowup(input: string): boolean {
  const positiveActionInput = stripExplicitlyNegatedMutationActions(input)
  return CONTEXTUAL_FOLLOWUP_REQUEST_PATTERN.test(positiveActionInput)
    || CONTEXTUAL_REMAINING_WORK_PATTERN.test(positiveActionInput)
    || inputLooksLikeContextualReviewFollowup(input)
    || CONTEXTUAL_EXECUTION_MODE_FOLLOWUP_PATTERN.test(input)
    || (
      inputExplicitlyForbidsFileMutation(input)
      && inputExpressesMutationIntent(input)
      && CONTEXTUAL_SCOPED_REPAIR_PATTERN.test(positiveActionInput)
    )
}

export function inputLooksLikeContextualReviewFollowup(input: string): boolean {
  return CONTEXTUAL_REVIEW_FOLLOWUP_PATTERN.test(input)
    || CONTEXTUAL_AUDIT_FOLLOWUP_PATTERN.test(input)
}

/**
 * Only an action follow-up should inherit an earlier turn's durable contract.
 * Review/audit/conclusion questions must be free to answer from the transcript
 * without re-entering artifact write and validation loops.
 */
export function inputContinuesActiveRunContract(input: string): boolean {
  return !isFocusedRepositoryChangeLookup(input)
    && !isSubstantiveRepositoryChangeReview(input)
    && !inputRequestsFocusedSingleBrowserObservation(input)
    && inputLooksLikeContextualFollowup(input)
    && !inputLooksLikeContextualReviewFollowup(input)
}

function normalizeText(value: string): string {
  return value.replace(/\s+/g, ' ').trim()
}

function truncateText(value: string, max: number): string {
  const normalized = normalizeText(value)
  return normalized.length <= max
    ? normalized
    : `${normalized.slice(0, max - 3).trimEnd()}...`
}

function normalizeNoTouchTarget(value: string): string {
  const trimmed = value
    .trim()
    .replace(/^[`"']|[`"']$/g, '')
    .replace(/[.,;:]+$/g, '')
    .replace(/(?:은|는|을|를)$/u, '')
    .trim()
  return truncateText(trimmed, 120)
}

function normalizeRequestNamedTargets(
  value: unknown,
  request: string,
): string[] {
  if (!Array.isArray(value)) return []
  const requestText = request.replace(/\\/g, '/').toLowerCase()
  const targets: string[] = []
  for (const entry of value) {
    if (typeof entry !== 'string' || /[\0\r\n]/u.test(entry)) continue
    const target = normalizeNoTouchTarget(entry).replace(/\\/g, '/')
    if (
      !target
      || target.length > 160
      || !requestText.includes(target.toLowerCase())
      || targets.some((existing) => existing.toLowerCase() === target.toLowerCase())
    ) continue
    targets.push(target)
    if (targets.length >= 32) break
  }
  return targets
}

/**
 * Validate an LLM-owned target-level write decision without re-classifying
 * the request in runtime code. Targets must be bounded, named verbatim by the
 * user, and paired with an execution posture that actually permits writes.
 * The LLM decides positive versus protected semantics; this function only
 * prevents malformed or invented paths from becoming write authority.
 */
export function normalizeAuthorizedWriteTargets(
  value: unknown,
  request: string,
  workspaceMutation: unknown,
  capabilities: unknown,
): string[] {
  if (
    workspaceMutation === 'forbidden'
    || !Array.isArray(capabilities)
    || !capabilities.includes('filesystem-write')
  ) return []
  return normalizeRequestNamedTargets(value, request)
}

/** Validate, but do not semantically infer, LLM-selected no-write targets. */
export function normalizeProtectedWriteTargets(
  value: unknown,
  request: string,
): string[] {
  return normalizeRequestNamedTargets(value, request)
}

function isSpecificNoTouchTarget(value: string): boolean {
  const normalized = value.trim().toLowerCase()
  if (!normalized || AMBIGUOUS_NO_TOUCH_TARGETS.has(normalized)) {
    return false
  }
  return /[A-Za-z0-9_@./~-]|[\uAC00-\uD7AF]/u.test(value)
}

function isKoreanNoTouchActionSeparator(
  input: string,
  match: RegExpMatchArray,
  rawTarget: string,
): boolean {
  const trimmed = rawTarget.trim()
  if (!/^[./~\\]+$/u.test(trimmed) || /^[`"'].*[`"']$/u.test(trimmed)) {
    return false
  }

  // In Korean, coordinated mutation verbs commonly share a slash, as in
  // "생성/수정하지 마라". The target-first no-touch grammar must not turn
  // that separator into protection for the filesystem root. A punctuation-
  // only path remains explicit when it is named as a path (or quoted above).
  const prefix = input.slice(Math.max(0, (match.index ?? 0) - 32), match.index ?? 0)
  return !/(?:root|home|루트|홈|현재\s*(?:작업\s*)?디렉터리|상위\s*디렉터리)\s*$/iu.test(prefix)
}

function extractExplicitNoTouchTargets(input: string): string[] {
  const targets: string[] = []
  for (const pattern of [EXPLICIT_NO_TOUCH_ENGLISH_PATTERN, EXPLICIT_NO_TOUCH_KOREAN_PATTERN]) {
    pattern.lastIndex = 0
    for (const match of input.matchAll(pattern)) {
      const rawTarget = match.groups?.target ?? ''
      if (
        pattern === EXPLICIT_NO_TOUCH_KOREAN_PATTERN
        && isKoreanNoTouchActionSeparator(input, match, rawTarget)
      ) {
        continue
      }
      const target = normalizeNoTouchTarget(rawTarget)
      if (!isSpecificNoTouchTarget(target)) {
        continue
      }
      if (!targets.some((existing) => existing.toLowerCase() === target.toLowerCase())) {
        targets.push(target)
      }
      if (targets.length >= MAX_EXPLICIT_NO_TOUCH_TARGETS) {
        return targets
      }
    }
  }
  // Coordinated path lists often share one trailing negative action, e.g.
  // "do not update A.md, B.md, or C.md" or
  // "A.md, B.md, C.md를 만들거나 확장하지 말고". Evaluate every concrete
  // path against the same bounded negative-clause parser used by artifact
  // extraction so all named boundaries survive planner normalization.
  const pathPattern = fallbackArtifactPathMatcher()
  let pathMatch: RegExpExecArray | null
  while ((pathMatch = pathPattern.exec(input)) !== null) {
    const target = normalizeNoTouchTarget(pathMatch[1] ?? '')
    if (
      !isSpecificNoTouchTarget(target)
      || !isNegatedArtifactOutputContext(input, pathMatch.index, pathPattern.lastIndex)
    ) {
      continue
    }
    if (!targets.some((existing) => existing.toLowerCase() === target.toLowerCase())) {
      targets.push(target)
    }
    if (targets.length >= MAX_EXPLICIT_NO_TOUCH_TARGETS) {
      break
    }
  }
  return targets
}

function noTouchConstraintForTarget(target: string): string {
  return `${NO_TOUCH_CONSTRAINT_PREFIX}"${target}"; keep direct file edits scoped to the requested agent/runtime implementation unless the user explicitly changes this boundary.`
}

function noTouchOutOfScopeForTarget(target: string): string {
  return `${NO_TOUCH_OUT_OF_SCOPE_PREFIX}"${target}" outside the requested agent/workflow.`
}

function normalizeArtifactPath(value: string): string {
  return value
    .trim()
    .replace(/[.,;:]+$/g, '')
    .replace(/\\/g, '/')
    .slice(0, MAX_ARTIFACT_PATH_CHARS)
}

function artifactKind(value: unknown, path: string): AgentRequiredArtifact['kind'] {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : ''
  if (normalized === 'file' || normalized === 'directory' || normalized === 'document' || normalized === 'other') {
    return normalized
  }
  if (isDocumentArtifactPath(path)) {
    return 'document'
  }
  return path.endsWith('/') ? 'directory' : 'file'
}

function artifactExtension(path: string): string | null {
  const fileName = path.split('/').pop() ?? path
  const match = /\.([A-Za-z0-9_-]{1,16})$/.exec(fileName)
  return match ? match[1]!.toLowerCase() : null
}

function hasFallbackDurableArtifactExtension(path: string): boolean {
  const extension = artifactExtension(path)
  return Boolean(extension && FALLBACK_DURABLE_ARTIFACT_EXTENSIONS.has(extension))
}

export function isDocumentArtifactPath(path: string): boolean {
  const extension = artifactExtension(path)
  return Boolean(extension && DOCUMENT_ARTIFACT_EXTENSIONS.has(extension))
}

/**
 * Whether a run contract describes genuine durable *document* artifact work —
 * a report/analysis/design deliverable that warrants the heavy evidence-map /
 * self-review / artifact-recovery machinery. A bare `requiredArtifacts` entry
 * that merely names an existing *source* file to edit (e.g. a weak planner
 * labelling `foo.py` as a "required artifact" on a plain bug-fix) is NOT
 * document work — forcing the document-recovery loop on it makes a code run
 * spin on a file that may not even be the right one to edit. Document work is
 * signalled by artifact sections, a document-kind artifact, or a
 * document-extension path; plain source edits are handled by the coder graph's
 * own edit guards and validation instead.
 */
export function contractHasDocumentArtifactWork(
  contract:
    | {
        requiredArtifacts?: AgentRequiredArtifact[]
        artifactSections?: { id: string; artifactPath?: string }[]
      }
    | undefined
    | null,
): boolean {
  if (!contract) return false
  // A genuine durable document deliverable: a document-kind/extension artifact,
  // or artifact sections that target a document path. A planner can hallucinate
  // report-style `artifactSections` ("Source Inventory", "Verification
  // Results") on a plain code fix — those carry no document path (or point at a
  // source file), so they do NOT count as document work and must not engage the
  // document evidence-map / self-review / artifact-recovery machinery.
  const hasDocumentArtifact = (contract.requiredArtifacts ?? []).some(
    (artifact) => artifact.kind === 'document' || isDocumentArtifactPath(artifact.path),
  )
  if (hasDocumentArtifact) return true
  return (contract.artifactSections ?? []).some(
    (section) => Boolean(section.artifactPath) && isDocumentArtifactPath(section.artifactPath!),
  )
}

export function contractRequiresRenderedUiValidation(
  contract:
    | {
        acceptanceCriteria?: Array<{ text?: string } | string>
        evidenceRequirements?: Array<{ description?: string } | string>
      }
    | undefined
    | null,
): boolean {
  if (!contract) return false
  // Run contracts can come from the deterministic fallback builder, an LLM
  // planner, a cowork specialist scope, or an older persisted checkpoint.
  // They therefore do not share one exact sentence or capitalization.  This
  // predicate is a capability check, not a prose-template check: recognize
  // the stable combination of a rendered/browser surface and concrete visual
  // evidence (screenshots, viewport rendering, or visual/layout QA).
  const renderedUiEvidencePattern =
    /(?:\brendered\s+ui\b[^\n]{0,180}\b(?:browser|screenshots?|viewports?|visual|layout)\b|\bbrowser\b[^\n]{0,180}\b(?:desktop|mobile|viewports?|rendered\s+(?:ui|pages?|screens?|apps?|applications?|interfaces?|frontends?)|visual\s+qa|layout\s+audit|capture[^\n]{0,60}screenshots?|screenshots?[^\n]{0,60}(?:attached|evidence))\b|\bscreenshots?\b[^\n]{0,120}\b(?:attached|evidence|desktop|mobile|viewports?|rendered\s+ui)\b)/iu
  return [
    ...(contract.acceptanceCriteria ?? []).map((criterion) =>
      typeof criterion === 'string' ? criterion : criterion.text
    ),
    ...(contract.evidenceRequirements ?? []).map((requirement) =>
      typeof requirement === 'string' ? requirement : requirement.description
    ),
  ]
    .filter((text): text is string => typeof text === 'string')
    .some((text) => renderedUiEvidencePattern.test(text))
}

function fallbackArtifactPathMatcher(): RegExp {
  return new RegExp(FALLBACK_ARTIFACT_PATH_PATTERN.source, FALLBACK_ARTIFACT_PATH_PATTERN.flags)
}

function containsFallbackArtifactPath(value: string): boolean {
  const pattern = fallbackArtifactPathMatcher()
  let match: RegExpExecArray | null
  while ((match = pattern.exec(value)) !== null) {
    if (hasFallbackDurableArtifactExtension(normalizeArtifactPath(match[1] ?? ''))) {
      return true
    }
  }
  return false
}

function containsSentenceBoundary(value: string): boolean {
  return /[.!?。！？]\s*/u.test(value)
}

function hasFallbackArtifactIntent(input: string, start: number, end: number): boolean {
  const beforeWindow = input.slice(Math.max(0, start - 120), start)
  const afterWindow = input.slice(end, Math.min(input.length, end + 120))
  const beforeMatches = Array.from(beforeWindow.matchAll(FALLBACK_ARTIFACT_INTENT_PATTERN))
  const latestBeforeIntent = beforeMatches.at(-1)
  if (latestBeforeIntent?.index != null) {
    const betweenIntentAndPath = beforeWindow.slice(
      latestBeforeIntent.index + latestBeforeIntent[0].length,
    )
    if (
      !containsSentenceBoundary(betweenIntentAndPath)
      && !containsFallbackArtifactPath(betweenIntentAndPath)
    ) {
      return true
    }
  }

  const afterIntent = Array.from(afterWindow.matchAll(FALLBACK_ARTIFACT_INTENT_PATTERN))[0]
  if (afterIntent?.index != null) {
    const betweenPathAndIntent = afterWindow.slice(0, afterIntent.index)
    if (
      !containsSentenceBoundary(betweenPathAndIntent)
      && !containsFallbackArtifactPath(betweenPathAndIntent)
    ) {
      return true
    }
  }
  return false
}

/**
 * A document path can be an input to a coding task rather than an output.
 * Phrases such as "implement the requirements in README.md" and
 * "README.md의 요구사항을 구현해" must not turn that reference document into
 * a required artifact: doing so authorizes an unrelated README rewrite and
 * engages the heavyweight document-recovery loop.
 */
function isReferencedInputArtifactContext(input: string, start: number, end: number): boolean {
  const before = input.slice(Math.max(0, start - 120), start)
  const after = input.slice(end, Math.min(input.length, end + 120))

  const referenceBeforePath = /(?:\b(?:read|consult|follow|use)\s+|\b(?:requirements?|instructions?|spec(?:ification)?s?|guidance)\s+(?:in|from|of)\s*)$/iu
  const referenceAfterPath = /^\s*(?:(?:의|에\s*(?:있는|명시된)|에서)\s*)?(?:요구\s*사항|지침|설명|명세|스펙|requirements?|instructions?|spec(?:ification)?s?|guidance)|^\s*(?:을|를|에)?\s*(?:읽|참고|따라|맞춰|기준으로)/iu
  const declaredInputAfterPath = /^\s*(?:(?:은|는|이|가)\s*)?(?:[^.!?。！？\n]{0,48}\s)?(?:입력|참조|참고)(?:용|\s*)?(?:사양|문서|자료|명세|스펙)|^\s*(?:is|are)\s+(?:an?\s+|the\s+)?(?:input|reference|source)\b/iu
  const coordinatedReferenceAfterPath = new RegExp(
    '^\\s*(?:(?:와|과|및|,|and|&)\\s*'
      + '(?:\\.{1,2}/|/|~/)?[A-Za-z0-9_.@-]+(?:/[A-Za-z0-9_.@-]+)*\\.[A-Za-z0-9][A-Za-z0-9_-]{0,15}'
      + ')*\\s*(?:(?:을|를|에)\\s*)?(?:읽|참고|따라|맞춰|기준으로|as\\s+(?:input|requirements?|spec(?:ification)?s?)|to\\s+(?:implement|build|develop|code))',
    'iu',
  )

  return referenceBeforePath.test(before)
    || referenceAfterPath.test(after)
    || declaredInputAfterPath.test(after)
    || coordinatedReferenceAfterPath.test(after)
}

function isArtifactClauseBoundary(input: string, index: number): boolean {
  const char = input[index]
  if (!char || !'.!?。！？\n'.includes(char)) return false
  if (char !== '.') return true
  // A literal period inside a file/path token (README.md, foo.bar/baz) is not
  // a sentence boundary. Treating it as one truncated coordinated negative
  // clauses at every filename, so only the final path inherited "do not edit".
  const previous = input[index - 1] ?? ''
  const next = input[index + 1] ?? ''
  return !(/[A-Za-z0-9_@~/-]/u.test(previous) && /[A-Za-z0-9_@~/-]/u.test(next))
}

function previousArtifactClauseBoundary(input: string, before: number): number {
  for (let index = Math.min(before, input.length - 1); index >= 0; index -= 1) {
    if (isArtifactClauseBoundary(input, index)) return index
  }
  return -1
}

function nextArtifactClauseBoundary(input: string, after: number): number {
  for (let index = Math.max(0, after); index < input.length; index += 1) {
    if (isArtifactClauseBoundary(input, index)) return index
  }
  return input.length
}

function splitArtifactClauses(input: string): string[] {
  const clauses: string[] = []
  let start = 0
  for (let index = 0; index < input.length; index += 1) {
    if (!isArtifactClauseBoundary(input, index)) continue
    const clause = input.slice(start, index).trim()
    if (clause) clauses.push(clause)
    start = index + 1
  }
  const tail = input.slice(start).trim()
  if (tail) clauses.push(tail)
  return clauses
}

/** A named file in a bounded negative clause is an exclusion, not an output. */
function isNegatedArtifactOutputContext(input: string, start: number, end: number): boolean {
  const clauseStart = previousArtifactClauseBoundary(input, start - 1) + 1
  const clauseEnd = nextArtifactClauseBoundary(input, end)
  const before = input.slice(clauseStart, start)
  const after = input.slice(end, clauseEnd)

  const englishNegativeBefore = /\b(?:do\s+not|don't|dont|never|without)\b[^.!?;\n]{0,120}\b(?:create|write|extend|expand|generate|produce|save|update|implement|build|develop|code|modify|edit)\b[^.!?;\n]{0,100}$/iu
  const koreanNegativeAfter = /(?:(?:작성|생성|저장|만들|기록|확장|수정|편집|변경|업데이트|갱신|덮어쓰|건드리|고치|삭제|지우|구현|개발)(?:은|는|을|를)?\s*(?:(?:하)?지\s*(?:마|말(?:아|라)?|마라|않(?:고|도록|는다|습니다)?)|말고|없이)|제외)/iu.exec(after)
  if (englishNegativeBefore.test(before)) return true
  if (!koreanNegativeAfter) return false
  if (koreanNegativeAfter.index === 0) return true

  // Korean commonly places a separate downstream exclusion after a positive
  // artifact action in the same sentence: "DESIGN.md를 작성하고 소스 코드는
  // 만들지 마라". The later source prohibition must not negate the earlier
  // document output. A completed sequencing suffix distinguishes that shape
  // from the direct target prohibition "DESIGN.md를 작성하지 마라".
  const beforeNegative = after.slice(0, koreanNegativeAfter.index)
  const completedPositiveArtifactAction =
    /(?:작성|생성|저장|기록)(?:해|하|했|할)?(?:고|후)|만들(?:고|어서|어\s*둔\s*뒤)/iu.test(beforeNegative)
  return !completedPositiveArtifactAction
}

function dedupeArtifacts(artifacts: AgentRequiredArtifact[]): AgentRequiredArtifact[] {
  const seen = new Set<string>()
  const out: AgentRequiredArtifact[] = []
  for (const artifact of artifacts) {
    const path = normalizeArtifactPath(artifact.path)
    if (
      !path
      || path.includes('\0')
      || path.startsWith('http://')
      || path.startsWith('https://')
      || seen.has(path.toLowerCase())
    ) {
      continue
    }
    seen.add(path.toLowerCase())
    out.push({
      path,
      kind: artifactKind(artifact.kind, path),
      ...(artifact.description ? { description: truncateText(artifact.description, MAX_LIST_ITEM_CHARS) } : {}),
    })
    if (out.length >= MAX_ARTIFACTS) {
      break
    }
  }
  return out
}

function extractPathLikeArtifacts(input: string): AgentRequiredArtifact[] {
  const artifacts: AgentRequiredArtifact[] = []
  let match: RegExpExecArray | null
  const pattern = fallbackArtifactPathMatcher()
  while ((match = pattern.exec(input)) !== null) {
    const path = normalizeArtifactPath(match[1] ?? '')
    if (
      path
      && !path.startsWith('http://')
      && !path.startsWith('https://')
      && !path.includes('*')
      && hasFallbackDurableArtifactExtension(path)
      && hasFallbackArtifactIntent(input, match.index, pattern.lastIndex)
      && !isReferencedInputArtifactContext(input, match.index, pattern.lastIndex)
      && !isNegatedArtifactOutputContext(input, match.index, pattern.lastIndex)
    ) {
      artifacts.push({ path, kind: artifactKind(undefined, path) })
    }
  }
  return dedupeArtifacts(artifacts)
}

function documentArtifacts(artifacts: AgentRequiredArtifact[]): AgentRequiredArtifact[] {
  return artifacts.filter((artifact) =>
    artifact.kind === 'document' || isDocumentArtifactPath(artifact.path)
  )
}

function fallbackEvidenceRequirementsForArtifacts(
  artifacts: AgentRequiredArtifact[],
): AgentEvidenceRequirement[] {
  if (documentArtifacts(artifacts).length === 0) {
    return []
  }
  return [{
    kind: 'artifact',
    description: 'Document artifacts must expose the evidence or coverage basis for their main claims and include an acceptance-criteria self-review before completion.',
    requiresArtifactEvidenceMap: true,
    requiresArtifactSelfReview: true,
  }]
}

function fallbackArtifactSectionsForArtifacts(
  artifacts: AgentRequiredArtifact[],
): AgentArtifactSection[] {
  const artifact = documentArtifacts(artifacts)[0]
  if (!artifact) {
    return []
  }
  return [
    {
      id: 'scope-and-requirements',
      title: 'Scope And Requirements',
      artifactPath: artifact.path,
      description: 'Restate the requested scope, expected deliverable, and any explicit exclusions or limits.',
      required: true,
    },
    {
      id: 'evidence-map',
      title: 'Evidence Map',
      artifactPath: artifact.path,
      description: 'Map important claims or coverage areas to observed tool results, source paths, references, or validation output.',
      required: true,
    },
    {
      id: 'core-content',
      title: 'Core Content',
      artifactPath: artifact.path,
      description: 'Contain the primary requested analysis, report, design, research result, or authored material.',
      required: true,
    },
    {
      id: 'validation-risks-and-gaps',
      title: 'Validation, Risks, And Gaps',
      artifactPath: artifact.path,
      description: 'Summarize validation performed, unsupported assumptions, risks, and any remaining incomplete scope.',
      required: true,
    },
    {
      id: 'acceptance-criteria-review',
      title: 'Acceptance Criteria Review',
      artifactPath: artifact.path,
      description: 'Close each run contract acceptance criterion by id with satisfied, partial, or blocked status.',
      required: true,
    },
  ]
}

type RenderedUiContractShape = {
  acceptanceCriteria: Array<{ id: string; text: string }>
  evidenceRequirements?: AgentEvidenceRequirement[]
  executionIntent?: AgentExecutionIntent
}

function reindexAcceptanceCriteria<T extends { id: string; text: string }>(
  criteria: T[],
): T[] {
  return criteria.map((criterion, index) => ({
    ...criterion,
    id: `AC${index + 1}`,
  }))
}

function hasRenderedUiScreenshotCriterion(criteria: Array<{ text: string }>): boolean {
  return criteria.some((criterion) =>
    /(?:Screenshot image attachment:\s*attached|screenshots?[^\n]{0,160}(?:attached|viewports?|layout[- ]audit)|layout[- ]audit results? at [^.;\n]+viewport|\bbrowser\.screenshot\b[^\n]{0,220}(?:\d{3,5}\s*[x×]\s*\d{3,5}|viewport|save[ds]?|path|\/[^\s]+\.(?:png|jpe?g|webp)))/iu
      .test(criterion.text)
  )
}

function hasRenderedUiDesignPlanCriterion(criteria: Array<{ text: string }>): boolean {
  return criteria.some((criterion) =>
    /(?:design\s+(?:plan|brief)|target\s+user|workflow|primary\s+screens?|responsive\s+layout|visual\s+style|usable\s+(?:product|tool|game)|디자인\s*(?:계획|브리프)|사용자\s*흐름|주요\s*화면|반응형\s*레이아웃)/iu
      .test(criterion.text)
  )
}

function hasRenderedUiSmokeCriterion(criteria: Array<{ text: string }>): boolean {
  return criteria.some((criterion) =>
    /(?:console|page\s+error|browser\s+smoke|representative\s+interaction|interaction|콘솔|페이지\s*오류|브라우저\s*오류|상호작용)/iu
      .test(criterion.text)
  )
}

function hasRenderedUiVisualQaCriterion(criteria: Array<{ text: string }>): boolean {
  return criteria.some((criterion) =>
    /(?:visual\s+(?:qa|review|inspection)|compare[sd]?\s+.*design\s+plan|fresh\s+screenshot|issues?\s+found|시각\s*(?:검수|리뷰)|재촬영)/iu
      .test(criterion.text)
  )
}

function hasRenderedUiEvidenceRequirement(requirements: AgentEvidenceRequirement[] | undefined): boolean {
  return (requirements ?? []).some((requirement) =>
    /(?:screenshot|desktop|mobile|viewport|browser|rendered\s+ui|스크린샷|브라우저|화면|뷰포트)/iu
      .test(requirement.description)
  )
}

function hasRenderedUiRequestedScopeText(
  texts: string[],
  scope: RenderedUiRequestedScope,
): boolean {
  if (!scope.bounded && !scope.excludesInteraction) return true
  const dimension = /(\d{3,5})×(\d{3,5})/u.exec(scope.viewportLabel)
  return texts.some((text) => {
    if (text.includes(scope.viewportLabel)) return true
    if (dimension) {
      return new RegExp(`${dimension[1]}\\s*[x×]\\s*${dimension[2]}`, 'iu').test(text)
    }
    if (scope.excludesMobile) return /\bdesktop\b|데스크톱|데스크탑/iu.test(text)
    if (scope.excludesDesktop) return /\b(?:mobile|phone|tablet)\b|모바일|휴대폰|태블릿/iu.test(text)
    return false
  })
}

export function addRenderedUiValidationToContract<T extends RenderedUiContractShape>(
  contract: T,
  input: string,
  maxCriteria = 6,
  preserveExistingAuthoringCriteria = false,
): T {
  // A structured execution intent is the semantic authority for capability
  // scope. Lexical UI detection is only a fallback when no such decision is
  // available; it must never widen an LLM-routed non-browser task because a
  // filename or domain term happens to resemble a rendered surface.
  if (
    contract.executionIntent?.capabilityPolicy === 'closed'
    && !contract.executionIntent.capabilities.includes('browser')
  ) {
    return reconcileRunContractExecutionCapabilities(contract)
  }
  if (!inputRequiresRenderedUiValidation(input)) {
    return reconcileRunContractExecutionCapabilities(contract)
  }

  const additionalCriteria: Array<{ id: string; text: string }> = []
  const requestedScope = renderedUiRequestedScope(input)
  const explicitValidationRequest = inputExplicitlyRequestsRenderedUiValidation(input)
  const executionCapabilities = contract.executionIntent?.capabilities ?? []
  const isExclusiveRuntimeOperation = !explicitValidationRequest
    && contract.executionIntent?.kind === 'operational-action'
    && contract.executionIntent.workspaceMutation === 'forbidden'
    && executionCapabilities.length > 0
    && executionCapabilities.every((capability) =>
      capability === 'process'
      || capability === 'service'
      || capability === 'terminal'
    )
  // Runtime commands often contain UI nouns in project or script names
  // ("game dev server", "dashboard preview", and similar). When the current
  // contract is already an exclusive, read-only operational action and the
  // user did not positively request browser evidence, those nouns describe
  // the process being started—not authority to add a browser/design phase.
  if (isExclusiveRuntimeOperation) return contract

  const explicitUiMutation = inputExplicitlyRequestsRenderedUiMutation(input)
    && inputExpressesMutationIntent(input)
  // Surface matchers deliberately recognize games, dashboards, and other UI
  // nouns, but those nouns also appear in bounded audit-only requests. An
  // explicit browser validation request without mutation authority is an
  // operational observation, not a new authoring phase.
  const validationOnlyRequest = explicitValidationRequest && !explicitUiMutation
  const authoringRequest = inputRequiresRenderedUiAuthoring(input)
    && !validationOnlyRequest
  const positiveNonValidationScope = stripExplicitlyNegatedMutationActions(input)
    .replace(BUILD_VALIDATION_ACTION_PATTERN, ' ')
    .replace(TEST_VALIDATION_ACTION_PATTERN, ' ')
  const preImplementationAuthoringRequest = authoringRequest
    && RENDERED_UI_PREIMPLEMENTATION_AUTHORING_PATTERN.test(positiveNonValidationScope)
  const explicitlyUsesDesignBaseline =
    /\b(?:design\s+(?:plan|brief|spec(?:ification)?)|DESIGN\.md)\b|(?:설계\s*(?:계획|문서|명세)|디자인\s*(?:계획|브리프))/iu.test(input)
  // Semantic planners see generic rendered-UI guardrails and can copy their
  // pre-implementation design requirement into a turn that only audits an
  // already-running product. Remove that generated dependency unless the
  // user explicitly named a design baseline; validation must not manufacture
  // an authoring phase before it can open the browser.
  const validationAdjustedCriteria = validationOnlyRequest
    && !explicitlyUsesDesignBaseline
    && !preserveExistingAuthoringCriteria
    ? contract.acceptanceCriteria.filter((criterion) =>
        !/(?:before\s+implementation|design\s+(?:plan|brief)[^\n]{0,220}(?:completed\s+todowrite|target\s+user|primary\s+screens?|responsive\s+layout)|compare[sd]?[^\n]{0,100}against\s+the\s+design\s+plan)/iu
          .test(criterion.text)
      )
    : contract.acceptanceCriteria
  // Planner prose is advisory. Remove any generated criterion that widens an
  // explicit user boundary before injecting the scope-aware invariant below.
  const baseCriteria = validationAdjustedCriteria.filter(
    (criterion) => !renderedUiCriterionExceedsRequestedScope(criterion.text, requestedScope),
  )
  const positiveValidationInput = explicitNegativeClauseScopes(input).positiveText
  const screenshotRequested = authoringRequest
    || EXPLICIT_RENDERED_UI_SCREENSHOT_REQUEST_PATTERN.test(positiveValidationInput)
  const visualQaRequested = authoringRequest
    || EXPLICIT_RENDERED_UI_VISUAL_QA_REQUEST_PATTERN.test(positiveValidationInput)
  const smokeRequested = authoringRequest
    || EXPLICIT_RENDERED_UI_SMOKE_REQUEST_PATTERN.test(positiveValidationInput)
  const interactionRequested = authoringRequest
    || EXPLICIT_RENDERED_UI_INTERACTION_REQUEST_PATTERN.test(positiveValidationInput)
  // A design brief is a pre-implementation requirement. Do not invent it for
  // a later validation-only turn whose requested artifact already exists.
  if (
    preImplementationAuthoringRequest
    && !hasRenderedUiDesignPlanCriterion(baseCriteria)
  ) {
    additionalCriteria.push({ id: '', text: RENDERED_UI_DESIGN_PLAN_CRITERION })
  }
  if (
    screenshotRequested
    && (
      !hasRenderedUiScreenshotCriterion(baseCriteria)
      || !hasRenderedUiRequestedScopeText(baseCriteria.map((criterion) => criterion.text), requestedScope)
    )
  ) {
    additionalCriteria.push({
      id: '',
      text: scopedRenderedUiScreenshotCriterion(
        requestedScope,
        validationOnlyRequest && !visualQaRequested,
      ),
    })
  }
  if (
    visualQaRequested
    && (
      !hasRenderedUiVisualQaCriterion(baseCriteria)
      || !hasRenderedUiRequestedScopeText(baseCriteria.map((criterion) => criterion.text), requestedScope)
    )
  ) {
    additionalCriteria.push({
      id: '',
      text: scopedRenderedUiVisualQaCriterion(
        requestedScope,
        validationOnlyRequest && !explicitlyUsesDesignBaseline,
      ),
    })
  }
  if (
    smokeRequested
    && (
      !hasRenderedUiSmokeCriterion(baseCriteria)
      || !hasRenderedUiRequestedScopeText(baseCriteria.map((criterion) => criterion.text), requestedScope)
    )
  ) {
    additionalCriteria.push({
      id: '',
      text: scopedRenderedUiSmokeCriterion(
        requestedScope,
        validationOnlyRequest && !explicitlyUsesDesignBaseline,
        interactionRequested,
      ),
    })
  }

  const keepCount = Math.max(0, maxCriteria - additionalCriteria.length)
  const acceptanceCriteria = reindexAcceptanceCriteria([
    ...baseCriteria.slice(0, keepCount),
    ...additionalCriteria,
  ])

  const exactBoundedScreenshotCovered = validationOnlyRequest
    && requestedScope.bounded
    && hasRenderedUiScreenshotCriterion(baseCriteria)
  const existingEvidence = (contract.evidenceRequirements ?? []).filter(
    (requirement) => !renderedUiCriterionExceedsRequestedScope(
      requirement.description,
      requestedScope,
    )
    && !(
      exactBoundedScreenshotCovered
      && hasRenderedUiEvidenceRequirement([requirement])
    ),
  )
  const requiresComprehensiveBrowserEvidence = requestedScope.excludesSemanticVisualInspection
    || (
      !exactBoundedScreenshotCovered
      && (
        visualQaRequested
        || smokeRequested
        || authoringRequest
        || screenshotRequested
      )
    )
  const evidenceRequirements = !requiresComprehensiveBrowserEvidence
    || (
      hasRenderedUiEvidenceRequirement(existingEvidence)
      && hasRenderedUiRequestedScopeText(
        existingEvidence.map((requirement) => requirement.description),
        requestedScope,
      )
    )
    ? existingEvidence
    : [
        ...existingEvidence.slice(0, Math.max(0, MAX_EVIDENCE_REQUIREMENTS - 1)),
        {
          kind: 'validation' as const,
          description: scopedRenderedUiEvidenceDescription(
            requestedScope,
            validationOnlyRequest && !explicitlyUsesDesignBaseline,
          ),
        },
      ]

  // The current turn owns its execution posture. A later request to inspect an
  // already-rendered UI is an operational action even when it inherits a
  // workspace-change contract from the turn that built the UI. Keeping the old
  // writer posture sends the follow-up through codebase exploration and the
  // coding planner before it can start a process or open a browser. Conversely,
  // "inspect and fix" still expresses mutation intent and therefore retains
  // the writer posture.
  // Rendered-UI enrichment owns evidence requirements, not semantic mutation
  // authority. Only refine a contract that is already explicitly read-only;
  // when the contract planner/router is unavailable, leave execution intent
  // unspecified instead of inferring workspaceMutation=forbidden from browser
  // validation wording. A composite request may legitimately need source
  // repair before the same requested screenshots can pass, and collapsing it
  // here strands the run on an operational-only graph. Existing explicit
  // read-only contracts remain bounded and gain the capabilities needed to
  // perform their requested browser observation.
  const validationOnlyExecutionIntent = validationOnlyRequest
    && (
      contract.executionIntent?.workspaceMutation === 'forbidden'
      || (
        inputExplicitlyForbidsFileMutation(input)
        && !inputExpressesMutationIntent(input)
      )
    )
    ? {
        kind: 'operational-action' as const,
        workspaceMutation: 'forbidden' as const,
        capabilities: applyExplicitCapabilityBoundaries(
          [
            'process',
            'terminal',
            'browser',
            'filesystem-read',
          ],
          input,
        ),
        ...(contract.executionIntent?.capabilityPolicy === 'closed'
          ? { capabilityPolicy: 'closed' as const }
          : {}),
        ...(contract.executionIntent?.allowedTools
          ? { allowedTools: [...contract.executionIntent.allowedTools] }
          : {}),
        ...(contract.executionIntent?.toolSequence
          ? { toolSequence: [...contract.executionIntent.toolSequence] }
          : {}),
        ...(contract.executionIntent?.retryPolicy
          ? { retryPolicy: contract.executionIntent.retryPolicy }
          : {}),
        ...(contract.executionIntent?.authorizedWriteTargets
          ? { authorizedWriteTargets: [...contract.executionIntent.authorizedWriteTargets] }
          : {}),
        ...(contract.executionIntent?.protectedWriteTargets
          ? { protectedWriteTargets: [...contract.executionIntent.protectedWriteTargets] }
          : {}),
        ...(contract.executionIntent?.requestedTerminalCommand
          ? {
              requestedTerminalCommand: {
                ...contract.executionIntent.requestedTerminalCommand,
                args: [...contract.executionIntent.requestedTerminalCommand.args],
              },
            }
          : {}),
        ...(contract.executionIntent?.requestedProcessStart
          ? {
              requestedProcessStart: structuredClone(
                contract.executionIntent.requestedProcessStart,
              ),
            }
          : {}),
        ...(contract.executionIntent?.constrainedProcessStart
          ? {
              constrainedProcessStart: structuredClone(
                contract.executionIntent.constrainedProcessStart,
              ),
            }
          : {}),
      }
    : undefined
  const executionIntent = validationOnlyExecutionIntent
    ?? (contract.executionIntent
      ? {
          ...contract.executionIntent,
          capabilities: [
            ...new Set<AgentExecutionCapability>([
              ...contract.executionIntent.capabilities,
              'browser',
            ]),
          ],
          ...(contract.executionIntent.allowedTools
            ? { allowedTools: [...contract.executionIntent.allowedTools] }
            : {}),
          ...(contract.executionIntent.toolSequence
            ? { toolSequence: [...contract.executionIntent.toolSequence] }
            : {}),
          ...(contract.executionIntent.authorizedWriteTargets
            ? { authorizedWriteTargets: [...contract.executionIntent.authorizedWriteTargets] }
            : {}),
          ...(contract.executionIntent.protectedWriteTargets
            ? { protectedWriteTargets: [...contract.executionIntent.protectedWriteTargets] }
            : {}),
        }
      : undefined)

  const {
    evidenceRequirements: _previousEvidenceRequirements,
    ...contractWithoutEvidenceRequirements
  } = contract
  return {
    ...contractWithoutEvidenceRequirements,
    acceptanceCriteria,
    ...(evidenceRequirements.length > 0 ? { evidenceRequirements } : {}),
    ...(executionIntent ? { executionIntent } : {}),
  } as T
}

// Distil a short, request-specific gist so the fallback contract has at least
// one acceptance criterion anchored to what the user actually asked for, rather
// than only generic boilerplate that cannot catch a task-specific wrong answer.
function deriveRequestGist(input: string): string {
  const firstLine = input.split(/\r?\n/).map((line) => line.trim()).find(Boolean) ?? input.trim()
  // Preserve all clauses on the first logical request line. The final clause
  // often narrows the current phase (for example, "first write the design
  // document"); retaining only the first sentence turns that boundary into an
  // implementation mandate and expands the run beyond what was requested.
  return truncateText(firstLine.trim(), MAX_REQUEST_GIST_CHARS)
}

export function addContextualFollowupReviewCriterionToContract(
  contract: AgentRunContract | undefined,
  input: string,
): AgentRunContract | undefined {
  if (!contract || !inputLooksLikeContextualReviewFollowup(input)) {
    return contract
  }

  return addContextualFollowupCriterionToContract(contract, input)
}

export function addContextualFollowupCriterionToContract(
  contract: AgentRunContract | undefined,
  input: string,
): AgentRunContract | undefined {
  if (!contract || !inputLooksLikeContextualFollowup(input)) {
    return contract
  }

  return mergeContextualFollowupCriterionIntoContract(contract, input)
}

function mergeContextualFollowupCriterionIntoContract(
  contract: AgentRunContract,
  input: string,
): AgentRunContract {
  // A validation-only continuation can inherit a build contract that predates
  // the browser request. Enrich that inherited contract before the implement
  // node derives tool visibility from it, otherwise browser tools are hidden
  // precisely when the current turn explicitly asks for browser evidence.
  const enrichedContract = addRenderedUiValidationToContract(
    contract,
    input,
    Math.min(10, Math.max(6, contract.acceptanceCriteria.length + 4)),
    true,
  )
  // Durable contracts preserve the objective across turns, but authority and
  // no-touch boundaries belong to the current turn. A continuation such as
  // "keep implementing, but do not update README" must not inherit README as
  // a required artifact merely because an earlier turn mentioned it. Apply
  // the current deterministic boundary contract before copying artifacts.
  const currentTurnBoundary = createDurableRunContract(input)
  const boundedRequiredArtifacts = filterNoTouchRequiredArtifacts(
    enrichedContract.requiredArtifacts ?? [],
    currentTurnBoundary,
  )
  const boundedArtifactSections = filterNoTouchArtifactSections(
    enrichedContract.artifactSections ?? [],
    currentTurnBoundary,
  )
  const {
    requiredArtifacts: _inheritedRequiredArtifacts,
    artifactSections: _inheritedArtifactSections,
    ...boundedContract
  } = enrichedContract
  const requestGist = truncateText(
    input.split(/\r?\n/).map((line) => line.trim()).find(Boolean) ?? input.trim(),
    200,
  )
  const isReviewFollowup = inputLooksLikeContextualReviewFollowup(input)
  const followupText = isReviewFollowup
    ? [
        `${CONTEXTUAL_FOLLOWUP_REVIEW_CRITERION_PREFIX}:`,
        requestGist ? `"${requestGist}".` : 'the latest user follow-up.',
        'The final answer reviews the process/progress and result against the active run contract, states what was actually inspected or validated, identifies any fixes made or remaining gaps, and does not claim completion for unverified work.',
      ].join(' ')
    : [
        `${CONTEXTUAL_FOLLOWUP_ACTION_CRITERION_PREFIX}:`,
        requestGist ? `"${requestGist}".` : 'the latest user follow-up.',
        'The run continues from the active contract: identify the remaining gap or requested improvement, perform the requested follow-up action when possible, preserve the contract evidence and validation requirements, and report INCOMPLETE with the next concrete step if the follow-up cannot be fully handled.',
      ].join(' ')
  const baseCriteria = enrichedContract.acceptanceCriteria.filter((criterion) =>
    !CONTEXTUAL_FOLLOWUP_CRITERION_PREFIXES.some((prefix) => criterion.text.startsWith(prefix))
  )

  return {
    ...boundedContract,
    acceptanceCriteria: reindexAcceptanceCriteria([
      ...baseCriteria,
      { id: '', text: followupText },
    ]),
    constraints: mergeStringLists(
      userBoundaryConstraintsFromFallback(currentTurnBoundary),
      enrichedContract.constraints,
      6,
    ),
    outOfScope: mergeStringLists(
      userBoundaryOutOfScopeFromFallback(currentTurnBoundary),
      enrichedContract.outOfScope,
      4,
    ),
    ...(boundedRequiredArtifacts.length > 0
      ? { requiredArtifacts: boundedRequiredArtifacts.map((artifact) => ({ ...artifact })) }
      : {}),
    ...(enrichedContract.evidenceRequirements
      ? {
          evidenceRequirements: enrichedContract.evidenceRequirements.map((requirement) => ({
            ...requirement,
          })),
        }
      : {}),
    ...(boundedArtifactSections.length > 0
      ? { artifactSections: boundedArtifactSections.map((section) => ({ ...section })) }
      : {}),
  }
}

/**
 * Carries a durable contract into the next turn only when the user explicitly
 * frames that turn as a continuation. A contract describes one task, not the
 * whole chat session; retaining it for an unrelated request can make valid
 * tool work fail review against obsolete acceptance criteria.
 */
export function resolveRunContractForTurn(
  contract: AgentRunContract | undefined,
  input: string,
): AgentRunContract | undefined {
  if (
    !contract
    || !inputLooksLikeContextualFollowup(input)
    || !inputContinuesActiveRunContract(input)
    || inputLooksLikeSimpleWorkspaceFileOperation(input)
    // A narrowly bounded, read-only browser observation is a fresh per-turn
    // operation even when phrased as “continue”. It must not inherit an old
    // build/start contract or its screenshot/design requirements.
    || inputRequestsFocusedSingleBrowserObservation(input)
    // A complete current-turn process.start or terminal invocation supersedes
    // every previous workflow contract, including one that did not itself
    // contain an exact command. Otherwise an old acceptance ledger can make a
    // newly authorized operation look already complete or out of scope.
    || extractFocusedProcessStart(input) !== undefined
    || isFocusedTerminalCommandExecution(input)
    // An exact single command is a per-turn authority grant, never a durable
    // workflow contract. Carrying it forward would either rerun the old argv
    // or block a newly authorized repair as read-only.
    || contract.executionIntent?.requestedTerminalCommand !== undefined
    || contract.executionIntent?.requestedProcessStart !== undefined
    || contract.executionIntent?.constrainedProcessStart !== undefined
    // A later positive mutation request supersedes an earlier observation-only
    // boundary. Re-plan the new turn instead of retaining stale authority.
    || (
      contract.executionIntent?.workspaceMutation === 'forbidden'
      && inputExpressesMutationIntent(input)
    )
  ) {
    return undefined
  }
  return mergeContextualFollowupCriterionIntoContract(contract, input)
}

/**
 * Apply an LLM-owned semantic continuation decision while retaining the same
 * structural authority boundaries as the conservative fallback above. Exact
 * one-turn commands, focused observations, and a newly positive mutation that
 * supersedes an old read-only contract remain fresh contracts; ordinary
 * continuation wording is not reclassified by keyword rules here.
 */
export function continueRunContractForTurn(
  contract: AgentRunContract | undefined,
  input: string,
): AgentRunContract | undefined {
  if (
    !contract
    || inputLooksLikeSimpleWorkspaceFileOperation(input)
    || inputRequestsFocusedSingleBrowserObservation(input)
    || (
      inputExplicitlyForbidsFileMutation(input)
      && !inputExpressesMutationIntent(input)
    )
    || extractFocusedProcessStart(input) !== undefined
    || isFocusedTerminalCommandExecution(input)
    || contract.executionIntent?.requestedTerminalCommand !== undefined
    || contract.executionIntent?.requestedProcessStart !== undefined
    || contract.executionIntent?.constrainedProcessStart !== undefined
    || (
      contract.executionIntent?.workspaceMutation === 'forbidden'
      && inputExpressesMutationIntent(input)
    )
  ) {
    return undefined
  }
  return mergeContextualFollowupCriterionIntoContract(contract, input)
}

export function createDurableRunContract(input: string): AgentRunContract {
  // Planner summaries stay compact, but the conservative fallback is also the
  // only durable copy of user authority when planning fails. Preserve ordinary
  // multi-clause requests far enough to retain late source/validation/no-touch
  // boundaries instead of cutting them off at an arbitrary short synopsis.
  const summary = truncateText(input, MAX_FALLBACK_REQUEST_CHARS) || 'Complete the requested task.'
  const requiredArtifacts = extractPathLikeArtifacts(input)
  const prospectiveDocumentPhase = inputRequestsProspectiveDocumentPhase(input)
  const positiveCapabilityScope = inputPositiveCapabilityScope(input)
  const requiresNamedWebSourceObservation = /\bwebfetch\b/iu.test(positiveCapabilityScope)
    && /https?:\/\/[^\s<>()]+/iu.test(positiveCapabilityScope)
  const evidenceRequirements: AgentEvidenceRequirement[] = prospectiveDocumentPhase
    ? requiresNamedWebSourceObservation
      ? [{
          kind: 'source',
          description: 'Observe the user-named HTTP(S) source before authoring the requested document.',
          minSourceObservations: 1,
          sourceToolNames: ['webfetch'],
          requiresSearch: false,
        }]
      : []
    : fallbackEvidenceRequirementsForArtifacts(requiredArtifacts)
  const artifactSections = prospectiveDocumentPhase
    ? []
    : fallbackArtifactSectionsForArtifacts(requiredArtifacts)
  const requestGist = deriveRequestGist(input)
  const responseOnlyOutput = inputRequestsResponseOnlyOutput(input)
  const positiveMutationIntent = inputExpressesMutationIntent(input)
  const materializedOutcomeCriterion = positiveMutationIntent || requiredArtifacts.length > 0
    ? 'Any requested artifact, code change, analysis, or document is actually created or updated before claiming completion.'
    : responseOnlyOutput
      ? 'Any requested response-only draft or analysis is present in the final answer; no durable artifact is invented.'
      : undefined
  const explicitNoTouchTargets = extractExplicitNoTouchTargets(input)
  // A fully structured process.start request already defines the exact action
  // and carries no source-mutation authority. Treat it as an operational
  // contract even when the user does not redundantly say “do not edit files”.
  // extractFocusedProcessStart rejects explicit mutation and multi-tool turns.
  const requestedProcessStart = extractFocusedProcessStart(input)
  const constrainedProcessStart = requestedProcessStart
    ? undefined
    : extractConstrainedProcessStart(input)
  const boundedNoEditCommand = requestedProcessStart !== undefined
    || (
      inputExplicitlyForbidsFileMutation(input)
      && isFocusedTerminalCommandExecution(input)
    )
  const boundedReadOnlyTerminalWorkflow = !boundedNoEditCommand
    && constrainedProcessStart === undefined
    && inputExplicitlyForbidsFileMutation(input)
    && !positiveMutationIntent
    && inputRequestsTerminalCommandExecution(input)
  const boundedReadOnlyInspection = !boundedNoEditCommand
    && !boundedReadOnlyTerminalWorkflow
    && constrainedProcessStart === undefined
    && inputExplicitlyForbidsFileMutation(input)
    && !positiveMutationIntent
  const requestedTerminalCommand = boundedNoEditCommand
    ? extractFocusedTerminalCommand(input)
    : undefined
  const contract: AgentRunContract = {
    summary,
    acceptanceCriteria: reindexAcceptanceCriteria([
      {
        id: '',
        text: requestGist
          ? `The specific request is directly satisfied: "${requestGist}".`
          : 'The specific user request is directly satisfied (not a generic or adjacent task).',
      },
      {
        id: '',
        text: 'The requested outcome is completed, or the response explicitly says INCOMPLETE with the remaining work.',
      },
      ...(materializedOutcomeCriterion
        ? [{ id: '', text: materializedOutcomeCriterion }]
        : []),
      {
        id: '',
        text: prospectiveDocumentPhase
          ? 'The prospective document distinguishes supplied facts and constraints from proposed design decisions or assumptions.'
          : 'Important claims are grounded in observed tool results, repository evidence, source references, or validation output.',
      },
    ]),
    constraints: [
      'Keep work scoped to the user request and current workspace.',
      DEFAULT_RUN_CONTINUATION_POLICY,
      ...(prospectiveDocumentPhase ? [] : [DEFAULT_BROAD_COVERAGE_POLICY]),
      ...(positiveMutationIntent || requiredArtifacts.length > 0
        ? ['Prefer incremental durable artifacts over holding all work in the model context.']
        : responseOnlyOutput
          ? ['Keep the requested draft or analysis in the final response; do not invent a durable artifact.']
          : []),
      ...(prospectiveDocumentPhase
        ? [`${CURRENT_DOCUMENT_PHASE_CONSTRAINT_PREFIX}; do not begin downstream implementation until the user requests the next phase.`]
        : []),
      ...(prospectiveDocumentPhase
        ? ['Do not add repository evidence-map or acceptance-ledger boilerplate to this prospective document unless the user explicitly requests it.']
        : []),
      ...explicitNoTouchTargets.map(noTouchConstraintForTarget),
    ],
    outOfScope: [
      'Claiming a large task is complete from a shallow overview.',
      'Discarding partial progress when the run needs continuation.',
      ...(prospectiveDocumentPhase
        ? [`${DEFERRED_IMPLEMENTATION_OUT_OF_SCOPE_PREFIX}.`]
        : []),
      ...explicitNoTouchTargets.map(noTouchOutOfScopeForTarget),
    ],
    ...(requiredArtifacts.length > 0 ? { requiredArtifacts } : {}),
    ...(evidenceRequirements.length > 0 ? { evidenceRequirements } : {}),
    ...(artifactSections.length > 0 ? { artifactSections } : {}),
    ...(boundedNoEditCommand || boundedReadOnlyTerminalWorkflow || boundedReadOnlyInspection || (
      constrainedProcessStart !== undefined
      && !positiveMutationIntent
    )
      ? {
          executionIntent: {
            kind: boundedReadOnlyInspection
              ? 'inspection' as const
              : 'operational-action' as const,
            workspaceMutation: 'forbidden' as const,
            capabilities: boundedReadOnlyInspection
              ? explicitReadOnlyInspectionCapabilities(input)
              : constrainedProcessStart
                ? explicitWorkflowCapabilities(input)
                : [requestedProcessStart ? 'process' as const : 'terminal' as const],
            ...(requestedTerminalCommand
              ? {
                  requestedTerminalCommand: {
                    executable: requestedTerminalCommand.executable,
                    args: [...requestedTerminalCommand.args],
                  },
                }
              : {}),
            ...(requestedProcessStart
              ? {
                  requestedProcessStart: {
                    ...requestedProcessStart,
                    args: [...requestedProcessStart.args],
                    ...(typeof requestedProcessStart.network === 'object'
                      ? {
                          network: {
                            ...requestedProcessStart.network,
                            ...(requestedProcessStart.network.ports
                              ? { ports: [...requestedProcessStart.network.ports] }
                              : {}),
                          },
                        }
                      : {}),
                  },
                }
              : {}),
            ...(constrainedProcessStart
              ? {
                  constrainedProcessStart: {
                    ...constrainedProcessStart,
                    args: [...constrainedProcessStart.args],
                    ...(typeof constrainedProcessStart.network === 'object'
                      ? {
                          network: {
                            ...constrainedProcessStart.network,
                            ...(constrainedProcessStart.network.ports
                              ? { ports: [...constrainedProcessStart.network.ports] }
                              : {}),
                          },
                        }
                      : {}),
                  },
                }
              : {}),
          },
        }
      : {}),
    source: 'fallback',
  }
  return addRenderedUiValidationToContract(contract, input)
}

export function buildDurableRunContractPlanningRequest(input: {
  model: string
  userRequest: string
  cwd?: string
  fallback?: AgentRunContract
  nativeToolUse?: boolean
  availableToolNames?: readonly string[]
}): ChatRequest {
  const fallback = input.fallback ?? createDurableRunContract(input.userRequest)
  const prospectiveDocumentPhase = inputRequestsProspectiveDocumentPhase(input.userRequest)
  const suggestedEvidenceRequirements = prospectiveDocumentPhase
    ? []
    : fallback.evidenceRequirements
      ?? fallbackEvidenceRequirementsForArtifacts(fallback.requiredArtifacts ?? [])
  const suggestedArtifactSections = prospectiveDocumentPhase
    ? []
    : fallback.artifactSections
      ?? fallbackArtifactSectionsForArtifacts(fallback.requiredArtifacts ?? [])
  return {
    model: input.model,
    temperature: 0,
    maxTokens: RUN_CONTRACT_PLANNER_MAX_TOKENS,
    thinkingLevel: ThinkingLevel.Off,
    ...(input.nativeToolUse === false
      ? {}
      : {
          tools: [durableRunContractTool()],
          toolChoice: 'required' as const,
        }),
    messages: [
      {
        role: 'system',
        content: [
          'You create a durable run contract for an autonomous CLI agent.',
          'Decide semantically from the user request. Do not use keyword matching.',
          'The contract is not a step-by-step plan; it is the completion standard future agent turns and outcome review must enforce.',
          'Also classify the semantic execution posture. An operational-action operates an existing runtime or changes user-requested durable application state without changing workspace files. A workspace-change edits source/configuration. Inspection reads existing state. Artifact-production creates a requested durable output. Conversation needs no tools.',
          'For delegated future or recurring work, the current outcome is durable task registration or management, not immediate execution of the eventual payload. Preserve scheduling, delivery and task-scoped access requirements; include application-state without authorizing unrelated workspace edits.',
          'workspaceMutation=forbidden prohibits workspace file changes, not all state changes. Include application-state for explicitly requested durable preference, task, application-record or remote-state mutations; this does not authorize workspace writes or bypass approval. Inspection and conversation must not gain application-state. Use allowed when edits are optional, and required only when the requested outcome itself needs workspace changes.',
          'When the user explicitly restricts the run to named tools, populate executionIntent.allowedTools with exactly those canonical tool names. Leave allowedTools absent for ordinary requests; never infer a closed tool list merely from examples, capabilities, or a likely workflow.',
          'When the user explicitly requires distinct named tools in a particular order, preserve that order in executionIntent.toolSequence. Leave toolSequence absent when order is not required or when multiple semantic actions use the same tool name; a name-only sequence cannot distinguish those actions.',
          'Set executionIntent.retryPolicy="forbidden" only when the user explicitly prohibits retries or limits each action to one attempt. Preserve this failure boundary even when a repeated same-tool workflow cannot be represented by toolSequence. Otherwise leave retryPolicy absent.',
          'When workspace mutation is requested and the user explicitly names the files or directories that may be edited, preserve those user-named targets in executionIntent.authorizedWriteTargets. Do not include read-only inputs, protected targets, inferred dependencies, or convenience paths.',
          'Preserve user-named files or directories that must remain unchanged in executionIntent.protectedWriteTargets. Keep authorizedWriteTargets and protectedWriteTargets disjoint.',
          'A requested screenshot or other validation artifact does not by itself authorize product/source mutation. Classify the source workspace as read-only unless the requested outcome also requires source changes.',
          'Explicit negative tool boundaries are authoritative. If the user says not to read/search/list files, use terminal commands, use browser tools, or use process tools, omit that capability even when it would normally be convenient for evidence recovery; do not add it back through generic validation defaults.',
          'Make criteria concrete enough to reject shallow progress updates, partial files claimed as complete, and unsupported broad-work claims.',
          'Acceptance criteria must describe observable terminal outcomes of a successful or explicitly incomplete run. Put contingent recovery behavior for context, time, tool, provider, or iteration exhaustion in constraints instead; a fallback policy whose trigger never occurred is not an outcome that needs tool evidence.',
          'Likewise, do not make a conditional broad-work coverage policy an acceptance criterion. A bounded chat answer with a fixed small number of findings is not broad merely because it uses research or report-style prose. For genuinely broad or exhaustive work, add a concrete scope-coverage outcome grounded in the actual requested scope.',
          'For large analysis, writing, coding, research, or artifact work, require durable intermediate artifacts or evidence notes, representative source/tool evidence, and explicit INCOMPLETE reporting when the full scope is not finished.',
          'Do not invent unrelated requirements. Keep the contract scoped to the request.',
          'Before calling the contract tool, audit every proposed acceptance criterion, constraint, and out-of-scope item against the user request and fallback contract. Keep only user-stated outcomes, explicitly requested validation, supplied specifications, and generic completion/evidence boundaries already present in the fallback.',
          'Do not resolve an underspecified desired outcome by choosing additional observable product behavior. Keep ambiguous behavior at the level stated by the user and leave its concrete interpretation to source, test, or runtime evidence gathered during implementation.',
          'Do not invent behavior-specific preservation or compatibility constraints merely because they seem conventional; include them only when the user request, supplied specification, or fallback contract establishes them.',
          'For one exact literal substitution in one named file, keep the contract lightweight: require the requested replacement, preservation of surrounding content, and a read-back or diff confirming it. Do not invent build, compiler, test, runtime, or syntax-check criteria when the replacement does not alter program structure and the user did not ask for them.',
          input.nativeToolUse === false
            ? 'Return exactly one JSON object with this semantic shape and no prose or markdown fence:'
            : `Call ${DURABLE_RUN_CONTRACT_TOOL_NAME} exactly once with this semantic shape:`,
          '{"summary":"short goal","acceptanceCriteria":["criterion",...],"constraints":["constraint",...],"outOfScope":["out of scope",...],"executionIntent":{"kind":"operational-action|workspace-change|inspection|artifact-production|conversation","workspaceMutation":"forbidden|allowed|required","capabilities":["process|service|terminal|browser|filesystem-read|filesystem-write|network|application-state"],"allowedTools":["exact.tool.name"],"toolSequence":["first.tool","second.tool"],"retryPolicy":"forbidden","authorizedWriteTargets":["user/named/path"],"protectedWriteTargets":["read-only/input/path"]},"requiredArtifacts":[{"path":"relative/or/absolute/path","kind":"file|directory|document|other","description":"why it is required"}],"evidenceRequirements":[{"kind":"source|repository|artifact|validation|other","description":"what evidence must be gathered before completion","minSourceObservations":2,"minSourceFiles":6,"minSourceScopes":3,"sourceToolNames":["exact.read.tool"],"requiresArtifactEvidenceMap":true,"requiresArtifactSelfReview":true,"requiresSearch":true}],"artifactSections":[{"id":"short-id","title":"section title","artifactPath":"analysis.md","description":"what this section must cover","required":true}]}',
          'Use 3-7 acceptance criteria, 0-6 constraints, and 0-4 out-of-scope items.',
          'Populate requiredArtifacts when the user requested a concrete file, directory, document, or durable output path; preserve relative paths as relative.',
          'When no path was named, populate requiredArtifacts only if the user explicitly requested persistent document authoring, such as creating architecture documentation, writing a report, or saving a runbook. A request to analyze, inspect, review, explain, or summarize in chat does not authorize creating a file, even if the requested analysis is broad or detailed.',
          'Never invent artifacts for conversational answers, short explanations, or plain code fixes.',
          'Distinguish producing a document from looking something up: a request to check, query, or report *current live state* (cluster/service/process status, logs, metrics, resource usage — e.g. "check the cluster status with kubectl") is an inspection task whose deliverable is the command output summarized in the chat answer. Do not require a document artifact, evidence map, or artifact sections for such inspection tasks, even when the wording contains words like "report" or "status report", unless the user explicitly asked to save the result to a file.',
          'Treat conversational repository history, activity, change-summary, and working-tree review as inspection tasks too. Structured git.log/status/diff evidence can directly satisfy them. Do not require source-file reads, source search, per-file code inspection, or a durable artifact unless the user explicitly asks for code-level review, exhaustive analysis, or saved output.',
          'Populate evidenceRequirements when the request cannot be validated from the final answer alone, especially whole-codebase analysis, architecture extraction, requirement extraction, or code changes needing validation.',
          'Use kind=source for successful observations from web, application, page, memory, service, or local document tools. It has no implicit filesystem-file or search quota; set minSourceObservations only when multiple distinct observations are actually required.',
          'Use kind=repository for broad source-code coverage. Repository evidence defaults to representative filesystem reads and search/inventory; set minSourceFiles and minSourceScopes explicitly when the requested breadth needs a stronger floor.',
          'When an independently required source system maps unambiguously to one or more available observation tools, set sourceToolNames to those exact canonical names. This binds evidence, not execution authority. Use one requirement per independently required source system; omit sourceToolNames rather than inventing a name.',
          'For a localized code change or defect repair, describe the behavior/change and validation evidence needed but normally omit minSourceFiles, minSourceScopes, and requiresSearch. Those breadth fields are for genuinely broad repository coverage, not a quota for tracing one causal path; do not turn source-reading means into acceptance outcomes.',
          'Do not add a mandatory run, execution, test, compile, or runtime-output criterion merely because an artifact could be executed. Require runtime validation only when the user explicitly asked to run, test, validate, or inspect the behavior/output. File existence and read-back evidence are not runtime validation.',
          'Use minSourceScopes for broad repository/product analysis where evidence must span multiple independent modules, services, packages, directories, or subsystems rather than many files from one place.',
          'Set requiresArtifactEvidenceMap=true when a durable report, analysis, research brief, or design artifact must show which observed sources/scopes support its claims.',
          'Set requiresArtifactSelfReview=true when the final durable artifact should include an explicit acceptance-criteria checklist or self-review before completion.',
          'Populate artifactSections when the user requested a structured document, report, presentation, design, plan, or analysis artifact; titles should describe the required document sections rather than implementation steps.',
          'A staged pre-implementation design, plan, or specification for a future product is not an as-is repository audit. Keep that current document phase lightweight and do not add repository evidence-map, validation-results, or acceptance-ledger sections unless the user explicitly asks for them. Current/existing implementation or architecture analysis still requires observed repository evidence.',
          'For browser-rendered UI authoring, require a concrete design plan recorded as a completed todowrite item before coding; it must contain task-specific details for the actual target user/workflow, primary screens/states, responsive behavior, visual style, controls/interactions, and assets/media, not just placeholder labels. When the user has not bounded validation scope, require real browser screenshots at desktop and mobile viewports whose outputs say `Screenshot image attachment: attached`, a completed visual QA todowrite after the latest browser audit comparing the screenshots against the design plan with issues found or explicitly none while checking layout/spacing, text wrapping/overflow, contrast/readability, overlap/collision, controls/touch targets, and assets/media completeness, fixes for visible defects including low contrast and overlapping text/controls, fresh screenshots after fixes, console/interaction smoke evidence with attached active-state screenshot/layout audits at desktop and mobile viewports when interaction exists, and dynamic browser.evaluate evidence for games or animated canvas/WebGL work showing frame, pixel, position, or game-state changes over time. These are defaults, not authority to widen the request: when the user explicitly asks for one viewport, one screenshot, no mobile/desktop validation, or no interaction, preserve that boundary exactly and do not add excluded viewports, screenshots, interactions, dynamic checks, or visual-QA bookkeeping.',
        ].join(' '),
      },
      {
        role: 'user',
        content: [
          input.cwd ? `Active cwd: ${input.cwd}` : '',
          '[User request]',
          input.userRequest,
          '',
          '[Fallback contract to refine if useful]',
          `Goal: ${fallback.summary}`,
          'Acceptance criteria:',
          ...fallback.acceptanceCriteria.map((criterion) => `- ${criterion.text}`),
          'Fallback constraints:',
          ...fallback.constraints.map((constraint) => `- ${constraint}`),
          fallback.requiredArtifacts?.length
            ? `Required artifacts:\n${fallback.requiredArtifacts.map((artifact) => `- ${artifact.path}`).join('\n')}`
            : '',
          suggestedEvidenceRequirements.length
            ? `Suggested evidence requirements for document/report artifacts:\n${suggestedEvidenceRequirements.map((requirement) => `- ${requirement.kind}: ${requirement.description}`).join('\n')}`
            : '',
          suggestedArtifactSections.length
            ? `Suggested artifact sections for structured document/report artifacts:\n${suggestedArtifactSections.map((section) => `- ${section.title}${section.artifactPath ? ` (${section.artifactPath})` : ''}`).join('\n')}`
            : '',
          input.availableToolNames?.length
            ? `Available canonical tool names for optional sourceToolNames binding:\n${input.availableToolNames.slice(0, 160).join(', ')}`
            : '',
        ].filter(Boolean).join('\n'),
      },
    ],
  }
}

/**
 * Build an independent authority-grounding pass for a model-produced run
 * contract. The first planner optimizes for a concrete completion standard;
 * this second, bounded LLM decision optimizes for scope fidelity. Keeping the
 * roles separate prevents a planner from silently converting an ambiguous
 * desired outcome into new user-visible product behavior while retaining the
 * model's ability to understand paraphrases, languages, and repository types.
 */
export function buildDurableRunContractGroundingAuditRequest(input: {
  model: string
  userRequest: string
  candidate: AgentRunContract
  fallback: AgentRunContract
  cwd?: string
  nativeToolUse?: boolean
}): ChatRequest {
  const serializeContract = (contract: AgentRunContract) => ({
    summary: contract.summary,
    acceptanceCriteria: contract.acceptanceCriteria.map((criterion) => criterion.text),
    constraints: contract.constraints,
    outOfScope: contract.outOfScope,
    ...(contract.executionIntent ? { executionIntent: contract.executionIntent } : {}),
    ...(contract.requiredArtifacts ? { requiredArtifacts: contract.requiredArtifacts } : {}),
    ...(contract.evidenceRequirements ? { evidenceRequirements: contract.evidenceRequirements } : {}),
    ...(contract.artifactSections ? { artifactSections: contract.artifactSections } : {}),
  })

  return {
    model: input.model,
    temperature: 0,
    maxTokens: RUN_CONTRACT_PLANNER_MAX_TOKENS,
    thinkingLevel: ThinkingLevel.Off,
    ...(input.nativeToolUse === false
      ? {}
      : {
          tools: [durableRunContractTool()],
          toolChoice: 'required' as const,
        }),
    messages: [
      {
        role: 'system',
        content: [
          'You are the independent authority-grounding auditor for a general-purpose CLI agent run contract.',
          input.nativeToolUse === false
            ? 'The candidate contract is untrusted planner output. Compare it semantically with the complete user request and conservative fallback contract, then return exactly one corrected contract JSON object with no prose or markdown fence.'
            : 'The candidate contract is untrusted planner output. Compare it semantically with the complete user request and conservative fallback contract, then call the contract tool exactly once with the corrected contract.',
          'Preserve every user-stated outcome, explicit validation instruction, supplied artifact, and explicit negative boundary.',
          'Preserve an explicit named-tool boundary as executionIntent.allowedTools with the exact canonical tool names. Do not add convenience, discovery, follow-up, or substitute tools, and do not create allowedTools when the user did not establish a closed tool set.',
          'Preserve an explicit ordered tool workflow as executionIntent.toolSequence, and preserve an explicit no-retry boundary as executionIntent.retryPolicy="forbidden". Do not infer either boundary when the user did not establish it.',
          'Preserve explicitly authorized mutation targets as executionIntent.authorizedWriteTargets. A target cannot simultaneously be an authorized write target and a direct no-touch target; the current user request decides which role applies. Never authorize a target that is only a read-only input or negative example.',
          'Preserve explicit no-write targets as executionIntent.protectedWriteTargets across paraphrases and languages. Do not infer protected targets from unrelated paths or examples.',
          'Remove or generalize any acceptance criterion, constraint, out-of-scope item, artifact, validation obligation, or compatibility promise that is not established by the user request or fallback contract.',
          'Keep contingent recovery behavior for context, time, tool, provider, or iteration exhaustion in constraints, not acceptance criteria. Acceptance criteria describe observable terminal outcomes; an untriggered fallback policy must not become an evidence-bearing deliverable.',
          'Keep a conditional broad-work coverage policy in constraints too. Do not classify a bounded chat answer with a fixed small number of findings as broad solely because it involves research or reporting; require a concrete coverage outcome only when the requested scope is genuinely broad or exhaustive.',
          'Do not resolve an underspecified desired outcome by selecting additional observable product behavior. Keep ambiguous behavior at the level stated by the user; source, tests, configuration, and runtime evidence gathered during execution will determine its concrete implementation.',
          'Do not assume conventional alternate interactions, defaults, compatibility guarantees, preservation duties, tools, or validation steps. They are contract requirements only when the request, supplied specification, or fallback establishes them.',
          'The fallback is a scope floor, not wording that must be copied. You may retain the candidate\'s clearer phrasing and explicit requested checks when they remain grounded.',
          input.nativeToolUse === false
            ? 'Return only the corrected JSON object.'
            : 'Return no prose and no other tool call.',
        ].join(' '),
      },
      {
        role: 'user',
        content: [
          input.cwd ? `Active cwd: ${input.cwd}` : '',
          '[Complete user request]',
          input.userRequest,
          '',
          '[Conservative fallback contract]',
          JSON.stringify(serializeContract(input.fallback)),
          '',
          '[Untrusted candidate contract]',
          JSON.stringify(serializeContract(input.candidate)),
        ].filter(Boolean).join('\n'),
      },
    ],
  }
}

function extractJsonObject(content: string): Record<string, unknown> | null {
  const match = content.match(/\{[\s\S]*\}/)
  if (!match) {
    return null
  }
  try {
    const parsed = JSON.parse(match[0]) as unknown
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : null
  } catch {
    return null
  }
}

function normalizeStringList(value: unknown, maxItems: number, maxChars: number): string[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value
    .map((entry) => {
      if (typeof entry === 'string') {
        return entry
      }
      if (entry && typeof entry === 'object' && !Array.isArray(entry)) {
        const record = entry as Record<string, unknown>
        return typeof record.text === 'string' ? record.text : ''
      }
      return ''
    })
    .map((entry) => truncateText(entry, maxChars))
    .filter(Boolean)
    .slice(0, maxItems)
}

function mergeStringLists(left: readonly string[], right: readonly string[], maxItems: number): string[] {
  const merged: string[] = []
  const seen = new Set<string>()
  for (const entry of [...left, ...right]) {
    const normalized = normalizeText(entry)
    if (!normalized) {
      continue
    }
    const key = normalized.toLowerCase()
    if (seen.has(key)) {
      continue
    }
    seen.add(key)
    merged.push(normalized)
    if (merged.length >= maxItems) {
      break
    }
  }
  return merged
}

function normalizedContractTargetKey(value: string): string {
  return normalizeNoTouchTarget(value).replace(/\\/g, '/').toLowerCase()
}

function boundaryEntryTarget(entry: string, prefix: string): string {
  if (!entry.startsWith(prefix)) return ''
  const remainder = entry.slice(prefix.length)
  const quoted = remainder.match(/^"([^"]+)"/u)?.[1]
  return normalizeNoTouchTarget(quoted ?? remainder.split(';')[0] ?? '')
}

function boundaryConflictsWithAuthorizedWrite(
  entry: string,
  prefix: string,
  intent: AgentExecutionIntent | undefined,
): boolean {
  const target = boundaryEntryTarget(entry, prefix)
  if (!target) return false
  const targetKey = normalizedContractTargetKey(target)
  return (intent?.authorizedWriteTargets ?? [])
    .some((authorized) => normalizedContractTargetKey(authorized) === targetKey)
}

/**
 * Resolve a structurally contradictory contract in favor of the LLM's
 * explicit, request-grounded write authority. This is target equality only:
 * authorizing one file does not erase protection on its parent directory or
 * on any sibling target.
 */
export function reconcileRunContractWriteAuthority(
  contract: AgentRunContract,
): AgentRunContract {
  const intent = contract.executionIntent
  const authorizedKeys = new Set(
    (intent?.authorizedWriteTargets ?? []).map(normalizedContractTargetKey),
  )
  const protectedWriteTargets = (intent?.protectedWriteTargets ?? [])
    .filter((target) => !authorizedKeys.has(normalizedContractTargetKey(target)))
  if (authorizedKeys.size === 0 && protectedWriteTargets.length === 0) return contract
  const filteredConstraints = contract.constraints.filter((entry) =>
    !boundaryConflictsWithAuthorizedWrite(
      entry,
      NO_TOUCH_CONSTRAINT_PREFIX,
      intent,
    ))
  const filteredOutOfScope = contract.outOfScope.filter((entry) =>
    !boundaryConflictsWithAuthorizedWrite(
      entry,
      NO_TOUCH_OUT_OF_SCOPE_PREFIX,
      intent,
    ))
  return {
    ...contract,
    constraints: mergeStringLists(
      protectedWriteTargets.map(noTouchConstraintForTarget),
      filteredConstraints,
      6,
    ),
    outOfScope: mergeStringLists(
      protectedWriteTargets.map(noTouchOutOfScopeForTarget),
      filteredOutOfScope,
      4,
    ),
    ...(intent
      ? {
          executionIntent: {
            ...intent,
            protectedWriteTargets: protectedWriteTargets.length > 0
              ? [...protectedWriteTargets]
              : undefined,
          },
        }
      : {}),
  }
}

function userBoundaryConstraintsFromFallback(fallback: AgentRunContract): string[] {
  return fallback.constraints.filter((entry) =>
    (entry.startsWith(NO_TOUCH_CONSTRAINT_PREFIX)
      && !boundaryConflictsWithAuthorizedWrite(
        entry,
        NO_TOUCH_CONSTRAINT_PREFIX,
        fallback.executionIntent,
      ))
    || entry.startsWith(CURRENT_DOCUMENT_PHASE_CONSTRAINT_PREFIX)
  )
}

function userBoundaryOutOfScopeFromFallback(fallback: AgentRunContract): string[] {
  return fallback.outOfScope.filter((entry) =>
    (entry.startsWith(NO_TOUCH_OUT_OF_SCOPE_PREFIX)
      && !boundaryConflictsWithAuthorizedWrite(
        entry,
        NO_TOUCH_OUT_OF_SCOPE_PREFIX,
        fallback.executionIntent,
      ))
    || entry.startsWith(DEFERRED_IMPLEMENTATION_OUT_OF_SCOPE_PREFIX)
  )
}

function noTouchTargetsFromFallback(fallback: AgentRunContract): string[] {
  const targets: string[] = []
  for (const entry of userBoundaryConstraintsFromFallback(fallback)) {
    const target = normalizeNoTouchTarget(
      entry.slice(NO_TOUCH_CONSTRAINT_PREFIX.length).split(';')[0] ?? '',
    )
    if (target && !targets.some((existing) => existing.toLowerCase() === target.toLowerCase())) {
      targets.push(target)
    }
  }
  return targets
}

function artifactPathTouchesNoTouchTarget(path: string, target: string): boolean {
  const normalizedPath = normalizeArtifactPath(path).replace(/\\/g, '/').toLowerCase()
  const normalizedTarget = normalizeNoTouchTarget(target).replace(/\\/g, '/').toLowerCase()
  if (!normalizedPath || !normalizedTarget) return false
  if (
    normalizedTarget.includes('/')
    || normalizedTarget.startsWith('.')
    || normalizedTarget.startsWith('~')
  ) {
    return normalizedPath === normalizedTarget
      || normalizedPath.startsWith(`${normalizedTarget}/`)
      || normalizedPath.includes(`/${normalizedTarget}/`)
  }
  return normalizedPath.split('/').includes(normalizedTarget)
}

function filterNoTouchRequiredArtifacts(
  artifacts: AgentRequiredArtifact[],
  fallback: AgentRunContract,
): AgentRequiredArtifact[] {
  const targets = noTouchTargetsFromFallback(fallback)
  if (targets.length === 0) return artifacts
  return artifacts.filter((artifact) =>
    !targets.some((target) => artifactPathTouchesNoTouchTarget(artifact.path, target)),
  )
}

function filterNoTouchArtifactSections(
  sections: AgentArtifactSection[],
  fallback: AgentRunContract,
): AgentArtifactSection[] {
  const targets = noTouchTargetsFromFallback(fallback)
  if (targets.length === 0) return sections
  return sections.filter((section) =>
    !section.artifactPath
    || !targets.some((target) => artifactPathTouchesNoTouchTarget(section.artifactPath!, target)),
  )
}

function filterArtifactSectionsToRequiredDocuments(
  sections: AgentArtifactSection[],
  artifacts: AgentRequiredArtifact[],
): AgentArtifactSection[] {
  const documentPaths = new Set(
    documentArtifacts(artifacts).map((artifact) => normalizeArtifactPath(artifact.path).toLowerCase()),
  )
  if (documentPaths.size === 0) return []
  return sections.filter((section) =>
    !section.artifactPath
    || documentPaths.has(normalizeArtifactPath(section.artifactPath).toLowerCase()),
  )
}

function normalizeRequiredArtifacts(value: unknown, fallback: AgentRequiredArtifact[] = []): AgentRequiredArtifact[] {
  if (!Array.isArray(value)) {
    return fallback
  }
  const artifacts = value.map((entry): AgentRequiredArtifact | null => {
    if (typeof entry === 'string') {
      return { path: entry, kind: artifactKind(undefined, entry) }
    }
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      return null
    }
    const record = entry as Record<string, unknown>
    const path = typeof record.path === 'string'
      ? record.path
      : typeof record.file === 'string'
        ? record.file
        : typeof record.filePath === 'string'
          ? record.filePath
          : ''
    if (!path) {
      return null
    }
    return {
      path,
      kind: artifactKind(record.kind, path),
      ...(typeof record.description === 'string' ? { description: record.description } : {}),
    }
  }).filter((entry): entry is AgentRequiredArtifact => entry !== null)
  const normalized = dedupeArtifacts(artifacts)
  return normalized.length > 0 ? normalized : fallback
}

function evidenceKind(value: unknown): AgentEvidenceRequirement['kind'] {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : ''
  if (
    normalized === 'source'
    || normalized === 'repository'
    || normalized === 'artifact'
    || normalized === 'validation'
    || normalized === 'other'
  ) {
    return normalized
  }
  return 'other'
}

function normalizeOptionalPositiveInteger(value: unknown, max: number): number | undefined {
  const numeric = typeof value === 'number'
    ? value
    : typeof value === 'string'
      ? Number(value)
      : NaN
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return undefined
  }
  return Math.min(max, Math.floor(numeric))
}

function normalizeEvidenceRequirements(
  value: unknown,
  availableToolNames?: readonly string[],
): AgentEvidenceRequirement[] {
  if (!Array.isArray(value)) {
    return []
  }
  const requirements: AgentEvidenceRequirement[] = []
  for (const entry of value) {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      continue
    }
    const record = entry as Record<string, unknown>
    const description = typeof record.description === 'string'
      ? truncateText(record.description, MAX_LIST_ITEM_CHARS)
      : typeof record.text === 'string'
        ? truncateText(record.text, MAX_LIST_ITEM_CHARS)
        : ''
    if (!description) {
      continue
    }
    const minSourceObservations = normalizeOptionalPositiveInteger(record.minSourceObservations, 24)
    const minSourceFiles = normalizeOptionalPositiveInteger(record.minSourceFiles, 24)
    const minSourceScopes = normalizeOptionalPositiveInteger(record.minSourceScopes, 12)
    const availableTools = availableToolNames
      ? new Map(availableToolNames.map((name) => [name.toLowerCase(), name]))
      : null
    const sourceToolNames = Array.isArray(record.sourceToolNames)
      ? [...new Set(record.sourceToolNames.flatMap((entry): string[] => {
          if (typeof entry !== 'string') return []
          const normalized = entry.trim().toLowerCase()
          if (!/^[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*$/u.test(normalized)) return []
          if (!availableTools) return [normalized]
          const available = availableTools.get(normalized)
          return available ? [available] : []
        }))].slice(0, 8)
      : []
    requirements.push({
      kind: evidenceKind(record.kind),
      description,
      ...(minSourceObservations != null ? { minSourceObservations } : {}),
      ...(minSourceFiles != null ? { minSourceFiles } : {}),
      ...(minSourceScopes != null ? { minSourceScopes } : {}),
      ...(sourceToolNames.length > 0 ? { sourceToolNames } : {}),
      ...(typeof record.requiresArtifactEvidenceMap === 'boolean'
        ? { requiresArtifactEvidenceMap: record.requiresArtifactEvidenceMap }
        : {}),
      ...(typeof record.requiresArtifactSelfReview === 'boolean'
        ? { requiresArtifactSelfReview: record.requiresArtifactSelfReview }
        : {}),
      ...(typeof record.requiresSearch === 'boolean'
        ? { requiresSearch: record.requiresSearch }
        : {}),
    })
    if (requirements.length >= MAX_EVIDENCE_REQUIREMENTS) {
      break
    }
  }
  return requirements
}

function normalizeSectionId(value: unknown, fallback: string): string {
  const raw = typeof value === 'string' ? value.trim().toLowerCase() : ''
  const normalized = raw
    .replace(/[^a-z0-9_.-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 64)
  return normalized || fallback
}

function normalizeArtifactSections(
  value: unknown,
  fallbackArtifactPath?: string,
): AgentArtifactSection[] {
  if (!Array.isArray(value)) {
    return []
  }
  const sections: AgentArtifactSection[] = []
  const seen = new Set<string>()
  for (const entry of value) {
    const record = entry && typeof entry === 'object' && !Array.isArray(entry)
      ? entry as Record<string, unknown>
      : undefined
    const title = typeof entry === 'string'
      ? truncateText(entry, MAX_LIST_ITEM_CHARS)
      : typeof record?.title === 'string'
        ? truncateText(record.title, MAX_LIST_ITEM_CHARS)
        : typeof record?.heading === 'string'
          ? truncateText(record.heading, MAX_LIST_ITEM_CHARS)
          : ''
    if (!title) {
      continue
    }
    const fallbackId = `section-${sections.length + 1}`
    const id = normalizeSectionId(record?.id ?? record?.key, fallbackId)
    const key = `${id}:${title.toLowerCase()}`
    if (seen.has(key)) {
      continue
    }
    seen.add(key)
    const artifactPath = typeof record?.artifactPath === 'string'
      ? normalizeArtifactPath(record.artifactPath)
      : typeof record?.path === 'string'
        ? normalizeArtifactPath(record.path)
        : fallbackArtifactPath
    sections.push({
      id,
      title,
      ...(typeof record?.description === 'string'
        ? { description: truncateText(record.description, MAX_LIST_ITEM_CHARS) }
        : {}),
      ...(artifactPath ? { artifactPath } : {}),
      ...(typeof record?.required === 'boolean' ? { required: record.required } : { required: true }),
    })
    if (sections.length >= MAX_ARTIFACT_SECTIONS) {
      break
    }
  }
  return sections
}

const EXECUTION_INTENT_KINDS = new Set<AgentExecutionIntent['kind']>([
  'operational-action',
  'workspace-change',
  'inspection',
  'artifact-production',
  'conversation',
])
const WORKSPACE_MUTATION_INTENTS = new Set<AgentExecutionIntent['workspaceMutation']>([
  'forbidden',
  'allowed',
  'required',
])
const EXECUTION_CAPABILITIES = new Set<AgentExecutionCapability>([
  'process',
  'service',
  'terminal',
  'browser',
  'filesystem-read',
  'filesystem-write',
  'network',
  'application-state',
])

function normalizeExecutionIntent(
  value: unknown,
  userRequest: string,
  availableToolNames?: readonly string[],
): AgentExecutionIntent | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  const record = value as Record<string, unknown>
  const kind = record.kind
  const workspaceMutation = record.workspaceMutation
  if (
    typeof kind !== 'string'
    || !EXECUTION_INTENT_KINDS.has(kind as AgentExecutionIntent['kind'])
    || typeof workspaceMutation !== 'string'
    || !WORKSPACE_MUTATION_INTENTS.has(workspaceMutation as AgentExecutionIntent['workspaceMutation'])
  ) {
    return undefined
  }

  // A planner classification may narrow authority, but it must never suppress
  // an explicit authoring request. Fail open to the existing coder path when
  // the shared contract intent signal and the planner disagree.
  const applicationStateOnly = kind === 'operational-action'
    && Array.isArray(record.capabilities)
    && record.capabilities.includes('application-state')
    && !record.capabilities.includes('filesystem-write')
  if (workspaceMutation === 'forbidden' && !applicationStateOnly && inputExpressesMutationIntent(userRequest)) {
    return undefined
  }
  if (kind === 'workspace-change' && workspaceMutation === 'forbidden') {
    return undefined
  }

  const declaredCapabilities = Array.isArray(record.capabilities)
    ? [...new Set(record.capabilities
        .filter((entry): entry is AgentExecutionCapability =>
          typeof entry === 'string'
          && EXECUTION_CAPABILITIES.has(entry as AgentExecutionCapability),
        ))]
    : []
  const capabilities = applyExplicitCapabilityBoundaries(declaredCapabilities, userRequest)
  const userNamedTools = explicitCanonicalToolNames(userRequest, availableToolNames)
  const validatedAllowedTools = Array.isArray(record.allowedTools)
    ? [...new Set(record.allowedTools.flatMap((entry): string[] => {
        if (typeof entry !== 'string') return []
        const normalized = entry.toLowerCase()
        return /^[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*$/u.test(normalized)
          && userNamedTools.has(normalized)
          ? [normalized]
          : []
      }))].slice(0, 32)
    : []
  const allowedTools = validatedAllowedTools.length > 0
    ? validatedAllowedTools
    : undefined
  const toolSequence = normalizeRepresentableToolSequence(
    Array.isArray(record.toolSequence)
      ? record.toolSequence.flatMap((entry): string[] => {
          if (typeof entry !== 'string') return []
          const normalized = entry.toLowerCase()
          return /^[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*$/u.test(normalized)
            && userNamedTools.has(normalized)
            && (!allowedTools || allowedTools.includes(normalized))
            ? [normalized]
            : []
        }).slice(0, 32)
      : [],
  )
  const retryPolicy = record.retryPolicy === 'forbidden'
    && inputExplicitlyForbidsRetries(userRequest)
    ? 'forbidden' as const
    : undefined
  const authorizedWriteTargets = normalizeAuthorizedWriteTargets(
    record.authorizedWriteTargets,
    userRequest,
    workspaceMutation,
    capabilities,
  )
  const protectedWriteTargets = normalizeProtectedWriteTargets(
    record.protectedWriteTargets,
    userRequest,
  ).filter((target) => !authorizedWriteTargets.some((authorized) => (
    normalizedContractTargetKey(authorized) === normalizedContractTargetKey(target)
  )))
  return {
    kind: kind as AgentExecutionIntent['kind'],
    workspaceMutation: workspaceMutation as AgentExecutionIntent['workspaceMutation'],
    capabilities: workspaceMutation === 'forbidden'
      ? capabilities.filter((capability) => capability !== 'filesystem-write')
      : capabilities,
    ...(allowedTools ? { allowedTools } : {}),
    ...(toolSequence.length > 0 ? { toolSequence } : {}),
    ...(retryPolicy ? { retryPolicy } : {}),
    ...(authorizedWriteTargets.length > 0 ? { authorizedWriteTargets } : {}),
    ...(protectedWriteTargets.length > 0 ? { protectedWriteTargets } : {}),
  }
}

// Tokens from an artifact filename stem, used to gauge whether a planner-named
// artifact is grounded in the request text.
function artifactStemTokens(path: string): string[] {
  const base = path.split(/[\\/]/).pop() ?? path
  const stem = base.replace(/\.[^.]+$/, '')
  return stem
    .split(/[^a-zA-Z0-9]+/)
    .filter((token) => token.length >= 3)
    .map((token) => token.toLowerCase())
}

// Whether a planner-proposed artifact path is related to the request. A path is
// kept when it matches one the request itself named, or when its filename stem
// appears in the request text. When the stem is uninformative we keep it (can't
// judge). This drops over-broad planner artifacts that are unrelated to a
// request that already specified its own concrete targets.
function isPlannerArtifactRelatedToRequest(
  path: string,
  requestText: string,
  requestNamedPaths: string[],
): boolean {
  const normalized = normalizeArtifactPath(path)
  if (requestNamedPaths.some((named) => normalizeArtifactPath(named) === normalized)) {
    return true
  }
  const tokens = artifactStemTokens(path)
  if (tokens.length === 0) return true
  const request = requestText.toLowerCase()
  return tokens.some((token) => request.includes(token))
}

const EXPLICIT_RUNTIME_VALIDATION_REQUEST_RE =
  /(?:실행(?:해|해서|하고|한\s*뒤|해주세요|해줘)|돌려(?:봐|서|줘|주세요)|테스트(?:해|해서|하고|해주세요|해줘)|검증(?:해|해서|하고|해주세요|해줘)|(?:동작|작동|출력|실행\s*결과).{0,24}확인(?:해|해서|해주세요|해줘)|\b(?:run|execute|validate|verify)\s+(?:it\b|this\b|the\b|that\b|generated\b)|\btest\s+(?:it\b|this\b|the\b|that\b|generated\b)|\bcheck\s+(?:its?\s+)?(?:runtime|behavior|output|result)\b)/iu
const PLANNER_RUNTIME_VALIDATION_RE =
  /(?:실행(?:하고|하여|해서|한\s*뒤|해야|되었|결과)|돌려(?:서|보고)|테스트(?:하고|하여|해서|해야|되었|통과)|검증(?:하고|하여|해서|해야|되었)|(?:스크립트|프로그램|코드|파일).{0,32}(?:정상적으로?\s*)?실행(?:된다|되어야|됨)|(?:동작|작동|출력|실행\s*결과).{0,32}(?:확인|검증)|\b(?:run|execute|executed|execution)\s+(?:it\b|this\b|the\b|that\b|generated\b)|\b(?:script|program|code|file|artifact)\b.{0,32}\b(?:runs?|executes?)\s+(?:successfully|correctly|without\s+errors?)\b|\b(?:tests?|test\s+suite)\s+(?:must\s+)?(?:pass|passes|passed|run|runs|succeed)|\b(?:output|runtime\s+behavior|execution\s+result)\b.{0,48}\b(?:verify|verified|validate|validated|confirm|confirmed|check|checked)\b|\b(?:verify|validate|confirm|check)\b.{0,48}\b(?:output|runtime\s+behavior|execution\s+result)\b)/iu

function inputExplicitlyRequestsRuntimeValidation(input: string): boolean {
  return EXPLICIT_RUNTIME_VALIDATION_REQUEST_RE.test(input)
}

function plannerRequirementMandatesRuntimeValidation(text: string): boolean {
  return PLANNER_RUNTIME_VALIDATION_RE.test(text)
}

/**
 * Semantic planners see product-wide browser UI guardrails and can copy them
 * into unrelated CLI/TUI or backend contracts. Those requirements are outside
 * the requested surface: they force browser tools, responsive viewports, and
 * visual-QA loops that the artifact cannot satisfy. Ordinary runtime and
 * keyboard-interaction criteria remain valid because they do not match these
 * browser-rendered surface signals.
 */
function plannerRequirementMandatesRenderedUiValidation(text: string): boolean {
  return /(?:\bbrowser(?:\.|\s)|\bscreenshots?\b|\bviewports?\b|\bresponsive\s+layout\b|\bdesktop\b[^\n]{0,80}\bmobile\b|\bmobile\b[^\n]{0,80}\bdesktop\b|\bvisual\s+qa\b|\blayout[- ]audit\b|\btouch\s+targets?\b|\bassets?\s*(?:or|and|\/)\s*media\b|브라우저|스크린샷|뷰포트|반응형\s*레이아웃|시각\s*검수|레이아웃\s*검수|터치\s*대상)/iu.test(text)
}

/**
 * Keep prose requirements subordinate to the structured capability contract.
 * This deliberately does not infer task meaning from the request: it only
 * removes positive browser-validation obligations when the semantic execution
 * intent did not authorize the browser capability.
 */
export function reconcileRunContractExecutionCapabilities<T extends RenderedUiContractShape>(
  contract: T,
): T {
  const intent = contract.executionIntent
  if (
    intent?.capabilityPolicy !== 'closed'
    || intent.capabilities.includes('browser')
  ) return contract

  const acceptanceCriteria = reindexAcceptanceCriteria(
    contract.acceptanceCriteria.filter((criterion) =>
      !plannerRequirementMandatesRenderedUiValidation(criterion.text)),
  )
  const evidenceRequirements = (contract.evidenceRequirements ?? []).filter((requirement) =>
    !plannerRequirementMandatesRenderedUiValidation(requirement.description))
  const {
    evidenceRequirements: _previousEvidenceRequirements,
    ...contractWithoutEvidenceRequirements
  } = contract
  return {
    ...contractWithoutEvidenceRequirements,
    acceptanceCriteria,
    ...(evidenceRequirements.length > 0 ? { evidenceRequirements } : {}),
  } as T
}

function fallbackAllowsRuntimeValidation(fallback: AgentRunContract): boolean {
  return inputExplicitlyRequestsRuntimeValidation(fallback.summary)
}

export function parseDurableRunContractPlan(
  content: string,
  fallback: AgentRunContract,
  options: { userRequest?: string; availableToolNames?: readonly string[] } = {},
): AgentRunContract {
  const parsed = extractJsonObject(content)
  if (!parsed) {
    return fallback
  }

  const summary = typeof parsed.summary === 'string'
    ? truncateText(parsed.summary, MAX_SUMMARY_CHARS)
    : ''
  const acceptanceCriteria = normalizeStringList(
    parsed.acceptanceCriteria ?? parsed.criteria,
    7,
    MAX_CRITERION_CHARS,
  ).filter((criterion) => !isDefaultContingentRunPolicy(criterion))
  if (!summary || acceptanceCriteria.length < 2) {
    return fallback
  }
  const requestText = options.userRequest ?? fallback.summary
  const prospectiveDocumentPhase = inputRequestsProspectiveDocumentPhase(requestText)
  const allowsRuntimeValidation = inputExplicitlyRequestsRuntimeValidation(requestText)
    || fallbackAllowsRuntimeValidation(fallback)
  const allowsRenderedUiValidation = fallback.executionIntent?.capabilityPolicy === 'closed'
    ? fallback.executionIntent.capabilities.includes('browser')
    : inputRequiresRenderedUiValidation(requestText)
  const interfaceScopedAcceptanceCriteria = allowsRenderedUiValidation
    ? acceptanceCriteria
    : acceptanceCriteria.filter((criterion) =>
      !plannerRequirementMandatesRenderedUiValidation(criterion))
  const groundedAcceptanceCriteria = allowsRuntimeValidation
    ? interfaceScopedAcceptanceCriteria
    : interfaceScopedAcceptanceCriteria.filter((criterion) =>
      !plannerRequirementMandatesRuntimeValidation(criterion))
  const finalAcceptanceCriteria = groundedAcceptanceCriteria.length >= 2
    ? groundedAcceptanceCriteria
    : fallback.acceptanceCriteria.map((criterion) => criterion.text)

  const normalizedRequiredArtifacts = normalizeRequiredArtifacts(
    parsed.requiredArtifacts ?? parsed.artifacts ?? parsed.outputs,
    fallback.requiredArtifacts ?? [],
  )
  const plannerMayProposeArtifact = (fallback.requiredArtifacts?.length ?? 0) > 0
    || inputRequestsDurableDocument(requestText)
  // When the request itself named concrete artifacts, drop planner artifacts
  // that are unrelated to the request text (over-broad hallucinated outputs).
  // If filtering removes everything, fall back to the request's own artifacts
  // so a real target is never lost.
  const requestNamedPaths = (fallback.requiredArtifacts ?? []).map((artifact) => artifact.path)
  const plannerRequiredArtifacts = !plannerMayProposeArtifact
    ? (fallback.requiredArtifacts ?? [])
    : requestNamedPaths.length > 0
      ? (() => {
          const related = normalizedRequiredArtifacts.filter((artifact) =>
            isPlannerArtifactRelatedToRequest(artifact.path, fallback.summary ?? '', requestNamedPaths),
          )
          return related.length > 0 ? related : (fallback.requiredArtifacts ?? [])
        })()
      : normalizedRequiredArtifacts
  const requiredArtifacts = filterNoTouchRequiredArtifacts(plannerRequiredArtifacts, fallback)
  const normalizedParsedEvidenceRequirements = normalizeEvidenceRequirements(
    parsed.evidenceRequirements ?? parsed.requiredEvidence ?? parsed.evidence,
    options.availableToolNames,
  )
  const interfaceScopedEvidenceRequirements = allowsRenderedUiValidation
    ? normalizedParsedEvidenceRequirements
    : normalizedParsedEvidenceRequirements.filter((requirement) =>
      !plannerRequirementMandatesRenderedUiValidation(requirement.description))
  const parsedEvidenceRequirements = allowsRuntimeValidation
    ? interfaceScopedEvidenceRequirements
    : interfaceScopedEvidenceRequirements.filter((requirement) =>
      requirement.kind !== 'validation'
      && !plannerRequirementMandatesRuntimeValidation(requirement.description))
  const rawEvidenceRequirements = parsedEvidenceRequirements.length > 0
    ? parsedEvidenceRequirements
    : fallback.evidenceRequirements ?? []
  const evidenceRequirements = prospectiveDocumentPhase
    // Do not let the semantic planner invent audit ceremony for a prospective
    // document, but preserve a stronger evidence contract supplied upstream.
    ? fallback.evidenceRequirements ?? []
    : plannerMayProposeArtifact
      ? rawEvidenceRequirements
      : rawEvidenceRequirements
          .filter((requirement) => requirement.kind !== 'artifact')
          .map((requirement) => ({
            ...requirement,
            requiresArtifactEvidenceMap: false,
            requiresArtifactSelfReview: false,
          }))
  const parsedArtifactSections = normalizeArtifactSections(
    parsed.artifactSections ?? parsed.sections ?? parsed.documentSections,
    requiredArtifacts[0]?.path ?? fallback.requiredArtifacts?.[0]?.path,
  )
  const artifactSections = prospectiveDocumentPhase
    ? filterArtifactSectionsToRequiredDocuments(
        filterNoTouchArtifactSections(fallback.artifactSections ?? [], fallback),
        requiredArtifacts,
      )
    : plannerMayProposeArtifact
      ? (() => {
          const scopedParsedSections = filterArtifactSectionsToRequiredDocuments(
            filterNoTouchArtifactSections(parsedArtifactSections, fallback),
            requiredArtifacts,
          )
          if (scopedParsedSections.length > 0) return scopedParsedSections
          return filterArtifactSectionsToRequiredDocuments(
            filterNoTouchArtifactSections(fallback.artifactSections ?? [], fallback),
            requiredArtifacts,
          )
        })()
      : []
  const constraints = mergeStringLists(
    userBoundaryConstraintsFromFallback(fallback),
    [
      DEFAULT_RUN_CONTINUATION_POLICY,
      ...(prospectiveDocumentPhase ? [] : [DEFAULT_BROAD_COVERAGE_POLICY]),
      ...normalizeStringList(parsed.constraints, 6, MAX_LIST_ITEM_CHARS),
    ],
    6,
  )
  const outOfScope = mergeStringLists(
    userBoundaryOutOfScopeFromFallback(fallback),
    normalizeStringList(parsed.outOfScope, 4, MAX_LIST_ITEM_CHARS),
    4,
  )
  const requestedTerminalCommand = fallback.executionIntent?.requestedTerminalCommand
  const requestedProcessStart = fallback.executionIntent?.requestedProcessStart
  const constrainedProcessStart = fallback.executionIntent?.constrainedProcessStart
  const normalizedPlannerIntent = normalizeExecutionIntent(
    parsed.executionIntent,
    requestText,
    options.availableToolNames,
  )
  const plannerIntent = normalizedPlannerIntent
    ? {
        ...normalizedPlannerIntent,
        ...(fallback.executionIntent?.capabilityPolicy === 'closed'
          ? {
              kind: fallback.executionIntent.kind,
              workspaceMutation: fallback.executionIntent.workspaceMutation,
              capabilities: [...fallback.executionIntent.capabilities],
              capabilityPolicy: 'closed' as const,
            }
          : {}),
        ...(fallback.executionIntent?.allowedTools
          ? { allowedTools: [...fallback.executionIntent.allowedTools] }
          : {}),
        ...(fallback.executionIntent?.toolSequence
          ? { toolSequence: [...fallback.executionIntent.toolSequence] }
          : {}),
        ...(fallback.executionIntent?.retryPolicy
          ? { retryPolicy: fallback.executionIntent.retryPolicy }
          : {}),
        ...(fallback.executionIntent?.authorizedWriteTargets
          ? { authorizedWriteTargets: [...fallback.executionIntent.authorizedWriteTargets] }
          : {}),
        ...(fallback.executionIntent?.protectedWriteTargets
          ? { protectedWriteTargets: [...fallback.executionIntent.protectedWriteTargets] }
          : {}),
      }
    : undefined
  const executionIntent = requestedTerminalCommand || requestedProcessStart
    ? {
        kind: 'operational-action' as const,
        workspaceMutation: 'forbidden' as const,
        capabilities: [requestedProcessStart ? 'process' as const : 'terminal' as const],
        ...(fallback.executionIntent?.allowedTools
          ? { allowedTools: [...fallback.executionIntent.allowedTools] }
          : {}),
        ...(fallback.executionIntent?.toolSequence
          ? { toolSequence: [...fallback.executionIntent.toolSequence] }
          : {}),
        ...(fallback.executionIntent?.retryPolicy
          ? { retryPolicy: fallback.executionIntent.retryPolicy }
          : {}),
        ...(fallback.executionIntent?.authorizedWriteTargets
          ? { authorizedWriteTargets: [...fallback.executionIntent.authorizedWriteTargets] }
          : {}),
        ...(fallback.executionIntent?.protectedWriteTargets
          ? { protectedWriteTargets: [...fallback.executionIntent.protectedWriteTargets] }
          : {}),
        ...(requestedTerminalCommand
          ? {
              requestedTerminalCommand: {
                executable: requestedTerminalCommand.executable,
                args: [...requestedTerminalCommand.args],
              },
            }
          : {}),
        ...(requestedProcessStart
          ? {
              requestedProcessStart: {
                ...requestedProcessStart,
                args: [...requestedProcessStart.args],
                ...(typeof requestedProcessStart.network === 'object'
                  ? {
                      network: {
                        ...requestedProcessStart.network,
                        ...(requestedProcessStart.network.ports
                          ? { ports: [...requestedProcessStart.network.ports] }
                          : {}),
                      },
                    }
                  : {}),
              },
            }
          : {}),
      }
    : constrainedProcessStart
      ? {
          ...(plannerIntent ?? fallback.executionIntent!),
          capabilities: applyExplicitCapabilityBoundaries([
            ...new Set([
              ...(plannerIntent?.capabilities ?? []),
              ...(fallback.executionIntent?.capabilities ?? []),
              'process' as const,
            ]),
          ], requestText),
          constrainedProcessStart: {
            ...constrainedProcessStart,
            args: [...constrainedProcessStart.args],
            ...(typeof constrainedProcessStart.network === 'object'
              ? {
                  network: {
                    ...constrainedProcessStart.network,
                    ...(constrainedProcessStart.network.ports
                      ? { ports: [...constrainedProcessStart.network.ports] }
                      : {}),
                  },
                }
              : {}),
          },
        }
      : plannerIntent ?? fallback.executionIntent

  return reconcileRunContractExecutionCapabilities(reconcileRunContractWriteAuthority(addRenderedUiValidationToContract({
    summary,
    acceptanceCriteria: finalAcceptanceCriteria.map((text, index) => ({
      id: `AC${index + 1}`,
      text,
    })),
    constraints,
    outOfScope,
    ...(requiredArtifacts.length > 0 ? { requiredArtifacts } : {}),
    ...(evidenceRequirements.length > 0 ? { evidenceRequirements } : {}),
    ...(artifactSections.length > 0 ? { artifactSections } : {}),
    ...(executionIntent ? { executionIntent } : {}),
    source: 'planner',
  }, requestText, 7)))
}

export function formatRunContractForPrompt(contract: AgentRunContract | undefined): string | null {
  if (!contract) return null
  const lines = [
    '[Durable run contract]',
    `Goal: ${contract.summary}`,
    'Acceptance criteria:',
    ...contract.acceptanceCriteria.map((criterion) => `- ${criterion.id}: ${criterion.text}`),
  ]
  if (contract.constraints.length > 0) {
    lines.push('Constraints:', ...contract.constraints.map((entry) => `- ${entry}`))
  }
  if (contract.outOfScope.length > 0) {
    lines.push('Out of scope:', ...contract.outOfScope.map((entry) => `- ${entry}`))
  }
  if (contract.executionIntent) {
    lines.push(
      'Execution intent:',
      `- kind=${contract.executionIntent.kind}; workspaceMutation=${contract.executionIntent.workspaceMutation}; capabilities=${contract.executionIntent.capabilities.join(',') || 'none'}`,
    )
    if (contract.executionIntent.capabilityPolicy === 'closed') {
      lines.push('- capabilities are a closed semantic upper bound; do not add tool surfaces or validation obligations outside them')
    }
    if (contract.executionIntent.allowedTools) {
      lines.push(
        `- exact allowed tools=${contract.executionIntent.allowedTools.join(',') || 'none'}; do not call convenience, discovery, follow-up, or substitute tools outside this set`,
      )
    }
    const representableToolSequence = normalizeRepresentableToolSequence(
      contract.executionIntent.toolSequence ?? [],
    )
    if (representableToolSequence.length > 0) {
      lines.push(
        `- required tool order=${representableToolSequence.join(' -> ')}; call only the next tool in this sequence`,
      )
    }
    if (contract.executionIntent.retryPolicy === 'forbidden') {
      lines.push(
        '- retry policy=forbidden; after any required semantic action fails or is blocked, call no more tools and report the incomplete workflow honestly',
      )
    }
    if (contract.executionIntent.authorizedWriteTargets?.length) {
      lines.push(
        `- explicitly authorized write targets=${contract.executionIntent.authorizedWriteTargets.join(',')}; do not reinterpret these same targets as direct no-touch boundaries`,
      )
    }
    if (contract.executionIntent.protectedWriteTargets?.length) {
      lines.push(
        `- explicitly protected write targets=${contract.executionIntent.protectedWriteTargets.join(',')}; never mutate these targets`,
      )
    }
    if (contract.executionIntent.requestedTerminalCommand) {
      lines.push(
        `- exact terminal argv=${JSON.stringify([
          contract.executionIntent.requestedTerminalCommand.executable,
          ...contract.executionIntent.requestedTerminalCommand.args,
        ])}; execute it once without adding or removing arguments`,
      )
    }
    if (contract.executionIntent.requestedProcessStart) {
      lines.push(
        `- exact managed process start=${JSON.stringify(contract.executionIntent.requestedProcessStart)}; use process.start once without changing the command or lifecycle options`,
      )
    }
    if (contract.executionIntent.constrainedProcessStart) {
      lines.push(
        `- exact process.start constraint=${JSON.stringify(contract.executionIntent.constrainedProcessStart)}; if process.start is called during this workflow, do not change its executable, arguments, cwd, lifecycle, or network options`,
      )
    }
    if (
      contract.executionIntent.kind === 'operational-action'
      && contract.executionIntent.workspaceMutation === 'forbidden'
    ) {
      lines.push(
        '- Fresh observation boundary: report requested current state only from tool results produced after the latest user message. Do not fill missing fields from earlier-turn files, memory, or tool results unless the user explicitly asks for historical comparison; instead state that the field was not observed.',
      )
    }
  }
  if (contract.requiredArtifacts?.length) {
    lines.push(
      'Required artifacts:',
      ...contract.requiredArtifacts.map((artifact) => {
        const description = artifact.description ? ` — ${artifact.description}` : ''
        return `- ${artifact.path} (${artifact.kind})${description}`
      }),
    )
  }
  if (contract.evidenceRequirements?.length) {
    lines.push(
      'Evidence requirements:',
      ...contract.evidenceRequirements.map((requirement) => {
        const details = [
          `${requirement.kind}: ${requirement.description}`,
          typeof requirement.minSourceObservations === 'number'
            ? `minSourceObservations=${requirement.minSourceObservations}`
            : '',
          typeof requirement.minSourceFiles === 'number'
            ? `minSourceFiles=${requirement.minSourceFiles}`
            : '',
          typeof requirement.minSourceScopes === 'number'
            ? `minSourceScopes=${requirement.minSourceScopes}`
            : '',
          requirement.sourceToolNames?.length
            ? `sourceToolNames=${requirement.sourceToolNames.join(',')}`
            : '',
          requirement.requiresArtifactEvidenceMap ? 'requiresArtifactEvidenceMap=true' : '',
          requirement.requiresArtifactSelfReview ? 'requiresArtifactSelfReview=true' : '',
          requirement.requiresSearch ? 'requiresSearch=true' : '',
        ].filter(Boolean)
        return `- ${details.join('; ')}`
      }),
    )
  }
  if (contract.artifactSections?.length) {
    lines.push(
      'Required artifact sections:',
      ...contract.artifactSections.map((section) => {
        const artifactPath = section.artifactPath ? `; artifact=${section.artifactPath}` : ''
        const description = section.description ? ` — ${section.description}` : ''
        const required = section.required === false ? '; optional' : '; required'
        return `- ${section.id}: ${section.title}${artifactPath}${required}${description}`
      }),
    )
  }
  lines.push(
    'Long-running policy:',
    '- Do not treat budget exhaustion, partial exploration, or a short overview as completion.',
    '- For large code, analysis, writing, or artifact tasks, update durable files or evidence notes incrementally before more discovery.',
    '- For broad artifact work, keep a working draft or evidence ledger current; label partial coverage honestly until the contract is satisfied.',
    '- Before a final answer, compare the result against every acceptance criterion.',
    '- Keep acceptance-criterion verdicts as internal diagnostics; the user-facing final answer should state the outcome naturally without exposing run-contract protocol lines.',
    '- For a satisfied read-only observation criterion, use `CRITERION <id>: MET EVIDENCE <tool-call-id,...>` with exact reference ids shown in the structured evidence ledger. Never invent or shorten an evidence id. Use `CRITERION <id>: UNMET` when the criterion is not satisfied.',
    '- Only claim completion when tool evidence (validation run or artifact read-back) supports it; completion claims without verified evidence will be rejected.',
    '- If any criterion is still unmet, answer with INCOMPLETE: and name the next concrete action; do not present the task as finished.',
  )
  return lines.join('\n')
}

const BUDGET_EXHAUSTED_CHECKPOINT_NOTICE =
  'A resumable checkpoint has been preserved so the run can continue instead of pretending the task is finished.'
const BUDGET_EXHAUSTED_NEXT_STEP =
  'Next step: resume this session and continue from the latest checkpoint, updating any partial artifacts before final synthesis.'
const BUDGET_EXHAUSTED_HEADER_PATTERN =
  /^INCOMPLETE: [^\r\n]+ reached its iteration budget \([^()\r\n]+\)\./

export function buildBudgetExhaustedMessage(input: {
  mode: string
  iterationBudget: number
  contract?: AgentRunContract
}): string {
  const criteria = input.contract?.acceptanceCriteria
    .map((criterion) => criterion.id)
    .join(', ')
  const contractLine = criteria
    ? `Acceptance criteria still need explicit closure: ${criteria}.`
    : ''
  return [
    `INCOMPLETE: ${input.mode} reached its iteration budget (${input.iterationBudget}).`,
    BUDGET_EXHAUSTED_CHECKPOINT_NOTICE,
    contractLine,
    BUDGET_EXHAUSTED_NEXT_STEP,
  ].filter(Boolean).join(' ')
}

/**
 * Identifies the synthetic incomplete result emitted by the agent engines when
 * their iteration budget is exhausted. Keep this beside the message builder so
 * downstream consumers do not duplicate a user-visible sentence as a sentinel.
 */
export function isBudgetExhaustedMessage(content: string): boolean {
  const normalized = content.trim()
  return BUDGET_EXHAUSTED_HEADER_PATTERN.test(normalized)
    && normalized.includes(BUDGET_EXHAUSTED_CHECKPOINT_NOTICE)
    && normalized.endsWith(BUDGET_EXHAUSTED_NEXT_STEP)
}

export function resolveMaxContinuationCycles(value?: number, defaultValue = 0): number {
  const configured = value ?? Number(process.env.SEPILOTD_MAX_CONTINUATION_CYCLES ?? defaultValue)
  if (!Number.isFinite(configured) || configured <= 0) {
    return 0
  }
  return Math.min(12, Math.floor(configured))
}

export function buildContinuationPrompt(input: {
  mode: string
  cycle: number
  maxCycles: number
  contract?: AgentRunContract
}): string {
  const criteria = input.contract?.acceptanceCriteria
    .map((criterion) => `${criterion.id}: ${criterion.text}`)
    .join('\n')
  return [
    '[Continuation supervisor]',
    `${input.mode} reached an execution budget and is continuing automatically (${input.cycle}/${input.maxCycles}).`,
    'Resume from the current evidence and artifacts. Do not restart broad discovery unless the contract requires it.',
    'Close only when every acceptance criterion is satisfied; otherwise continue making concrete progress.',
    criteria ? `Acceptance criteria:\n${criteria}` : '',
  ].filter(Boolean).join('\n')
}
