export const SCHEDULE_CREATE_TOOL_NAME = 'schedule_create'
export const SCHEDULE_LIST_TOOL_NAME = 'schedule_list'
export const SCHEDULE_GET_TOOL_NAME = 'schedule_get'
export const SCHEDULE_RUNS_TOOL_NAME = 'schedule_runs'
export const SCHEDULE_CANCEL_TOOL_NAME = 'schedule_cancel'
export const SCHEDULE_PAUSE_TOOL_NAME = 'schedule_pause'
export const SCHEDULE_RESUME_TOOL_NAME = 'schedule_resume'
export const SCHEDULE_RUN_NOW_TOOL_NAME = 'schedule_run_now'
export const SCHEDULE_UPDATE_TOOL_NAME = 'schedule_update'

export type ScheduleManagementIntent = 'cancel' | 'pause' | 'resume' | 'run_now' | 'update'
export type ScheduleInspectionIntent = 'detail' | 'runs'

const FUTURE_TIME_ANCHOR_PATTERNS = [
  /(?:이따|나중에|오늘|내일|모레|글피|매일|매주|매달|매월|매시간)/u,
  /(?:(?:오전|오후|밤|새벽)\s*)?\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?/u,
  /\d+\s*(?:초|분|시간|일)\s*(?:후|뒤|마다|간격(?:으로)?|단위(?:로)?)/u,
  /\b(?:today|tomorrow|tonight|later|next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b/iu,
  /\b(?:in\s+\d+\s*(?:seconds?|minutes?|hours?|days?)|at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?|every\s+\S+)\b/iu,
  /@(?:every\s+\d+\s*[smhd]|daily|hourly|weekly|monthly|yearly)\b/iu,
]

const EXPLICIT_SCHEDULE_ACTION_PATTERNS = [
  /(?:알려\s*(?:줘|주세요|주십시오)|알려\s*줄\s*수\s*있(?:어|나요|습니까)|알림(?:을|를)?\s*(?:설정|등록|예약)(?:해)?\s*(?:줘|주세요|주십시오)?|리마인드(?:해)?\s*(?:줘|주세요|주십시오)|예약(?:해)?\s*(?:줘|주세요|주십시오))/u,
  /\b(?:remind|notify)\s+(?:me|us)\b/iu,
  /\b(?:schedule|set\s+up|create)\s+(?:a\s+)?(?:reminder|scheduled\s+(?:task|job)|recurring\s+(?:task|job))\b/iu,
]

const NEGATED_SCHEDULE_ACTION_PATTERNS = [
  /(?:알려|알림|리마인드|예약)(?:하|해|주)?지\s*(?:마|말)|(?:안|못)\s*(?:알려|예약)/u,
  /\b(?:do\s+not|don't|never)\s+(?:remind|notify|schedule|set)\b/iu,
]

const SCHEDULE_META_MENTION_PATTERNS = [
  /(?:알려\s*(?:줘|주세요)|알림\s*(?:설정|등록|예약)|예약(?:해)?\s*(?:줘|주세요))[.!?。！？]?\s*(?:라고|하고)\s*(?:했|말|입력|요청|썼|보냈|하니|했더니)/u,
  /\b(?:remind|notify|schedule)\b[^.!?\n]{0,120}[.!?]\s*(?:is|was)\s+what\s+(?:I|we|the user)\s+(?:said|typed|asked)\b/iu,
]

const EXPLICIT_SCHEDULE_LIST_PATTERNS = [
  /(?:현재|지금)?\s*(?:등록|예약|설정)(?:되어|돼|된|해\s*둔|한)\s*(?:예약\s*)?(?:작업|일정|알림|리마인더)(?:들)?(?:은|는|이|가|을|를)?\s*(?:뭐|무엇|어떤|있|목록|리스트|보여|알려|확인|조회|[?？]|$)/u,
  /(?:예약|스케줄)(?:된|해\s*둔|한)?\s*(?:작업|일정|알림|리마인더)?\s*(?:목록|리스트)(?:을|를|은|는|이|가)?\s*(?:보여|알려|확인|조회|있|[?？]|$)/u,
  /(?:내|제가|우리가)?\s*(?:예약|등록|설정)(?:한|된)?\s*(?:알림|리마인더|작업)(?:들)?(?:은|는|이|가)?\s*(?:뭐|무엇|어떤|몇\s*개|있)/u,
  /\b(?:list|show|check|what(?:'s|\s+is|\s+are)?|which|do\s+(?:i|we)\s+have|are\s+there)\b[^.!?\n]{0,80}\b(?:scheduled\s+(?:tasks?|jobs?)|schedules?|reminders?)\b/iu,
  /\b(?:scheduled\s+(?:tasks?|jobs?)|schedules?|reminders?)\b[^.!?\n]{0,80}\b(?:list|pending|registered|active|do\s+(?:i|we)\s+have|are\s+there)\b/iu,
]

const SCHEDULE_HISTORY_LIST_PATTERNS = [
  /(?:모든|전체|전부|완료(?:되거나|된|한)?|취소(?:되거나|된|한)?|실패(?:하거나|한|했던)?|지난|과거|이전)[^.!?\n]{0,40}(?:예약|스케줄|작업|일정|알림|리마인더)/u,
  /(?:예약|스케줄|작업|일정|알림|리마인더)\s*(?:전체|전부|이력|기록|내역|히스토리)/u,
  /(?:사라진|없어진|이미\s*실행(?:된|한)?|끝난)\s*(?:예약|스케줄|작업|일정|알림|리마인더)/u,
  /\b(?:all|completed|cancelled|canceled|failed|past|previous)\s+(?:scheduled\s+)?(?:tasks?|jobs?|schedules?|reminders?)\b/iu,
  /\b(?:schedule|scheduled\s+task|scheduled\s+job|reminder)\s+(?:history|records?)\b/iu,
]

const SCHEDULE_HISTORY_QUERY_PATTERNS = [
  /(?:보여|알려|확인|조회|목록|리스트|내역|기록|몇\s*개|있(?:어|나요|습니까)?|[?？])/u,
  /\b(?:list|show|check|history|records?|what|which|how\s+many|are\s+there|do\s+(?:i|we)\s+have)\b/iu,
]

const SCHEDULE_MANAGEMENT_PATTERNS: ReadonlyArray<{
  intent: ScheduleManagementIntent
  patterns: readonly RegExp[]
}> = [
  {
    intent: 'update',
    patterns: [
      /(?:(?:예약|스케줄)(?:을|를|된|한)?\s*(?:작업|일정|알림|리마인더)?|(?:알림|리마인더)(?:을|를)?)[^.!?\n]{0,100}(?:변경|수정|바꿔|옮겨|재예약|조정)(?:해)?\s*(?:(?:줘|주세요|주십시오)\s*)?(?=[.!?]?\s*$)/u,
      /(?:변경|수정|바꿔|옮겨|재예약|조정)(?:해)?\s*(?:줘|주세요|주십시오)[^.!?\n]{0,60}(?:(?:예약|스케줄)(?:을|를|된|한)?\s*(?:작업|일정|알림|리마인더)?|(?:알림|리마인더)(?:을|를)?)/u,
      /^\s*(?:please\s+|can\s+you\s+)?(?:change|edit|update|reschedule|move)\b[^.!?\n]{0,100}\b(?:scheduled\s+(?:task|job)|schedule|reminder|it|that)\b/iu,
      /\b(?:scheduled\s+(?:task|job)|schedule|reminder)\b[^.!?\n]{0,100}\b(?:change|edit|update|reschedule|move)\b/iu,
    ],
  },
  {
    intent: 'run_now',
    patterns: [
      /(?:(?:예약|스케줄)(?:을|를|된|한)?\s*(?:작업|일정|알림|리마인더)?|(?:알림|리마인더)(?:을|를)?)[^.!?\n]{0,80}(?:지금|바로)\s*(?:실행|돌려|시작)(?:해)?\s*(?:(?:줘|주세요|주십시오)\s*)?(?=[.!?]?\s*$)/u,
      /(?:지금|바로)\s*(?:실행|돌려|시작)(?:해)?\s*(?:줘|주세요|주십시오)[^.!?\n]{0,50}(?:(?:예약|스케줄)(?:을|를|된|한)?\s*(?:작업|일정|알림|리마인더)?|(?:알림|리마인더)(?:을|를)?)/u,
      /^\s*(?:please\s+|can\s+you\s+)?(?:run|execute|start)\b[^.!?\n]{0,80}\b(?:scheduled\s+(?:task|job)|schedule|reminder|it|that)\b[^.!?\n]{0,30}\bnow\b/iu,
    ],
  },
  {
    intent: 'cancel',
    patterns: [
      /(?:(?:예약|스케줄)(?:을|를|된|한)?\s*(?:작업|일정|알림|리마인더)?|(?:알림|리마인더)(?:을|를)?)[^.!?\n]{0,80}(?:취소|삭제|지워|없애)(?:해)?\s*(?:(?:줘|주세요|주십시오)\s*)?(?=[.!?]?\s*$)/u,
      /(?:취소|삭제|지워|없애)(?:해)?\s*(?:줘|주세요|주십시오)[^.!?\n]{0,50}(?:(?:예약|스케줄)(?:을|를|된|한)?\s*(?:작업|일정|알림|리마인더)?|(?:알림|리마인더)(?:을|를)?)/u,
      /^\s*(?:please\s+|can\s+you\s+)?(?:cancel|delete|remove)\b[^.!?\n]{0,80}\b(?:scheduled\s+(?:task|job)|schedule|reminder|it|that)\b/iu,
    ],
  },
  {
    intent: 'pause',
    patterns: [
      /(?:(?:예약|스케줄)(?:을|를|된|한)?\s*(?:작업|일정|알림|리마인더)?|(?:알림|리마인더)(?:을|를)?)[^.!?\n]{0,80}(?:일시\s*정지|잠시\s*(?:멈춰|중지)|중단)(?:해)?\s*(?:(?:줘|주세요|주십시오)\s*)?(?=[.!?]?\s*$)/u,
      /(?:일시\s*정지|잠시\s*(?:멈춰|중지)|중단)(?:해)?\s*(?:줘|주세요|주십시오)[^.!?\n]{0,50}(?:(?:예약|스케줄)(?:을|를|된|한)?\s*(?:작업|일정|알림|리마인더)?|(?:알림|리마인더)(?:을|를)?)/u,
      /^\s*(?:please\s+|can\s+you\s+)?(?:pause|temporarily\s+stop)\b[^.!?\n]{0,80}\b(?:scheduled\s+(?:task|job)|schedule|reminder|it|that)\b/iu,
    ],
  },
  {
    intent: 'resume',
    patterns: [
      /(?:(?:예약|스케줄)(?:을|를|된|한)?\s*(?:작업|일정|알림|리마인더)?|(?:알림|리마인더)(?:을|를)?)[^.!?\n]{0,80}(?:재개|다시\s*(?:시작|켜|활성화))(?:해)?\s*(?:(?:줘|주세요|주십시오)\s*)?(?=[.!?]?\s*$)/u,
      /(?:재개|다시\s*(?:시작|켜|활성화))(?:해)?\s*(?:줘|주세요|주십시오)[^.!?\n]{0,50}(?:(?:예약|스케줄)(?:을|를|된|한)?\s*(?:작업|일정|알림|리마인더)?|(?:알림|리마인더)(?:을|를)?)/u,
      /^\s*(?:please\s+|can\s+you\s+)?(?:resume|reactivate)\b[^.!?\n]{0,80}\b(?:scheduled\s+(?:task|job)|schedule|reminder|it|that)\b/iu,
    ],
  },
]

const NEGATED_SCHEDULE_MANAGEMENT_PATTERNS = [
  /(?:취소|삭제|지우|없애|일시\s*정지|중단|재개|실행|변경|수정|바꾸|옮기|재예약|조정)(?:하|해)?지\s*(?:마|말)/u,
  /\b(?:do\s+not|don't|never)\s+(?:cancel|delete|remove|pause|resume|run|execute|change|edit|update|reschedule|move)\b/iu,
]

const SCHEDULE_MANAGEMENT_EXPLANATION_PATTERNS = [
  /(?:기능|방법|사용법|설명|뜻|의미|가능(?:해|한|합니까)|어떻게\s*(?:해|하|동작))/u,
  /\b(?:explain|how\s+(?:do|does|to)|what\s+does|is\s+it\s+possible)\b/iu,
]

const SCHEDULE_INSPECTION_PATTERNS: ReadonlyArray<{
  intent: ScheduleInspectionIntent
  patterns: readonly RegExp[]
}> = [
  {
    intent: 'runs',
    patterns: [
      /(?:(?:그|이|저|해당|아까|방금(?:\s*만든)?)\s*)?(?:예약|스케줄|알림|리마인더)(?:\s*작업)?(?:은|는|이|가|을|를)?[^.!?\n]{0,80}(?:실행\s*(?:이력|기록|내역|결과)|최근\s*(?:실행|동작)|마지막\s*(?:실행|동작)|실패\s*(?:원인|이유)|왜\s*(?:실패|안\s*(?:됐|되었|돌았|실행)))[^.!?\n]{0,40}(?:보여|알려|확인|조회|뭐|무엇|왜|[?？]|$)/u,
      /왜[^.!?\n]{0,40}(?:(?:그|이|저|해당)\s*)?(?:예약|스케줄|알림|리마인더)(?:\s*작업)?(?:이|가|은|는)?[^.!?\n]{0,40}(?:실패|안\s*(?:됐|되었|돌았|실행))[.!?？]?\s*$/u,
      /\b(?:show|list|check|inspect)\b[^.!?\n]{0,100}\b(?:run\s+(?:history|records?|results?)|execution\s+(?:history|records?|results?)|recent\s+runs?|last\s+run)\b[^.!?\n]{0,80}\b(?:scheduled\s+(?:task|job)|schedule|reminder)\b/iu,
      /\b(?:scheduled\s+(?:task|job)|schedule|reminder)\b[^.!?\n]{0,100}\b(?:run\s+(?:history|records?|results?)|execution\s+(?:history|records?|results?)|recent\s+runs?|last\s+run)\b/iu,
      /\bwhy\b[^.!?\n]{0,80}\b(?:scheduled\s+(?:task|job)|schedule|reminder)\b[^.!?\n]{0,80}\b(?:fail(?:ed)?|(?:did(?:n['’]t|\s+not)\s+|not\s+)?(?:run|fire|execute))\b/iu,
      /\b(?:scheduled\s+(?:task|job)|schedule|reminder)\b[^.!?\n]{0,80}\bwhy\b[^.!?\n]{0,80}\b(?:fail(?:ed)?|(?:did(?:n['’]t|\s+not)\s+|not\s+)?(?:run|fire|execute))\b/iu,
    ],
  },
  {
    intent: 'detail',
    patterns: [
      /(?:(?:그|이|저|해당|아까|방금(?:\s*만든)?)\s*)?(?:예약|스케줄|알림|리마인더)(?:\s*작업)?(?:은|는|이|가|을|를)?[^.!?\n]{0,80}(?:상세|세부|설정(?:값|내용)?|등록\s*내용|지시\s*내용|무슨\s*내용|어떤\s*내용|어떻게\s*(?:설정|등록)|언제\s*(?:실행|동작))[^.!?\n]{0,40}(?:보여|알려|확인|조회|뭐|무엇|되어|돼|됐|있|[?？]|$)/u,
      /\b(?:show|give|check|inspect)\b[^.!?\n]{0,100}\b(?:details?|configuration|definition|instruction)\b[^.!?\n]{0,80}\b(?:scheduled\s+(?:task|job)|schedule|reminder)\b/iu,
      /\b(?:scheduled\s+(?:task|job)|schedule|reminder)\b[^.!?\n]{0,100}\b(?:details?|configuration|definition|instruction)\b/iu,
      /\bhow\s+is\b[^.!?\n]{0,80}\b(?:scheduled\s+(?:task|job)|schedule|reminder)\b[^.!?\n]{0,50}\b(?:configured|defined|set\s+up)\b/iu,
      /\bwhat\s+does\b[^.!?\n]{0,80}\b(?:scheduled\s+(?:task|job)|schedule|reminder)\b[^.!?\n]{0,40}\bdo\b/iu,
    ],
  },
]

const NEGATED_SCHEDULE_INSPECTION_PATTERNS = [
  /(?:상세|세부|설정|내용|정보|실행\s*(?:이력|기록|내역|결과)|최근\s*실행|마지막\s*실행|실패\s*(?:원인|이유))[^.!?\n]{0,30}(?:보여|알려|확인|조회|설명)(?:하|해)?지\s*(?:마|말)/u,
  /\b(?:do\s+not|don't|never)\s+(?:show|list|check|inspect|explain)\b[^.!?\n]{0,100}\b(?:schedule|scheduled\s+(?:task|job)|reminder|run\s+history)\b/iu,
]

const SCHEDULE_INSPECTION_EXPLANATION_PATTERNS = [
  /(?:예약|스케줄|알림|리마인더)[^.!?\n]{0,80}(?:상세|세부|실행\s*(?:이력|기록|내역)|실패\s*(?:원인|이유))[^.!?\n]{0,40}(?:기능|사용법|조회\s*방법|어떻게\s*조회)/u,
  /(?:기능|사용법|조회\s*방법)[^.!?\n]{0,40}(?:예약|스케줄|알림|리마인더)[^.!?\n]{0,80}(?:상세|세부|실행\s*(?:이력|기록|내역))/u,
  /\b(?:explain|describe|how\s+does|how\s+to)\b[^.!?\n]{0,100}\b(?:schedule|scheduled\s+(?:task|job)|reminder)\b[^.!?\n]{0,80}\b(?:feature|work|usage|api|tool)\b/iu,
]

function directiveText(input: string): string {
  return input
    .normalize('NFKC')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`[^`]*`/g, ' ')
    .replace(/"[^"\r\n]*"/g, ' ')
    .replace(/'[^'\r\n]*'/g, ' ')
    .replace(/“[^”]*”|‘[^’]*’|「[^」]*」|『[^』]*』|《[^》]*》/gu, ' ')
    .trim()
}

/**
 * Conservative intent gate for a request that must create future work now.
 * It requires both a future-time anchor and an imperative reminder/scheduling
 * action so ordinary time questions stay conversational.
 */
export function isExplicitScheduleCreateRequest(input: string): boolean {
  const directive = directiveText(input)
  if (!directive) return false
  if (NEGATED_SCHEDULE_ACTION_PATTERNS.some((pattern) => pattern.test(directive))) return false
  if (SCHEDULE_META_MENTION_PATTERNS.some((pattern) => pattern.test(directive))) return false
  return FUTURE_TIME_ANCHOR_PATTERNS.some((pattern) => pattern.test(directive))
    && EXPLICIT_SCHEDULE_ACTION_PATTERNS.some((pattern) => pattern.test(directive))
}

/**
 * Conservative read-only intent gate for inspecting the user's registered
 * schedules. Requiring both scheduling vocabulary and a list/existence query
 * keeps ordinary mentions of reminders conversational.
 */
export function isExplicitScheduleListRequest(input: string): boolean {
  const directive = directiveText(input)
  if (!directive) return false
  if (SCHEDULE_META_MENTION_PATTERNS.some((pattern) => pattern.test(directive))) return false
  return EXPLICIT_SCHEDULE_LIST_PATTERNS.some((pattern) => pattern.test(directive))
    || (
      SCHEDULE_HISTORY_LIST_PATTERNS.some((pattern) => pattern.test(directive))
      && SCHEDULE_HISTORY_QUERY_PATTERNS.some((pattern) => pattern.test(directive))
    )
}

/** Select the scheduler's narrow current view unless the user explicitly asks for history. */
export function scheduleListStatusForRequest(input: string): 'pending' | 'all' {
  const directive = directiveText(input)
  return SCHEDULE_HISTORY_LIST_PATTERNS.some((pattern) => pattern.test(directive))
    ? 'all'
    : 'pending'
}

/** Detect explicit management of an existing schedule without widening ordinary chat turns. */
export function resolveExplicitScheduleManagementIntent(
  input: string,
): ScheduleManagementIntent | null {
  const directive = directiveText(input)
  if (!directive) return null
  if (SCHEDULE_META_MENTION_PATTERNS.some((pattern) => pattern.test(directive))) return null
  if (NEGATED_SCHEDULE_MANAGEMENT_PATTERNS.some((pattern) => pattern.test(directive))) return null
  if (SCHEDULE_MANAGEMENT_EXPLANATION_PATTERNS.some((pattern) => pattern.test(directive))) return null
  for (const candidate of SCHEDULE_MANAGEMENT_PATTERNS) {
    if (candidate.patterns.some((pattern) => pattern.test(directive))) {
      return candidate.intent
    }
  }
  return null
}

/**
 * Detect an explicit read of one durable schedule definition or its persisted
 * execution history. These capabilities stay separate from schedule_list:
 * a job summary cannot prove that a particular run happened or why it failed.
 */
export function resolveExplicitScheduleInspectionIntent(
  input: string,
): ScheduleInspectionIntent | null {
  const directive = directiveText(input)
  if (!directive) return null
  if (SCHEDULE_META_MENTION_PATTERNS.some((pattern) => pattern.test(directive))) return null
  if (NEGATED_SCHEDULE_INSPECTION_PATTERNS.some((pattern) => pattern.test(directive))) return null
  if (SCHEDULE_INSPECTION_EXPLANATION_PATTERNS.some((pattern) => pattern.test(directive))) return null
  for (const candidate of SCHEDULE_INSPECTION_PATTERNS) {
    if (candidate.patterns.some((pattern) => pattern.test(directive))) {
      return candidate.intent
    }
  }
  return null
}
