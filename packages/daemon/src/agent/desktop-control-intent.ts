import { isTerseVisibleWebSurfaceRequest } from '@sepilotd/api-client'

const DESKTOP_APP_TARGET_RE =
  /(?:메모장|계산기|그림판|파일\s*탐색기|탐색기|브라우저|크롬|엣지|워드|엑셀|파워포인트|데스크톱|응용\s*프로그램|프로그램|앱|창|\bnotepad\b|\bcalculator\b|\bpaint\b|\bfile\s+explorer\b|\bbrowser\b|\bchrome\b|\bedge\b|\bword\b|\bexcel\b|\bpowerpoint\b|\bdesktop\b|\bapplication\b|\bapp\b|\bwindow\b)/iu
const DESKTOP_CONTROL_ACTION_KO_RE =
  /(?:열고|열어|실행해|실행하고|띄워|띄우고|켜고|켜줘|클릭해|눌러|입력해|타이핑해|작성해|써줘|닫아|전환해|선택해)/iu
const DESKTOP_CONTROL_COMMAND_END_KO_RE =
  /(?:줘|주세요|해|하세요|하자|해라|작성해|써줘)[.!?\s]*$/iu
const DESKTOP_CONTROL_ACTION_EN_RE =
  /^\s*(?:please\s+)?(?:open|launch|start|focus|click|type|write\s+in|close|switch\s+to|select)\b|\bplease\s+(?:open|launch|start|focus|click|type|write\s+in|close|switch\s+to|select)\b/iu
const DESKTOP_CONTROL_NEGATION_RE =
  /(?:열지\s*마|실행하지\s*마|클릭하지\s*마|입력하지\s*마|닫지\s*마)|\b(?:do\s+not|don't|never)\s+(?:open|launch|start|focus|click|type|close|select)\b/iu
const VISIBLE_URL_TARGET_RE =
  /https?:\/\/|(?:브라우저|크롬|엣지|웹\s*브라우저|웹사이트|사이트|페이지|링크|지도|검색\s*결과)|\b(?:browser|chrome|edge|website|site|page|link|url|map|search\s+results?)\b/iu
const VISIBLE_URL_ACTION_KO_RE =
  /(?:열어|열고|띄워|띄우고|보여\s*줘|보여\s*주세요|표시해|표시해\s*줘|접속해|접속해\s*줘)/iu
const VISIBLE_URL_COMMAND_END_KO_RE =
  /(?:줘|주세요|해|하세요|하자|해라|띄워|열어|보여\s*줘|보여\s*주세요)[.!?\s]*$/iu
const VISIBLE_URL_ACTION_EN_RE =
  /^\s*(?:please\s+)?(?:open|show|display|launch|visit|navigate\s+to)\b|\bplease\s+(?:open|show|display|launch|visit|navigate\s+to)\b/iu
const VISIBLE_URL_NEGATION_RE =
  /(?:브라우저[^\r\n]{0,40})?(?:열지\s*마|띄우지\s*마|보여\s*주지\s*마|표시하지\s*마|접속하지\s*마)|\b(?:do\s+not|don't|never)\s+(?:open|show|display|launch|visit|navigate)\b/iu
const HEADLESS_RENDER_VALIDATION_RE =
  /(?:렌더(?:링)?|레이아웃|DOM|뷰포트|스크린샷).{0,48}(?:검증|확인|검사|테스트)|(?:검증|확인|검사|테스트).{0,48}(?:렌더(?:링)?|레이아웃|DOM|뷰포트|스크린샷)|\b(?:validate|verify|inspect|test|audit)\b.{0,48}\b(?:rendered|layout|DOM|viewport|screenshot|screen)\b|\b(?:rendered|layout|DOM|viewport|screenshot|screen)\b.{0,48}\b(?:validate|verify|inspect|test|audit)\b/iu
const EXPLICIT_HEADLESS_BROWSER_TOOL_RE =
  /\bbrowser\.(?:navigate|extract|screenshot|click|evaluate)\b/iu

function withoutQuotedExamples(input: string): string {
  return input
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`[^`\r\n]*`/g, ' ')
    .replace(/["“”][^"“”\r\n]*["“”]/g, ' ')
}

/**
 * Detect explicit control of an already-installed desktop application. It is
 * deliberately narrower than generic "app" requests so authoring or
 * explaining desktop software still goes through the normal agent router.
 */
export function isExplicitDesktopGuiControlRequest(input: string): boolean {
  const actionable = withoutQuotedExamples(input).trim()
  if (!actionable || DESKTOP_CONTROL_NEGATION_RE.test(actionable)) return false
  if (!DESKTOP_APP_TARGET_RE.test(actionable)) return false
  return (
    DESKTOP_CONTROL_ACTION_KO_RE.test(actionable)
    && DESKTOP_CONTROL_COMMAND_END_KO_RE.test(actionable)
  ) || DESKTOP_CONTROL_ACTION_EN_RE.test(actionable)
}

/**
 * Detect a request for a user-visible browser surface. Besides explicit
 * commands, a bounded map-result fragment is accepted as a conversational
 * UI request because the policy still asks before changing desktop focus.
 * Search/research and technical render validation stay on headless tools.
 */
export function isVisibleUrlOpenRequest(input: string): boolean {
  const actionable = withoutQuotedExamples(input).trim()
  if (!actionable || VISIBLE_URL_NEGATION_RE.test(actionable)) return false
  if (
    EXPLICIT_HEADLESS_BROWSER_TOOL_RE.test(actionable)
    || HEADLESS_RENDER_VALIDATION_RE.test(actionable)
  ) return false
  if (isTerseVisibleWebSurfaceRequest(actionable)) return true
  if (!VISIBLE_URL_TARGET_RE.test(actionable)) return false
  return (
    VISIBLE_URL_ACTION_KO_RE.test(actionable)
    && VISIBLE_URL_COMMAND_END_KO_RE.test(actionable)
  ) || VISIBLE_URL_ACTION_EN_RE.test(actionable)
}
