export const WEB_SEARCH_TOOL_NAME = 'web.search'
export const MARKET_QUOTE_TOOL_NAME = 'market.quote'

const EXPLICIT_WEB_SEARCH_KO_RE =
  /(?:웹|인터넷|온라인|web\.search)(?:에서|으로|을|를)?[\s\S]{0,48}?(?:검색|찾아|조회|확인)(?:해|해서|하면|해도|해줘|해주세요|봐|보면|해보|하자|하면\s*되)/iu
const EXPLICIT_WEB_SEARCH_EN_RE =
  /(?:\b(?:search|browse|look(?:\s+(?:this|it))?\s+up|check)\b[\s\S]{0,48}\b(?:the\s+)?(?:web|internet|online)\b|\b(?:web|internet|online)\b[\s\S]{0,48}\b(?:search|browse|look(?:\s+(?:this|it))?\s+up|check)\b|\bweb\.search\b)/iu
const WEB_SEARCH_NEGATION_RE =
  /(?:검색|조회|확인)(?:하지\s*마|하지마|하지\s*말|하지\s*않)|\b(?:do\s+not|don't|never)\b[\s\S]{0,32}\b(?:search|browse|look\s+up|check)\b/iu

const MARKET_SUBJECT_RE =
  /(?:주가|종가|현재가|시세|주식|증권|코스피|코스닥|나스닥|다우|S&P\s*500|가격\s*추이|수익률|\bstock(?:s|\s+price)?\b|\bshare\s+price\b|\bclosing\s+price\b|\bmarket\s+quote\b|\bticker\b|\bKOSPI\b|\bKOSDAQ\b|\bNASDAQ\b)/iu
const CURRENT_MARKET_RE =
  /(?:오늘|현재|지금|실시간|이번\s*주|금일|장\s*마감|\btoday(?:'s)?\b|\bcurrent(?:ly)?\b|\bright\s+now\b|\blive\b|\blatest\b)/iu
const HISTORICAL_MARKET_RE =
  /(?:어제|지난|최근|과거|추이|변동|흐름|기간|\bpast\b|\blast\s+(?:week|month|year|\d+)\b|\brecent\b|\bhistory|historical|trend|performance\b)/iu
const FORECAST_MARKET_RE =
  /(?:전망|예측|향후|앞으로|내일|다음\s*주|\bforecast|prediction|predict|outlook|tomorrow|next\s+week\b)/iu
const MARKET_NEGATION_RE =
  /(?:조회|검색|확인|알려)(?:하지\s*마|하지마|하지\s*말|하지\s*않)|\b(?:do\s+not|don't|never)\b[\s\S]{0,32}\b(?:quote|search|check|look\s+up)\b/iu

function withoutQuotedExamples(input: string): string {
  return input
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`[^`\r\n]*`/g, ' ')
    .replace(/["“”][^"“”\r\n]*["“”]/g, ' ')
}

export function isExplicitWebSearchRequest(input: string): boolean {
  const actionable = withoutQuotedExamples(input)
  if (WEB_SEARCH_NEGATION_RE.test(actionable)) return false
  return EXPLICIT_WEB_SEARCH_KO_RE.test(actionable)
    || EXPLICIT_WEB_SEARCH_EN_RE.test(actionable)
}

/**
 * Resolve the smallest live-information tool surface that can satisfy an
 * explicit request. This intentionally does not match generic factual
 * questions: instant mode remains tool-free unless the user names web access
 * or asks for time-sensitive market data.
 */
export function resolveExplicitLiveInformationToolNames(
  input: string,
): readonly string[] | null {
  const actionable = withoutQuotedExamples(input)
  const explicitWebSearch = isExplicitWebSearchRequest(actionable)
  const marketRequest = MARKET_SUBJECT_RE.test(actionable)
    && !MARKET_NEGATION_RE.test(actionable)
    && (
      CURRENT_MARKET_RE.test(actionable)
      || HISTORICAL_MARKET_RE.test(actionable)
      || FORECAST_MARKET_RE.test(actionable)
    )

  if (!marketRequest) {
    return explicitWebSearch ? [WEB_SEARCH_TOOL_NAME] : null
  }

  const needsCurrentQuote = CURRENT_MARKET_RE.test(actionable)
  const needsWebContext = explicitWebSearch
    || HISTORICAL_MARKET_RE.test(actionable)
    || FORECAST_MARKET_RE.test(actionable)

  if (needsCurrentQuote && needsWebContext) {
    return [MARKET_QUOTE_TOOL_NAME, WEB_SEARCH_TOOL_NAME]
  }
  if (needsCurrentQuote) {
    return [MARKET_QUOTE_TOOL_NAME]
  }
  return [WEB_SEARCH_TOOL_NAME]
}
