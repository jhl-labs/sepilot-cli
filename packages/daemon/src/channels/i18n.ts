export type ChannelLang = 'en' | 'ko'

export type ChannelMessageKey =
  | 'approvalFeedback'
  | 'approvalDenied'
  | 'approvalRun'
  | 'approvalSessionAll'
  | 'approvalAlways'
  | 'approvalSession'
  | 'approvalOnce'
  | 'approvalNoLiveRequest'
  | 'error'

type Catalog = Record<ChannelMessageKey, Record<ChannelLang, string>>

const CATALOG: Catalog = {
  approvalFeedback: {
    en: 'Got it. I will flag the{tool} request as needing changes.',
    ko: '알겠습니다.{tool} 요청은 수정이 필요하다고 전달할게요.',
  },
  approvalDenied: {
    en: 'Got it. I will not run the{tool} request.',
    ko: '알겠습니다.{tool} 요청은 진행하지 않을게요.',
  },
  approvalRun: {
    en: 'Got it. I will auto-approve the follow-up tool requests for this task and keep going.',
    ko: '알겠습니다. 이번 작업이 끝날 때까지 이어지는 도구 요청은 자동으로 승인하고 계속 진행할게요.',
  },
  approvalSessionAll: {
    en: 'Got it. I will auto-approve every tool request in this session and continue with this one.',
    ko: '알겠습니다. 이 세션 동안 모든 도구 요청을 자동으로 승인하고, 이번 요청도 계속 진행할게요.',
  },
  approvalAlways: {
    en: 'Got it. I will auto-allow the same kind of tool request from now on and continue with this one.',
    ko: '알겠습니다. 앞으로 같은 유형의 도구 요청은 자동으로 허용하고, 이번 요청도 계속 진행할게요.',
  },
  approvalSession: {
    en: 'Got it. I will allow the same kind of tool request for this session and continue with this one.',
    ko: '알겠습니다. 이 세션 동안 같은 유형의 도구 요청을 허용하고, 이번 요청도 계속 진행할게요.',
  },
  approvalOnce: {
    en: 'Got it. I approved the{tool} request and will keep going.',
    ko: '알겠습니다.{tool} 요청을 승인했고 계속 진행할게요.',
  },
  approvalNoLiveRequest: {
    en: 'No live approval request for {id}.',
    ko: '{id} 에 해당하는 대기 중인 승인 요청이 없습니다.',
  },
  error: {
    en: 'Something went wrong while processing your request: {error}',
    ko: '요청을 처리하는 중 문제가 발생했습니다: {error}',
  },
}

const HANGUL_RE = /[가-힯ᄀ-ᇿ㄰-㆏]/

export function detectChannelLang(text: string | undefined): ChannelLang {
  if (text && HANGUL_RE.test(text)) return 'ko'
  return 'en'
}

export function channelLangSignal(text: string | undefined): ChannelLang | null {
  if (text && HANGUL_RE.test(text)) return 'ko'
  return null
}

export function channelMessage(
  key: ChannelMessageKey,
  lang: ChannelLang,
  params: Record<string, string> = {},
): string {
  const entry = CATALOG[key]
  const template = entry[lang] ?? entry.en
  return template.replace(/\{(\w+)\}/g, (_, name: string) => params[name] ?? '')
}
