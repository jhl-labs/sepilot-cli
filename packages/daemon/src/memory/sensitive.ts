import { sanitizeText } from '../sessions/sanitize.js'

const SENSITIVE_LABEL_PATTERN =
  /\b(api[-_\s]?key|access[-_\s]?token|refresh[-_\s]?token|bot[-_\s]?token|token|password|passwd|secret|private[-_\s]?key)\b|(?:비밀번호|패스워드|시크릿|개인키|인증키|토큰)/i
const LONG_SECRET_LIKE_PATTERN =
  /\b(?:[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}(?:\.[A-Za-z0-9_-]{20,})?|[A-Za-z0-9_:\-]{42,})\b/

const KOREAN_RRN_PATTERN = /(?<![\d-])\d{6}[-\s]?[1-8]\d{6}(?![\d-])/
const US_SSN_PATTERN = /(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])/
const KOREAN_PHONE_PATTERN =
  /(?<![\d-])(?:\+?82[-\s]?)?(?:0?(?:1[016789]|2|3[1-3]|4[1-4]|5[1-5]|6[1-4]|7|70|80))[-\s]?\d{3,4}[-\s]?\d{4}(?![\d])/
const INTL_PHONE_PATTERN = /\+\d{1,3}[\s-]?(?:\(?\d{1,4}\)?[\s-]?){2,4}\d{2,4}/
const CARD_LIKE_PATTERN = /(?<![\d])(?:\d[\s-]?){12,18}\d(?![\d])/
const PII_LABEL_PATTERN =
  /(?:phone[-_\s]?number|mobile[-_\s]?number|social[-_\s]?security|ssn|credit[-_\s]?card|debit[-_\s]?card|card[-_\s]?number|account[-_\s]?number|routing[-_\s]?number|transit[-_\s]?number|passport|driver[-_\s]?license|tax[-_\s]?id|national[-_\s]?id|date[-_\s]?of[-_\s]?birth)|(?:전화\s*번호|휴대\s*폰|휴대\s*전화|핸드폰|핸폰|주민\s*등록\s*번호|주민\s*번호|카드\s*번호|계좌\s*번호|운전\s*면허\s*번호|여권\s*번호|생년월일|주소|우편\s*번호|건강\s*보험)/i
const PHONE_CONTEXT_PATTERN =
  /(?:phone[-_\s]?number|mobile[-_\s]?number|(?:call|contact|reach)\s+(?:me\s+)?(?:at|on))|(?:전화\s*번호|휴대\s*폰|휴대\s*전화|핸드폰|핸폰|연락처)/i

const REDACTION = '<redacted>'

function luhnValid(digits: string): boolean {
  let sum = 0
  let alternate = false
  for (let i = digits.length - 1; i >= 0; i--) {
    let n = Number(digits[i])
    if (Number.isNaN(n)) return false
    if (alternate) {
      n *= 2
      if (n > 9) n -= 9
    }
    sum += n
    alternate = !alternate
  }
  return sum > 0 && sum % 10 === 0
}

function containsLuhnValidCardNumber(content: string): boolean {
  const matches = content.match(new RegExp(CARD_LIKE_PATTERN.source, 'g'))
  if (!matches) return false
  for (const candidate of matches) {
    const digits = candidate.replace(/\D/g, '')
    if (digits.length < 13 || digits.length > 19) continue
    if (luhnValid(digits)) return true
  }
  return false
}

function looksLikePII(content: string): boolean {
  if (KOREAN_RRN_PATTERN.test(content)) return true
  if (US_SSN_PATTERN.test(content)) return true
  if (containsLuhnValidCardNumber(content)) return true
  if (PHONE_CONTEXT_PATTERN.test(content)
    && (KOREAN_PHONE_PATTERN.test(content) || INTL_PHONE_PATTERN.test(content))) {
    return true
  }
  if (PII_LABEL_PATTERN.test(content) && /\d{6,}/.test(content)) {
    return true
  }
  return false
}

export function looksSensitive(content: string): boolean {
  if (SENSITIVE_LABEL_PATTERN.test(content) && LONG_SECRET_LIKE_PATTERN.test(content)) {
    return true
  }
  if (looksLikePII(content)) {
    return true
  }
  return false
}

function globalize(pattern: RegExp): RegExp {
  return new RegExp(pattern.source, pattern.flags.includes('g') ? pattern.flags : `${pattern.flags}g`)
}

export function redactSensitive(text: string): { redacted: string; found: boolean } {
  if (!text) return { redacted: text, found: false }
  let found = false
  let out = text

  out = out.replace(globalize(CARD_LIKE_PATTERN), (match) => {
    const digits = match.replace(/\D/g, '')
    if (digits.length >= 13 && digits.length <= 19 && luhnValid(digits)) {
      found = true
      return REDACTION
    }
    return match
  })

  for (const pattern of [KOREAN_RRN_PATTERN, US_SSN_PATTERN]) {
    out = out.replace(globalize(pattern), () => {
      found = true
      return REDACTION
    })
  }

  if (PHONE_CONTEXT_PATTERN.test(out)) {
    for (const pattern of [KOREAN_PHONE_PATTERN, INTL_PHONE_PATTERN]) {
      out = out.replace(globalize(pattern), () => {
        found = true
        return REDACTION
      })
    }
  }

  const scrubbed = sanitizeText(out, { home: '' })
  if (scrubbed !== out) found = true
  out = scrubbed

  return { redacted: out, found }
}
