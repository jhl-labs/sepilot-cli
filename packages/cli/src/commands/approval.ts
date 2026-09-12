import chalk from 'chalk'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { friendlyErrorMessage, printApiError } from '../utils/error-message.js'
import { detectCliLocale } from '../utils/locale.js'

const APPROVAL_COPY = {
  en: {
    unknownScope: (raw: string) => `Unknown --scope: ${raw}. Use once|session|always|run|session-all.`,
    persistentDenialsUnsupported:
      'Persistent denials (--scope session|always|run|session-all) are not supported. '
      + 'Use `sepilot approve --scope session` or edit the policy file.',
    approved: 'Approved',
    denied: 'Denied',
    scopeLabel: (scope: string) => ` (scope: ${scope})`,
    patternPrefix: (tool: string, pattern: string) => `  pattern: ${tool} → ${pattern}`,
    liveFollowUp: (sessionId: string) =>
      `  run: live; return to the waiting command, or inspect with: sepilot sessions show ${sessionId}`,
    staleFollowUp: (sessionId: string) =>
      `  run: stale; resume with: sepilot sessions resume ${sessionId}`,
    failedVerb: (verb: string, requestId: string, msg: string) => `Failed to ${verb} ${requestId}: ${msg}`,
  },
  ko: {
    unknownScope: (raw: string) => `알 수 없는 --scope: ${raw}. once|session|always|run|session-all을 사용하세요.`,
    persistentDenialsUnsupported:
      '지속적 거부(--scope session|always|run|session-all)는 지원되지 않습니다. '
      + '`sepilot approve --scope session`을 사용하거나 정책 파일을 편집하세요.',
    approved: '승인됨',
    denied: '거부됨',
    scopeLabel: (scope: string) => ` (범위: ${scope})`,
    patternPrefix: (tool: string, pattern: string) => `  패턴: ${tool} → ${pattern}`,
    liveFollowUp: (sessionId: string) =>
      `  실행: live; 기다리던 명령으로 돌아가거나 확인하세요: sepilot sessions show ${sessionId}`,
    staleFollowUp: (sessionId: string) =>
      `  실행: stale; 재개하세요: sepilot sessions resume ${sessionId}`,
    failedVerb: (verb: string, requestId: string, msg: string) => `${requestId} ${verb} 실패: ${msg}`,
  },
} as const

export interface ApprovalCommandOptions {
  url?: string
  session?: string
  scope?: string
  reason?: string
}

const VALID_SCOPES = new Set(['once', 'session', 'always', 'run', 'command', 'task', 'session-all', 'session_all'])

function parseScope(raw: string | undefined): 'once' | 'session' | 'always' | 'run' | 'session-all' {
  if (!raw) return 'once'
  const trimmed = raw.trim().toLowerCase()
  if (trimmed === 'command' || trimmed === 'task') return 'run'
  if (trimmed === 'session_all') return 'session-all'
  if (VALID_SCOPES.has(trimmed)) {
    return trimmed as 'once' | 'session' | 'always' | 'run' | 'session-all'
  }
  const copy = APPROVAL_COPY[detectCliLocale()] ?? APPROVAL_COPY.en
  console.error(chalk.red(copy.unknownScope(raw)))
  process.exit(1)
}

async function respond(
  verb: 'approve' | 'deny',
  requestId: string,
  options: ApprovalCommandOptions,
  approved: boolean,
): Promise<void> {
  const copy = APPROVAL_COPY[detectCliLocale()] ?? APPROVAL_COPY.en
  const scope = parseScope(options.scope)
  // Persistent denials are a heavier policy decision than a one-off "no"
  // — mirror the chat shell's `/deny` rule and refuse session/always
  // here so users land on `approve --scope session` (or the policy
  // file) for the explicit case.
  if (!approved && scope !== 'once') {
    console.error(chalk.red(copy.persistentDenialsUnsupported))
    process.exit(1)
  }

  const client = new DaemonClient(options.url)
  try {
    const result = await client.respondApproval(
      requestId,
      approved,
      { sessionId: options.session, scope, note: options.reason },
    )
    output(result, (data) => {
      const verbCap = verb === 'approve' ? copy.approved : copy.denied
      const scopeLabel = scope !== 'once' ? copy.scopeLabel(scope) : ''
      const noteLabel = options.reason ? ` — ${options.reason}` : ''
      const lines = [chalk.green(`${verbCap} ${requestId}${scopeLabel}${noteLabel}`)]
      // Surface the derived rule pattern when the daemon persisted a
      // session/always decision. The pattern is the *exact* shape
      // future invocations must match to short-circuit (e.g.
      // `ls /tmp/foo *`); without seeing it the operator might
      // assume a different command would also auto-approve and be
      // surprised by a re-prompt. This closes the round-trip ux gap
      // discovered while live-testing --scope session.
      if (data?.rule?.pattern) {
        lines.push(chalk.gray(copy.patternPrefix(data.rule.tool, data.rule.pattern)))
      }
      if (options.session) {
        lines.push(chalk.gray(
          data?.state === 'stale'
            ? copy.staleFollowUp(options.session)
            : copy.liveFollowUp(options.session),
        ))
      }
      return lines.join('\n')
    })
  } catch (err) {
    if (printApiError(err)) {
      process.exit(1)
    }
    console.error(chalk.red(copy.failedVerb(verb, requestId, friendlyErrorMessage(err))))
    process.exit(1)
  }
}

export async function approveCommand(
  requestId: string,
  options: ApprovalCommandOptions = {},
): Promise<void> {
  await respond('approve', requestId, options, true)
}

export async function denyCommand(
  requestId: string,
  options: ApprovalCommandOptions = {},
): Promise<void> {
  await respond('deny', requestId, options, false)
}
