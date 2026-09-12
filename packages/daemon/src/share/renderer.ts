import { toolExecutionPostureLabel } from '@sepilotd/core'
import type { SessionEvent, SessionMeta } from '@sepilotd/core'
import { homedir } from 'node:os'
import { sanitizeEvent, sanitizeText } from '../sessions/sanitize.js'
import { prepareEventForPublicShare, shareIncludesToolOutput } from './payload.js'

const SHARE_SANITIZE_OPTIONS = { home: homedir() }

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function renderTodoItems(event: Extract<SessionEvent, { type: 'todo_list' }>): string {
  const items = event.items
    .map((item) => {
      const status =
        item.status === 'completed'
          ? 'Done'
          : item.status === 'in_progress'
            ? 'Active'
            : item.status === 'blocked'
              ? 'Blocked'
              : item.status === 'cancelled'
                ? 'Cancelled'
                : 'Todo'
      return `<li><strong>${status}</strong><span>${escapeHtml(item.content)}</span></li>`
    })
    .join('')

  return [
    '<section class="event system">',
    '<div class="event-meta"><span class="event-kicker">Plan</span><strong>Todo list</strong></div>',
    `<ul class="todo-list">${items}</ul>`,
    '</section>',
  ].join('')
}

function renderContextItems(event: Extract<SessionEvent, { type: 'memory_context' }>): string {
  const items = event.items
    .map((item) =>
      [
        '<li>',
        `<strong>${escapeHtml(item.citationLabel)}</strong>`,
        `<span>${escapeHtml(item.snippet)}</span>`,
        '</li>',
      ].join(''),
    )
    .join('')

  return [
    '<section class="event context">',
    '<div class="event-meta"><span class="event-kicker">Context</span><strong>Relevant references</strong></div>',
    `<ul class="context-list">${items}</ul>`,
    '</section>',
  ].join('')
}

function renderSystemNote(title: string, detail: string, kicker = 'Run'): string {
  return [
    '<section class="event system">',
    `<div class="event-meta"><span class="event-kicker">${escapeHtml(kicker)}</span><strong>${escapeHtml(title)}</strong></div>`,
    `<pre>${escapeHtml(detail)}</pre>`,
    '</section>',
  ].join('')
}

function renderPanelRoster(personas: Array<{ id: string; name: string; description?: string }>): string {
  if (personas.length === 0) return 'No resolved panelists.'
  return personas
    .map((persona, index) =>
      `${index + 1}. ${persona.name}${persona.description ? ` - ${persona.description}` : ''}`,
    )
    .join('\n')
}

function renderDebateRound(event: Extract<SessionEvent, { type: 'debate_round' }>): string {
  const round = event.round
  const entries = Array.isArray(round.entries)
    ? round.entries
        .map((entry) => `${entry.role.toUpperCase()}\n${entry.content}`)
        .join('\n\n')
    : ''
  const detail = [
    `Topic: ${round.topic ?? round.roundId ?? 'untitled'}`,
    round.finalDecision ? `Decision: ${round.finalDecision}` : '',
    entries,
    round.rationale ? `Rationale: ${round.rationale}` : '',
  ]
    .filter(Boolean)
    .join('\n\n')
  return renderSystemNote('Debate round', detail, 'Debate')
}

function renderEvent(event: SessionEvent): string {
  switch (event.type) {
    case 'user_message':
      return [
        '<section class="event user">',
        '<div class="event-meta"><span class="event-kicker">User</span><strong>Prompt</strong></div>',
        `<pre>${escapeHtml(event.content)}</pre>`,
        '</section>',
      ].join('')
    case 'assistant_message':
      return [
        '<section class="event assistant">',
        '<div class="event-meta"><span class="event-kicker">Assistant</span><strong>Response</strong></div>',
        `<pre>${escapeHtml(event.content)}</pre>`,
        '</section>',
      ].join('')
    case 'memory_context':
      return renderContextItems(event)
    case 'tool_call':
      return renderSystemNote(
        `Tool call: ${event.tool}`,
        JSON.stringify(event.input, null, 2),
        'Tool',
      )
    case 'tool_result': {
      const posture = toolExecutionPostureLabel(event.executionPosture)
      return renderSystemNote(
        `Tool result: ${event.status}`,
        posture ? `${event.output}\n\nSafety: ${posture}` : event.output,
        'Tool',
      )
    }
    case 'approval_request':
      return renderSystemNote(
        `Approval required: ${event.tool}`,
        `Request ${event.toolCallId}`,
        'Approval',
      )
    case 'approval_response': {
      // Defensive `approved` fallback for legacy imports — the typed
      // schema makes decision required, but old jsonl rows could
      // omit it. (Cast sidesteps TS exhaustive narrowing.)
      const fallbackApproved = (event as { approved?: boolean }).approved
      return renderSystemNote(
        `Approval ${event.decision ?? (fallbackApproved ? 'approved' : 'denied')}`,
        `By ${event.approvedBy}${event.note ? `\n${event.note}` : ''}`,
        'Approval',
      )
    }
    case 'auto_approval':
      return renderSystemNote(
        `Auto-${event.decision}: ${event.tool}`,
        `Matched ${event.scope} rule '${event.rule.pattern}'`,
        'Approval',
      )
    case 'cowork_plan':
      return renderSystemNote(
        'Cowork plan',
        event.plan.map((step) => `${step.role}: ${step.instruction}`).join('\n'),
        'Cowork',
      )
    case 'cowork_task_start':
      return renderSystemNote(`Cowork ${event.role} started`, event.instruction, 'Cowork')
    case 'cowork_task_complete':
      return renderSystemNote(
        `Cowork ${event.role} completed`,
        `${event.instruction}\n\n${event.result}`,
        'Cowork',
      )
    case 'cowork_task_failed':
      return renderSystemNote(
        `Cowork ${event.role} failed`,
        `${event.instruction}\n\n${event.error}`,
        'Cowork',
      )
    case 'cowork_synthesizing':
      return renderSystemNote('Cowork synthesizing', event.summary, 'Cowork')
    case 'cowork_discuss_request':
      return renderSystemNote(
        'Cowork question',
        `${event.prompt}${event.choices?.length ? `\n\nChoices: ${event.choices.join(', ')}` : ''}`,
        'Cowork',
      )
    case 'cowork_discuss_response':
      return renderSystemNote('Cowork answer', `${event.prompt}\n\n${event.response}`, 'Cowork')
    case 'panel_open':
      return renderSystemNote('Persona panel opened', renderPanelRoster(event.personas), 'Panel')
    case 'panel_turn_complete':
      return [
        '<section class="event assistant">',
        `<div class="event-meta"><span class="event-kicker">Panelist</span><strong>${escapeHtml(event.personaName)}</strong></div>`,
        `<pre>${escapeHtml(event.text)}</pre>`,
        '</section>',
      ].join('')
    case 'panel_turn_failed':
      return renderSystemNote(`Panelist ${event.personaName} failed`, event.error, 'Panel')
    case 'panel_synthesizing':
      return renderSystemNote(
        'Persona panel synthesizing',
        `${event.panelists} panelists`,
        'Panel',
      )
    case 'debate_round':
      return renderDebateRound(event)
    case 'todo_list':
      return renderTodoItems(event)
    case 'delegation_state':
      return renderSystemNote(
        `Delegation ${event.claimHealth}`,
        `${event.targetDevice}\n${event.detail}`,
        'Delegation',
      )
    case 'delegation_result':
      return renderSystemNote(
        `Delegation ${event.status}`,
        [
          event.targetDevice,
          event.result,
          event.artifactHandles?.length
            ? `Artifacts: ${event.artifactHandles.join(', ')}`
            : undefined,
        ].filter(Boolean).join('\n'),
        'Delegation',
      )
    case 'context_compact':
      return renderSystemNote(
        'Context compacted',
        `${event.beforeTokens} -> ${event.afterTokens} tokens\n\n${event.summary}`,
      )
    case 'provider_attempt':
      return renderSystemNote(
        `Provider attempt ${event.status}`,
        [
          `${event.provider}/${event.model}`,
          event.errorMessage,
          event.nextProvider && event.nextModel
            ? `Retrying with ${event.nextProvider}/${event.nextModel}`
            : undefined,
        ]
          .filter(Boolean)
          .join('\n'),
        'Provider',
      )
    default:
      return ''
  }
}

export function renderSharedSessionHtml(
  session: Pick<SessionMeta, 'id' | 'title' | 'createdAt' | 'provider' | 'model' | 'device'>,
  events: SessionEvent[],
  options: { includeToolOutput?: boolean } = {},
): string {
  // Public share URLs are token-gated but anyone with the URL can read
  // the full transcript. Strip verbatim tool outputs (default) then run
  // every event through the same sanitiser the session export uses so
  // accidentally-pasted bearer tokens / API keys / $HOME paths and raw
  // tool stdout/file contents don't bleed into the rendered HTML.
  const includeToolOutput = options.includeToolOutput ?? shareIncludesToolOutput()
  const sanitized = events.map((event) =>
    sanitizeEvent(prepareEventForPublicShare(event, includeToolOutput), SHARE_SANITIZE_OPTIONS),
  )
  const transcript = sanitized.map(renderEvent).filter(Boolean).join('\n')
  // Header fields also pass through sanitizeText so $HOME / bearer
  // tokens / API keys in user-customised session titles or device
  // names don't leak.
  const title = escapeHtml(sanitizeText(session.title, SHARE_SANITIZE_OPTIONS))
  const createdAt = escapeHtml(session.createdAt)
  const providerModel = escapeHtml(`${session.provider}/${session.model}`)
  const device = escapeHtml(sanitizeText(session.device, SHARE_SANITIZE_OPTIONS))
  const sessionId = escapeHtml(session.id)

  return [
    '<!doctype html>',
    '<html lang="en">',
    '<head>',
    '<meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width, initial-scale=1">',
    `<title>${title}</title>`,
    '<style>',
    ':root { color-scheme: dark; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; }',
    'body { margin: 0; background: linear-gradient(180deg, #0f1319 0%, #101722 100%); color: #eef2f7; }',
    '.shell { max-width: 920px; margin: 0 auto; padding: 40px 20px 64px; }',
    '.hero { display: grid; gap: 12px; margin-bottom: 24px; padding: 24px; border-radius: 24px; border: 1px solid rgba(231,238,248,0.08); background: rgba(14,18,25,0.88); box-shadow: 0 20px 60px rgba(0,0,0,0.24); }',
    '.eyebrow { font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase; color: rgba(231,238,248,0.6); }',
    '.hero h1 { margin: 0; font-size: clamp(28px, 4vw, 40px); }',
    '.meta { display: flex; flex-wrap: wrap; gap: 10px; }',
    '.meta span { display: inline-flex; padding: 6px 10px; border-radius: 999px; border: 1px solid rgba(231,238,248,0.08); background: rgba(255,255,255,0.03); color: rgba(231,238,248,0.7); font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; }',
    '.transcript { display: grid; gap: 14px; }',
    '.event { display: grid; gap: 12px; padding: 18px 20px; border-radius: 20px; border: 1px solid rgba(231,238,248,0.08); background: rgba(255,255,255,0.03); }',
    '.event.user { background: rgba(124,181,255,0.11); border-color: rgba(124,181,255,0.22); }',
    '.event.assistant { background: rgba(255,255,255,0.025); }',
    '.event.context { background: rgba(102,209,143,0.08); border-color: rgba(102,209,143,0.22); }',
    '.event.system { background: rgba(255,184,77,0.08); border-color: rgba(255,184,77,0.18); }',
    '.event-meta { display: grid; gap: 4px; }',
    '.event-kicker { font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: rgba(231,238,248,0.55); }',
    'pre { margin: 0; white-space: pre-wrap; word-break: break-word; font: 14px/1.6 "IBM Plex Mono", monospace; }',
    '.todo-list, .context-list { margin: 0; padding-left: 18px; display: grid; gap: 10px; }',
    '.todo-list li, .context-list li { display: grid; gap: 4px; }',
    '.todo-list strong, .context-list strong { font-size: 13px; }',
    '.todo-list span, .context-list span { color: rgba(231,238,248,0.72); line-height: 1.5; }',
    '.footer { margin-top: 20px; color: rgba(231,238,248,0.56); font-size: 12px; text-align: center; }',
    '@media (max-width: 640px) { .shell { padding: 20px 14px 40px; } .hero, .event { padding: 16px; border-radius: 18px; } }',
    '</style>',
    '</head>',
    '<body>',
    '<main class="shell">',
    '<section class="hero">',
    '<span class="eyebrow">Shared Session</span>',
    `<h1>${title}</h1>`,
    '<div class="meta">',
    `<span>${providerModel}</span>`,
    `<span>${device}</span>`,
    `<span>${createdAt}</span>`,
    `<span>${sessionId}</span>`,
    '</div>',
    '</section>',
    `<section class="transcript">${transcript}</section>`,
    '<p class="footer">Read-only share generated by sepilotd</p>',
    '</main>',
    '</body>',
    '</html>',
  ].join('')
}
