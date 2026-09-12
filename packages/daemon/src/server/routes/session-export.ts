import { toolExecutionPostureLabel } from '@sepilotd/core'
import type { SessionEvent, SessionMeta } from '@sepilotd/core'
import type { HealthExportSnapshot } from '../health-support.js'

type SessionExportMeta = SessionMeta & { primaryAgentId?: string }
const TOOL_EXPORT_BLOCK_CHAR_LIMIT = 4_000

export interface SessionJsonExport {
  session: SessionExportMeta
  events: SessionEvent[]
  health?: HealthExportSnapshot
}

export function buildSessionJsonExport(
  session: SessionExportMeta,
  events: SessionEvent[],
  healthSnapshot: HealthExportSnapshot | null,
): SessionJsonExport {
  return {
    session,
    events,
    ...(healthSnapshot ? { health: healthSnapshot } : {}),
  }
}

function renderHealthSection(healthSnapshot: HealthExportSnapshot): string {
  let md = '## Runtime Health\n\n'
  md += `**Status:** ${healthSnapshot.health.status}\n`
  md += `**Snapshot:** ${healthSnapshot.generatedAt}\n`
  md += `**Readiness:** ${healthSnapshot.readiness.status}\n\n`
  for (const [name, component] of Object.entries(healthSnapshot.health.components)) {
    md += `- **${name}**: ${component.status}`
    if (component.details) {
      md += ` — ${component.details}`
    }
    md += '\n'
  }
  md += '\n---\n\n'
  return md
}

function renderPanelRoster(personas: Array<{ id: string; name: string; description?: string }>): string {
  if (personas.length === 0) return 'No resolved panelists.'
  return personas
    .map((persona, index) =>
      `${index + 1}. ${persona.name}${persona.description ? ` - ${persona.description}` : ''}`,
    )
    .join('\n')
}

function renderDebateRound(
  round: Extract<SessionEvent, { type: 'debate_round' }>['round'],
): string {
  const entries = Array.isArray(round.entries)
    ? round.entries
        .map((entry) => [
          `#### ${entry.role}`,
          '',
          entry.content,
          '',
        ].join('\n'))
        .join('\n')
    : ''
  const decision = round.finalDecision ? `**Decision:** ${round.finalDecision}\n\n` : ''
  const rationale = round.rationale ? `**Rationale:** ${round.rationale}\n\n` : ''
  return [
    `### Debate Round: ${round.topic ?? round.roundId ?? 'untitled'}`,
    '',
    decision,
    entries,
    rationale,
  ].join('\n')
}

function truncateForExportBlock(
  value: string,
  limit = TOOL_EXPORT_BLOCK_CHAR_LIMIT,
): { text: string; omitted: number } {
  if (value.length <= limit) return { text: value, omitted: 0 }
  return { text: value.slice(0, limit), omitted: value.length - limit }
}

function markdownFenceForContent(value: string): string {
  const maxBackticks = Math.max(0, ...Array.from(value.matchAll(/`+/g), (match) => match[0].length))
  return '`'.repeat(Math.max(3, maxBackticks + 1))
}

function renderQuotedCodeBlock(
  label: string,
  value: string,
  language = '',
  limit = TOOL_EXPORT_BLOCK_CHAR_LIMIT,
): string {
  const { text, omitted } = truncateForExportBlock(value, limit)
  const fence = markdownFenceForContent(text)
  const lines = [
    `> **${label}:**`,
    `> ${fence}${language}`,
    ...text.split(/\r?\n/).map((line) => `> ${line}`),
    `> ${fence}`,
  ]
  if (omitted > 0) {
    lines.push(`> _Truncated ${omitted.toLocaleString()} characters._`)
  }
  return `${lines.join('\n')}\n\n`
}

function renderEvent(event: SessionEvent): string {
  switch (event.type) {
    case 'user_message':
      return `### User\n\n${event.content}\n\n`
    case 'memory_context': {
      let md = `> **Relevant context**\n`
      for (const item of event.items) {
        md += `> - **${item.citationLabel}**: ${item.snippet}\n`
      }
      return `${md}\n`
    }
    case 'assistant_message':
      return `### Assistant\n\n${event.content}\n\n`
    case 'tool_call': {
      const input = JSON.stringify(event.input ?? null, null, 2)
      return `> **Tool:** \`${event.tool}\`\n${renderQuotedCodeBlock('Input', input, 'json')}`
    }
    case 'tool_result': {
      const recoveryNote =
        event.recovery === 'journal'
          ? ', recovered from saved execution'
          : event.recovery === 'probe'
            ? ', recovered from verified side effect'
            : ''
      const postureNote = toolExecutionPostureLabel(event.executionPosture)
      const header = `> **Result** (${event.status}${recoveryNote})`
      const safety = postureNote ? `> Safety: ${postureNote}\n` : ''
      return `${header}\n${safety}${renderQuotedCodeBlock('Output', event.output)}`
    }
    case 'approval_request':
      return `> **Approval required:** \`${event.tool}\`\n> Request: \`${event.toolCallId}\`\n\n`
    case 'approval_response': {
      const fallbackApproved = (event as { approved?: boolean }).approved
      return `> **Approval ${event.decision ?? (fallbackApproved ? 'approved' : 'denied')}** by \`${event.approvedBy}\`${event.note ? `\n> ${event.note}` : ''}\n\n`
    }
    case 'auto_approval':
      return `> **Auto-${event.decision}:** \`${event.tool}\`\n> Matched ${event.scope} rule \`${event.rule.pattern}\`\n\n`
    case 'cowork_plan': {
      const steps = event.plan.map((step) => `- \`${step.role}\`: ${step.instruction}`).join('\n')
      return `> **Cowork plan**\n${steps ? `${steps}\n\n` : '\n'}`
    }
    case 'cowork_task_start':
      return `> **Cowork ${event.role} started**\n> ${event.instruction}\n\n`
    case 'cowork_task_complete':
      return `> **Cowork ${event.role} completed**\n> ${event.instruction}\n>\n> ${event.result.slice(0, 200)}\n\n`
    case 'cowork_task_failed':
      return `> **Cowork ${event.role} failed**\n> ${event.instruction}\n>\n> ${event.error}\n\n`
    case 'cowork_synthesizing':
      return `> **Cowork synthesizing**\n> ${event.summary}\n\n`
    case 'cowork_discuss_request':
      return `> **Cowork question**\n> ${event.prompt}${event.choices?.length ? `\n> Choices: ${event.choices.join(', ')}` : ''}\n\n`
    case 'cowork_discuss_response':
      return `> **Cowork answer**\n> ${event.prompt}\n>\n> ${event.response}\n\n`
    case 'panel_open':
      return `> **Persona panel opened**\n${renderPanelRoster(event.personas)
        .split('\n')
        .map((line) => `> ${line}`)
        .join('\n')}\n\n`
    case 'panel_turn_complete':
      return `### Panelist: ${event.personaName}\n\n${event.text}\n\n`
    case 'panel_turn_failed':
      return `> **Panelist ${event.personaName} failed**\n> ${event.error}\n\n`
    case 'panel_synthesizing':
      return `> **Persona panel synthesizing**\n> ${event.panelists} panelists\n\n`
    case 'debate_round':
      return `${renderDebateRound(event.round)}\n`
    case 'context_compact':
      return `---\n*Context compacted: ${event.beforeTokens} -> ${event.afterTokens} tokens*\n\n${event.summary}\n\n---\n\n`
    case 'phase_change': {
      const entered = event.enteredPhase ?? 'finalize'
      if (event.closedPhase) {
        const closedTokens =
          event.closedPhase.usage.inputTokens + event.closedPhase.usage.outputTokens
        return `> **Phase:** entered \`${entered}\`, closed \`${event.closedPhase.phase}\` (${closedTokens.toLocaleString()} tokens)\n\n`
      }
      return `> **Phase:** entered \`${entered}\`\n\n`
    }
    case 'post_edit_findings': {
      // Surface the same structural information the agent saw via
      // reflectionMemo so the markdown reader can review what the run
      // touched and which downstream files lit up.
      const lines = ['> **Post-edit findings**']
      if (event.editedFiles.length > 0) {
        lines.push(`> Edited: ${event.editedFiles.map((f) => `\`${f}\``).join(', ')}`)
      }
      if (event.reverseCallers.length > 0) {
        lines.push(
          `> Likely callers: ${event.reverseCallers
            .slice(0, 5)
            .map((f) => `\`${f}\``)
            .join(', ')}`,
        )
      }
      const broken = event.diagnostics.filter((d) => d.summary.startsWith('[caller]'))
      const own = event.diagnostics.filter((d) => !d.summary.startsWith('[caller]'))
      if (broken.length > 0) {
        lines.push(`> Broken callers (${broken.length}):`)
        for (const d of broken.slice(0, 3)) {
          lines.push(`> - \`${d.file}\`: ${d.summary.replace(/\[caller\]\s*/, '').split('\n')[0]}`)
        }
      }
      if (own.length > 0) {
        lines.push(`> Diagnostics (${own.length}):`)
        for (const d of own.slice(0, 3)) {
          lines.push(`> - \`${d.file}\`: ${d.summary.split('\n')[0]}`)
        }
      }
      return `${lines.join('\n')}\n\n`
    }
    case 'memory_summary':
      return (
        `> **Memory summary:** ${event.stage} via \`${event.source}\`` +
        `${event.turnId ? ` for turn \`${event.turnId}\`` : ''}` +
        ` (${event.lightCaptured ? 'light captured' : 'no light note'}, ` +
        `${event.semanticMemoriesExtracted} semantic memories extracted, ` +
        `${event.ragContextPromotions ?? 0} RAG context promotions)\n\n`
      )
    case 'provider_attempt': {
      const target = `\`${event.provider}/${event.model}\``
      if (event.status === 'failed') {
        const next =
          event.nextProvider && event.nextModel
            ? `; retrying with \`${event.nextProvider}/${event.nextModel}\``
            : ''
        return `> **Provider attempt failed:** ${target}${next}${event.errorMessage ? `\n> ${event.errorMessage}` : ''}\n\n`
      }
      return `> **Provider attempt ${event.status}:** ${target}\n\n`
    }
    case 'recovery':
      return `> **Recovery:** \`${event.scope}/${event.kind}\` via \`${event.action}\`\n> ${event.message}${event.recoverable ? '' : '\n> Not recoverable'}\n\n`
    case 'todo_list': {
      const items = event.items
        .map((item) => {
          const box = item.status === 'completed' ? '[x]' : '[ ]'
          const suffix = item.status === 'in_progress'
            ? ' (in progress)'
            : item.status === 'blocked'
              ? ' (blocked)'
              : item.status === 'cancelled'
                ? ' (cancelled)'
                : ''
          return `- ${box} ${item.content}${suffix}`
        })
        .join('\n')
      return `> **Todo list updated**\n${items ? `${items}\n\n` : '\n'}`
    }
    case 'delegation_state': {
      const sourceNote = event.source ? ` via \`${event.source}\`` : ''
      return `> **Delegation ${event.claimHealth}** on \`${event.targetDevice}\`${sourceNote}\n> ${event.detail}\n\n`
    }
    case 'delegation_result': {
      const handles = event.artifactHandles?.length
        ? `\n> Artifacts: ${event.artifactHandles.map((handle) => `\`${handle}\``).join(', ')}`
        : ''
      return `> **Delegation ${event.status}** on \`${event.targetDevice}\`${handles}${event.result ? `\n> ${event.result.slice(0, 200)}` : ''}\n\n`
    }
    default:
      return ''
  }
}

export function serializeSessionMarkdown(
  session: SessionExportMeta,
  events: SessionEvent[],
  healthSnapshot: HealthExportSnapshot | null,
): string {
  let md = `# ${session.title}\n\n`
  md += `**Session:** ${session.id}\n`
  md += `**Date:** ${session.createdAt}\n`
  md += `**Provider:** ${session.provider}/${session.model}\n`
  md += `**Messages:** ${session.messageCount}\n\n---\n\n`

  if (healthSnapshot) {
    md += renderHealthSection(healthSnapshot)
  }

  for (const event of events) {
    md += renderEvent(event)
  }

  md += `\n---\n*Tokens: ${session.totalTokens.input} in / ${session.totalTokens.output} out | Cost: $${session.totalCost.toFixed(4)}*\n`

  // Phase trajectory summary — collapsed table so the analyst can see
  // at a glance how the run's tokens were spent across named phases
  // without grepping every phase_change line.
  const phaseTrajectory = renderPhaseTrajectorySection(events)
  if (phaseTrajectory) {
    md += `\n${phaseTrajectory}`
  }
  return md
}

function renderPhaseTrajectorySection(events: SessionEvent[]): string {
  const totals = new Map<string, { input: number; output: number }>()
  for (const event of events) {
    if (event.type !== 'phase_change') continue
    if (!event.closedPhase) continue
    const { phase, usage } = event.closedPhase
    const acc = totals.get(phase) ?? { input: 0, output: 0 }
    acc.input += usage.inputTokens
    acc.output += usage.outputTokens
    totals.set(phase, acc)
  }
  if (totals.size === 0) return ''
  const lines = [
    '## Phase trajectory',
    '',
    '| Phase | Input | Output | Total |',
    '| --- | --- | --- | --- |',
  ]
  for (const [phase, usage] of totals) {
    const total = usage.input + usage.output
    lines.push(
      `| \`${phase}\` | ${usage.input.toLocaleString()} | ${usage.output.toLocaleString()} | ${total.toLocaleString()} |`,
    )
  }
  return lines.join('\n') + '\n'
}
