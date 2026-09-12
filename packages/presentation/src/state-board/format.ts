import type { AgentRunContract, AgentStateBoardSnapshot } from '@sepilotd/core'

/**
 * Shared, dependency-light text renderer for the agent state board. Lives in
 * `presentation` (core-only) so cli and web render the same board the same way
 * without importing daemon. Mirrors the daemon's prompt-side formatter but
 * operates on the serializable `AgentStateBoardSnapshot`.
 */

const MAX_PLAN_STEPS = 20
const MAX_TODO_ITEMS = 20
const MAX_OPEN_QUESTIONS = 10
const MAX_FAILED_ATTEMPTS = 10

export function stateBoardIsEmpty(board: AgentStateBoardSnapshot): boolean {
  return (
    !board.goal
    && !board.contract
    && board.completionCriteria.length === 0
    && board.plan.length === 0
    && board.todos.length === 0
    && board.decisions.length === 0
    && board.failedAttempts.length === 0
    && board.openQuestions.length === 0
    && board.evidenceSection === null
  )
}

function formatContract(contract: AgentRunContract | undefined): string[] {
  if (!contract) return []
  const lines = ['Run contract:', `  Goal: ${contract.summary}`]
  if (contract.acceptanceCriteria.length > 0) {
    lines.push('  Acceptance criteria:')
    for (const criterion of contract.acceptanceCriteria) {
      lines.push(`    - ${criterion.id}: ${criterion.text}`)
    }
  }
  if (contract.constraints.length > 0) {
    lines.push('  Constraints:', ...contract.constraints.map((entry) => `    - ${entry}`))
  }
  if (contract.outOfScope.length > 0) {
    lines.push('  Out of scope:', ...contract.outOfScope.map((entry) => `    - ${entry}`))
  }
  return lines
}

/**
 * Render the board as human-readable text. Returns `null` for an empty board so
 * callers can show an explicit "no state board" message.
 */
export function formatStateBoardText(board: AgentStateBoardSnapshot): string | null {
  if (stateBoardIsEmpty(board)) return null
  const lines: string[] = ['Agent state board']
  if (board.goal) lines.push('', `Goal: ${board.goal}`)

  if (board.completionCriteria.length > 0) {
    lines.push('', 'Completion criteria:')
    for (const criterion of board.completionCriteria) {
      lines.push(`  - ${criterion.id}: ${criterion.text}`)
    }
  }

  const contractLines = formatContract(board.contract)
  if (contractLines.length > 0) lines.push('', ...contractLines)

  if (board.plan.length > 0) {
    lines.push('', 'Plan:')
    for (const step of board.plan.slice(0, MAX_PLAN_STEPS)) {
      lines.push(`${'  '.repeat(step.depth + 1)}- [${step.status}] ${step.id}: ${step.title}`)
    }
    if (board.plan.length > MAX_PLAN_STEPS) {
      lines.push(`  (+${board.plan.length - MAX_PLAN_STEPS} more plan steps)`)
    }
  }

  if (board.todos.length > 0) {
    const doneCount = board.todos.filter((item) => item.status === 'completed').length
    lines.push('', `Todos (${doneCount}/${board.todos.length}):`)
    for (const item of board.todos.slice(0, MAX_TODO_ITEMS)) {
      const marker = item.status === 'completed' ? 'x' : ' '
      lines.push(`  - [${marker}] ${item.content}`)
    }
    if (board.todos.length > MAX_TODO_ITEMS) {
      lines.push(`  (+${board.todos.length - MAX_TODO_ITEMS} more todo items)`)
    }
  }

  if (board.decisions.length > 0) {
    lines.push('', 'Decisions taken:')
    for (const decision of board.decisions) lines.push(`  - ${decision}`)
  }

  if (board.openQuestions.length > 0) {
    lines.push('', 'Open questions:')
    for (const question of board.openQuestions.slice(0, MAX_OPEN_QUESTIONS)) {
      lines.push(`  - ${question.id}: ${question.text}${question.blocking ? ' (blocking)' : ''}`)
    }
    if (board.openQuestions.length > MAX_OPEN_QUESTIONS) {
      lines.push(`  (+${board.openQuestions.length - MAX_OPEN_QUESTIONS} more open questions)`)
    }
  }

  if (board.failedAttempts.length > 0) {
    lines.push('', 'Failed attempts (do NOT repeat):')
    const recent = board.failedAttempts.slice(-MAX_FAILED_ATTEMPTS)
    for (const attempt of recent) lines.push(`  - [${attempt.tool}] ${attempt.reason}`)
    if (board.failedAttempts.length > recent.length) {
      lines.push(`  (+${board.failedAttempts.length - recent.length} earlier failed attempts)`)
    }
  }

  if (board.evidenceSection) lines.push('', board.evidenceSection)

  return lines.join('\n')
}
