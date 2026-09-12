export interface InlineAgentProgressInput {
  phase?: string | null
  streamStatus?: string | null
  counts?: {
    criteriaTotal: number
    planTotal: number
    planDone: number
    todosTotal: number
    todosDone: number
  } | null
}

export function formatInlineAgentProgress(input: InlineAgentProgressInput): string | null {
  const parts: string[] = []
  const activity = input.phase?.trim() || input.streamStatus?.trim()
  if (activity) parts.push(activity)
  if (input.counts?.planTotal) parts.push(`plan ${input.counts.planDone}/${input.counts.planTotal}`)
  if (input.counts?.todosTotal) parts.push(`tasks ${input.counts.todosDone}/${input.counts.todosTotal}`)
  if (input.counts?.criteriaTotal) parts.push(`checks ${input.counts.criteriaTotal}`)
  return parts.length > 0 ? parts.join(' · ') : null
}
