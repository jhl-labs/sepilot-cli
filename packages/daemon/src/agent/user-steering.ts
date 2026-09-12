export interface AgentSteeringNote {
  id: string
  message: string
  kind: 'instruction' | 'question'
  createdAt: number
  /** Delivery acknowledgement, not instruction expiry. */
  consumedAt?: number
  cancelledAt?: number
}

export function activeUserInstructions(notes: readonly AgentSteeringNote[] = []): string[] {
  return notes.filter((note) => note.kind === 'instruction' && note.cancelledAt === undefined)
    .map((note) => note.message)
}

export const USER_INSTRUCTION_PRECEDENCE =
  'User instructions below remain active throughout execution and verification. Apply them in arrival order; newer instructions amend conflicting earlier goals, plans, and acceptance criteria. Preserve non-conflicting requirements and tool/security policy. Delivery acknowledgement does not retire an instruction.'

export function formatActiveUserInstructions(instructions: readonly string[] = []): string {
  return instructions.length > 0
    ? ['[Active user instructions]', USER_INSTRUCTION_PRECEDENCE, ...instructions.map((message) => `- ${message}`)].join('\n')
    : ''
}
