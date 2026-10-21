export interface ChatPersonaSelectionInput {
  persona?: string
  personaIds?: readonly string[]
}

export interface SessionPersonaSelection {
  personaIds?: readonly string[]
}

export interface ChatPersonaSelection {
  persona?: string
  personaIds?: string[]
  source: 'request' | 'session' | 'default'
}

export function resolveChatPersonaSelection(
  request: ChatPersonaSelectionInput,
  session: SessionPersonaSelection | null | undefined,
): ChatPersonaSelection {
  const hasRequestSelection = request.persona !== undefined || request.personaIds !== undefined
  const personaIds = hasRequestSelection
    ? request.personaIds === undefined
      ? undefined
      : [...request.personaIds]
    : session?.personaIds === undefined
      ? undefined
      : [...session.personaIds]
  const persona = request.persona ?? (personaIds?.length === 1 ? personaIds[0] : undefined)

  return {
    ...(persona ? { persona } : {}),
    ...(personaIds !== undefined ? { personaIds } : {}),
    source: hasRequestSelection ? 'request' : personaIds !== undefined ? 'session' : 'default',
  }
}

export function findUnknownChatPersonaIds(
  selection: ChatPersonaSelection,
  resolves: (id: string) => boolean,
): string[] {
  const candidates = [selection.persona, ...(selection.personaIds ?? [])]
  const seen = new Set<string>()
  const unknown: string[] = []

  for (const candidate of candidates) {
    if (!candidate || seen.has(candidate)) continue
    seen.add(candidate)
    if (!resolves(candidate)) unknown.push(candidate)
  }
  return unknown
}

export function unknownChatPersonaResponse(personaIds: readonly string[]) {
  return {
    error: {
      code: 'PERSONA_NOT_FOUND',
      message: '선택한 페르소나를 찾을 수 없습니다.',
      details: { personaIds: [...personaIds] },
    },
  }
}
