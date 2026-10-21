export const MAX_SELECTED_CHAT_SKILLS = 8

export interface ChatSkillSelectionItem {
  id?: string
  name: string
  enabled?: boolean
}

export interface SelectedChatSkillRef {
  name: string
}

export function chatSkillSelectionId(skill: ChatSkillSelectionItem): string {
  return skill.id?.trim() || skill.name.trim()
}

export function normalizeSelectedChatSkillIds(selectedIds: readonly string[]): string[] {
  const normalized: string[] = []
  for (const value of selectedIds) {
    const id = value.trim()
    if (!id) continue
    const previous = normalized.indexOf(id)
    if (previous >= 0) normalized.splice(previous, 1)
    normalized.push(id)
  }
  return normalized.slice(-MAX_SELECTED_CHAT_SKILLS)
}

export function reconcileSelectedChatSkillIds(
  selectedIds: readonly string[],
  skills: readonly ChatSkillSelectionItem[],
): string[] {
  const available = new Set(
    skills
      .filter((skill) => skill.enabled !== false)
      .map(chatSkillSelectionId)
      .filter(Boolean),
  )
  return normalizeSelectedChatSkillIds(selectedIds).filter((id) => available.has(id))
}

export function toggleSelectedChatSkillId(selectedIds: readonly string[], id: string): string[] {
  const normalized = normalizeSelectedChatSkillIds(selectedIds)
  const target = id.trim()
  if (!target) return normalized
  return normalized.includes(target)
    ? normalized.filter((selectedId) => selectedId !== target)
    : normalizeSelectedChatSkillIds([...normalized, target])
}

export function selectedChatSkillRefs(selectedIds: readonly string[]): SelectedChatSkillRef[] {
  return normalizeSelectedChatSkillIds(selectedIds).map((name) => ({ name }))
}
