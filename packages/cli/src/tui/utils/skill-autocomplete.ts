import type { DaemonSkill } from '@sepilotd/api-client'
import { fuzzyScore } from './fuzzy.js'

const MAX_SKILL_AUTOCOMPLETE_CANDIDATES = 8
const SKILL_REFERENCE_PATTERN = /(^|\s)@skills?:([^\s]*)$/i

export interface SkillAutocompleteReference {
  raw: string
  query: string
  start: number
  end: number
}

export function findActiveSkillReference(input: string): SkillAutocompleteReference | null {
  const match = input.match(SKILL_REFERENCE_PATTERN)
  if (!match) return null

  const boundary = match[1] ?? ''
  const query = match[2] ?? ''
  const start = input.length - match[0].length + boundary.length
  const raw = input.slice(start)

  return {
    raw,
    query,
    start,
    end: input.length,
  }
}

export function applySkillCompletion(
  input: string,
  reference: SkillAutocompleteReference,
  skillId: string,
): string {
  const before = input.slice(0, reference.start).trim()
  const after = input.slice(reference.end).trim()
  const prompt = [before, after].filter(Boolean).join(' ')
  return prompt ? `/run ${skillId} ${prompt}` : `/run ${skillId} `
}

function skillSearchText(skill: DaemonSkill): string {
  return [
    skill.id,
    skill.name,
    skill.description,
    ...(skill.tags ?? []),
    ...(skill.tools ?? []),
  ].join(' ')
}

function skillSortLabel(skill: DaemonSkill): string {
  return `${skill.name || skill.id} ${skill.id}`.toLowerCase()
}

export function filterSkillAutocompleteCandidates(
  skills: readonly DaemonSkill[],
  query: string,
  limit = MAX_SKILL_AUTOCOMPLETE_CANDIDATES,
): DaemonSkill[] {
  const runnableSkills = skills.filter((skill) => skill.enabled !== false)
  const normalizedQuery = query.trim().toLowerCase()

  if (!normalizedQuery) {
    return runnableSkills
      .slice()
      .sort((left, right) => skillSortLabel(left).localeCompare(skillSortLabel(right)))
      .slice(0, limit)
  }

  return runnableSkills
    .map((skill) => {
      const target = skillSearchText(skill).toLowerCase()
      const score = fuzzyScore(normalizedQuery, target)
      if (score === null) return null
      const idStartsWith = skill.id.toLowerCase().startsWith(normalizedQuery)
      const nameStartsWith = skill.name.toLowerCase().startsWith(normalizedQuery)
      return {
        skill,
        score: score.score + (idStartsWith ? 1000 : 0) + (nameStartsWith ? 800 : 0),
      }
    })
    .filter((entry): entry is { skill: DaemonSkill; score: number } => entry !== null)
    .sort((left, right) => {
      if (left.score !== right.score) return right.score - left.score
      return skillSortLabel(left.skill).localeCompare(skillSortLabel(right.skill))
    })
    .slice(0, limit)
    .map((entry) => entry.skill)
}
