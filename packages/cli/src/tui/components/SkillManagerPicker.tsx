import { useEffect, useMemo } from 'react'
import type { DaemonSkill } from '@sepilotd/api-client'
import { Box, Text, useInput } from 'ink'
import { colors } from '../theme.js'
import { isReturnKey } from '../utils/key.js'
import { calculateVisibleWindow } from '../utils/layout.js'
import { getSkillCategoryTitle } from '../utils/skill-store.js'
import { ControlSafeTextInput } from './ControlSafeTextInput.js'

interface SkillManagerPickerProps {
  query: string
  skills: DaemonSkill[]
  selectedIndex: number
  loading: boolean
  togglingSkillId: string | null
  error: string | null
  message: string | null
  maxVisibleItems?: number
  onQueryChange: (value: string) => void
  onSelectIndex: (index: number) => void
  onToggle: (skill: DaemonSkill) => void
  onRun: (skill: DaemonSkill) => void
  onClose: () => void
}

function normalizeSearch(value: string): string {
  return value.replace(/\s+/g, ' ').trim().toLowerCase()
}

function compactText(value: string | null | undefined): string {
  return (value ?? '').replace(/\s+/g, ' ').trim()
}

function formatSource(skill: DaemonSkill): string | null {
  if (!skill.source) return null
  return `${skill.source.type}:${skill.source.ref}`
}

function formatLabels(skill: DaemonSkill): string | null {
  const labels = [
    ...(skill.tools ?? []).slice(0, 3).map((tool) => `tool:${tool}`),
    ...(skill.tags ?? []).slice(0, 4).map((tag) => `#${tag}`),
  ]
  return labels.length > 0 ? labels.join(' ') : null
}

function skillMatches(skill: DaemonSkill, query: string): boolean {
  if (!query) return true
  const source = formatSource(skill) ?? ''
  const haystack = normalizeSearch(
    [
      skill.id,
      skill.name,
      skill.version,
      skill.description,
      getSkillCategoryTitle(skill),
      source,
      ...(skill.tools ?? []),
      ...(skill.tags ?? []),
    ].join(' '),
  )
  return haystack.includes(query)
}

function buildSkillDetailLines(skill: DaemonSkill): string[] {
  const source = formatSource(skill)
  const labels = formatLabels(skill)
  const description = compactText(skill.description)
  return [
    `${skill.id} v${skill.version}`,
    description || null,
    `category ${getSkillCategoryTitle(skill)}`,
    skill.autoDiscovered ? 'scope auto-discovered (managed from SKILL.md)' : null,
    source ? `source ${source}` : null,
    labels,
  ].filter((line): line is string => Boolean(line))
}

function renderSkillRow(skill: DaemonSkill, selected: boolean, toggling: boolean) {
  const enabled = skill.enabled !== false
  const marker = skill.autoDiscovered ? '[A]' : toggling ? '[*]' : enabled ? '[x]' : '[ ]'
  const category = getSkillCategoryTitle(skill)
  const description = compactText(skill.description)

  return (
    <Box key={skill.id} gap={1} width="100%" minWidth={0} height={1} overflow="hidden">
      <Text
        color={selected ? colors.text : enabled ? colors.primary : colors.dimText}
        backgroundColor={selected ? colors.primary : undefined}
        bold={selected}
      >
        {' '}
        {marker}{' '}
      </Text>
      <Box flexGrow={1} flexShrink={1} minWidth={0} height={1} overflow="hidden">
        <Text color={selected ? colors.text : colors.muted} wrap="truncate-end">
          {skill.name} v{skill.version} - {category}
          {description ? ` - ${description}` : ''}
        </Text>
      </Box>
      <Text color={selected ? colors.text : enabled ? colors.primary : colors.dimText}>
        {skill.autoDiscovered ? 'auto' : toggling ? 'saving' : enabled ? 'enabled' : 'disabled'}
      </Text>
    </Box>
  )
}

export function SkillManagerPicker({
  query,
  skills,
  selectedIndex,
  loading,
  togglingSkillId,
  error,
  message,
  maxVisibleItems = 10,
  onQueryChange,
  onSelectIndex,
  onToggle,
  onRun,
  onClose,
}: SkillManagerPickerProps) {
  const normalizedQuery = normalizeSearch(query)
  const filteredSkills = useMemo(
    () => skills.filter((skill) => skillMatches(skill, normalizedQuery)),
    [normalizedQuery, skills],
  )
  const clampedIndex =
    filteredSkills.length === 0
      ? 0
      : Math.max(0, Math.min(selectedIndex, filteredSkills.length - 1))
  const selectedSkill = filteredSkills[clampedIndex] ?? null
  const { start, end } = useMemo(
    () => calculateVisibleWindow(filteredSkills.length, clampedIndex, maxVisibleItems),
    [clampedIndex, filteredSkills.length, maxVisibleItems],
  )
  const visibleSkills = filteredSkills.slice(start, end)
  const detailLines = selectedSkill ? buildSkillDetailLines(selectedSkill) : []
  const busy = Boolean(togglingSkillId)

  useEffect(() => {
    if (clampedIndex !== selectedIndex) {
      onSelectIndex(clampedIndex)
    }
  }, [clampedIndex, onSelectIndex, selectedIndex])

  useInput((input, key) => {
    if (key.escape) {
      onClose()
      return
    }

    if (key.upArrow) {
      onSelectIndex(Math.max(0, clampedIndex - 1))
      return
    }

    if (key.downArrow) {
      onSelectIndex(
        filteredSkills.length === 0 ? 0 : Math.min(filteredSkills.length - 1, clampedIndex + 1),
      )
      return
    }

    if (isReturnKey(input, key)) {
      if (selectedSkill && !selectedSkill.autoDiscovered && !busy) {
        onToggle(selectedSkill)
      }
      return
    }

    if (key.tab && selectedSkill && selectedSkill.enabled !== false) {
      onRun(selectedSkill)
    }
  })

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={colors.primary}
      paddingX={1}
      marginY={1}
      width="100%"
      minWidth={0}
    >
      <Text color={colors.primary} bold>
        Installed Skills
      </Text>
      <Box>
        <Text color={colors.dimText}>search </Text>
        <ControlSafeTextInput
          value={query}
          onChange={onQueryChange}
          placeholder="skill id, tag, source..."
        />
      </Box>
      {message && (
        <Text color={colors.dimText} wrap="truncate-end">
          {message}
        </Text>
      )}
      {loading && <Text color={colors.dimText}>Loading installed skills...</Text>}
      {error && <Text color={colors.error}>{error}</Text>}
      {!loading && !error && filteredSkills.length === 0 && (
        <Text color={colors.dimText}>
          {normalizedQuery
            ? `No installed skills matched "${query.trim()}".`
            : 'No installed skills found.'}
        </Text>
      )}
      {!loading &&
        !error &&
        visibleSkills.map((skill, index) =>
          renderSkillRow(skill, start + index === clampedIndex, togglingSkillId === skill.id),
        )}
      {!loading && !error && selectedSkill && (
        <>
          <Text color={colors.dimText}>Selected Skill</Text>
          {detailLines.map((line) => (
            <Text key={line} color={colors.muted} wrap="truncate-end">
              {line}
            </Text>
          ))}
        </>
      )}
      <Text color={colors.dimText}>Up/Down move Enter toggle managed Tab run Esc close · [A] auto-discovered</Text>
    </Box>
  )
}
