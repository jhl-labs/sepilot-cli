import type { DaemonProject } from '@sepilotd/api-client'
import { Box, Text, useInput } from 'ink'
import { useMemo, useState } from 'react'
import { ControlSafeTextInput } from '../../components/ControlSafeTextInput.js'
import { colors } from '../../theme.js'

export interface ProjectDialogProps {
  projects: DaemonProject[]
  currentProjectId: string | null
  width: number
  loading: boolean
  error: string | null
  onSelect(project: DaemonProject | null): void
  onClose(): void
}

export function ProjectDialog({
  projects,
  currentProjectId,
  width,
  loading,
  error,
  onSelect,
  onClose,
}: ProjectDialogProps) {
  const [query, setQuery] = useState('')
  const [index, setIndex] = useState(0)
  const options = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    const matching = normalized
      ? projects.filter((project) =>
          `${project.name} ${project.description} ${project.workingDirectory ?? ''}`
            .toLowerCase()
            .includes(normalized),
        )
      : projects
    return [null, ...matching]
  }, [projects, query])
  const selectedIndex = Math.min(index, Math.max(0, options.length - 1))

  useInput((_input, key) => {
    if (key.upArrow) setIndex((value) => Math.max(0, value - 1))
    else if (key.downArrow) setIndex((value) => Math.min(options.length - 1, value + 1))
    else if (key.escape) onClose()
    else if (key.return) onSelect(options[selectedIndex] ?? null)
  })

  return (
    <Box flexDirection="column" width={width} borderStyle="round" borderColor={colors.primary} paddingX={1}>
      <Text color={colors.primary} bold>Projects</Text>
      <Box>
        <Text color={colors.dimText}>search </Text>
        <ControlSafeTextInput
          value={query}
          onChange={(value) => { setQuery(value); setIndex(0) }}
          placeholder="name, description, workspace…"
          focus={!loading}
        />
      </Box>
      {loading ? <Text color={colors.dimText}>Loading projects…</Text> : null}
      {error ? <Text color={colors.error}>{error}</Text> : null}
      {!loading && !error ? options.slice(0, 12).map((project, optionIndex) => {
        const selected = optionIndex === selectedIndex
        const current = project?.id === currentProjectId || (!project && !currentProjectId)
        return (
          <Text key={project?.id ?? 'none'} color={selected ? colors.primary : colors.text}>
            {`${selected ? '› ' : '  '}${project?.name ?? 'No project'}${current ? ' [current]' : ''}${project?.workingDirectory ? ` · ${project.workingDirectory}` : ''}`}
          </Text>
        )
      }) : null}
      <Text color={colors.dimText}>↑↓ move · enter select · esc close</Text>
    </Box>
  )
}
