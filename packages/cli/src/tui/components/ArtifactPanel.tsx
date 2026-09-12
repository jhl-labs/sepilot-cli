import { artifactLabel, artifactPreview } from '@sepilotd/presentation'
import type { DaemonArtifact } from '@sepilotd/api-client'
import { Box, Text } from 'ink'
import { colors, symbols } from '../theme.js'

interface ArtifactPanelProps {
  artifacts: DaemonArtifact[]
  height: number
}

export function ArtifactPanel({ artifacts, height }: ArtifactPanelProps) {
  if (artifacts.length === 0 || height < 3) {
    return null
  }

  const visibleArtifacts = artifacts.slice().reverse().slice(0, Math.max(1, height - 1))

  return (
    <Box
      flexDirection="column"
      height={height}
      borderStyle="single"
      borderColor={colors.border}
      paddingX={1}
      marginBottom={1}
      overflow="hidden"
      width="100%"
      minWidth={0}
    >
      <Box justifyContent="space-between" width="100%" minWidth={0}>
        <Text color={colors.info} bold>Artifacts</Text>
        <Text color={colors.dimText}>{artifacts.length}</Text>
      </Box>
      {visibleArtifacts.map((artifact) => (
        <Text key={artifact.id} wrap="truncate-end">
          <Text color={colors.text}>{artifactLabel(artifact)}</Text>
          <Text color={colors.dimText}> {symbols.separator} {artifact.type}{artifact.language ? `/${artifact.language}` : ''} {symbols.separator} {artifactPreview(artifact, 56)}</Text>
        </Text>
      ))}
    </Box>
  )
}
