import type { DaemonArtifact } from '@sepilotd/api-client'
import { artifactLabel, artifactPreview } from '@sepilotd/presentation'
import { Box, Text, useInput } from 'ink'
import { useMemo, useState } from 'react'
import { colors } from '../../theme.js'

export interface ArtifactDialogProps {
  artifacts: DaemonArtifact[]
  width: number
  onCopy(artifact: DaemonArtifact): Promise<boolean>
  onClose(): void
}

export function ArtifactDialog({ artifacts, width, onCopy, onClose }: ArtifactDialogProps) {
  const ordered = useMemo(() => artifacts.slice().reverse(), [artifacts])
  const [index, setIndex] = useState(0)
  const [message, setMessage] = useState<string | null>(null)
  const selectedIndex = Math.min(index, Math.max(0, ordered.length - 1))
  const selected = ordered[selectedIndex]

  useInput((input, key) => {
    if (key.upArrow) setIndex((value) => Math.max(0, value - 1))
    else if (key.downArrow) setIndex((value) => Math.min(ordered.length - 1, value + 1))
    else if (key.escape || input === 'q') onClose()
    else if ((key.return || input === 'c') && selected) {
      setMessage('Copying…')
      void onCopy(selected).then((copied) => setMessage(copied ? `Copied ${artifactLabel(selected)}.` : null))
    }
  })

  return (
    <Box flexDirection="column" width={width} borderStyle="round" borderColor={colors.info} paddingX={1}>
      <Text color={colors.info} bold>{`Artifacts · ${artifacts.length}`}</Text>
      {ordered.length === 0 ? <Text color={colors.dimText}>No saved artifacts in the current session.</Text> : null}
      {ordered.slice(0, 10).map((artifact, optionIndex) => (
        <Text key={artifact.id} color={optionIndex === selectedIndex ? colors.primary : colors.text} wrap="truncate-end">
          {`${optionIndex === selectedIndex ? '› ' : '  '}${artifactLabel(artifact)} · ${artifact.type}${artifact.language ? `/${artifact.language}` : ''}`}
        </Text>
      ))}
      {selected ? (
        <Box flexDirection="column" marginTop={1}>
          <Text color={colors.dimText}>Preview</Text>
          <Text wrap="truncate-end">{artifactPreview(selected, Math.max(80, width * 3))}</Text>
        </Box>
      ) : null}
      {message ? <Text color={colors.success}>{message}</Text> : null}
      <Text color={colors.dimText}>↑↓ move · enter/c copy · esc/q close</Text>
    </Box>
  )
}
