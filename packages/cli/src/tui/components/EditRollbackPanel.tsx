import { Box, Text } from 'ink'
import type { EditCheckpointSummary } from '@sepilotd/api-client'

interface EditRollbackPanelProps {
  checkpoints: EditCheckpointSummary[]
  height?: number
  maxRows?: number
}

export function EditRollbackPanel({
  checkpoints,
  height,
  maxRows = 6,
}: EditRollbackPanelProps) {
  const recent = [...checkpoints].slice(-maxRows).reverse()
  if (recent.length === 0) return null
  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor="yellow"
      paddingX={1}
      height={height}
      width="100%"
      minWidth={0}
    >
      <Text color="yellow" bold>
        Edit checkpoints
      </Text>
      {recent.map((cp) => {
        const isRevert = cp.status === 'reverted'
        const isOpen = cp.status === 'open'
        const label = isOpen ? 'open' : isRevert ? '↶ revert' : '✓ commit'
        const color = isOpen ? 'yellow' : isRevert ? 'red' : 'green'
        return (
          <Box key={cp.checkpointId} flexDirection="column" width="100%" minWidth={0}>
            <Box width="100%" minWidth={0}>
              <Text color={color} bold>
                {label}{' '}
              </Text>
              <Box flexGrow={1} flexShrink={1} minWidth={0}>
                <Text color="gray" wrap="truncate-end">
                  {cp.label} · {cp.files.length} file
                  {cp.files.length === 1 ? '' : 's'}
                </Text>
              </Box>
            </Box>
            {isRevert && cp.revertReason
              ? (
                  <Text color="gray" wrap="truncate-end">
                    {' '}↪ {cp.revertReason}
                  </Text>
                )
              : null}
            {cp.files.slice(0, 3).map((f) => (
              <Text key={`${cp.checkpointId}-${f.path}`} wrap="truncate-end" color="gray">
                {' '}· {f.path}
              </Text>
            ))}
            {cp.files.length > 3
              ? <Text color="gray"> · +{cp.files.length - 3} more</Text>
              : null}
          </Box>
        )
      })}
    </Box>
  )
}
