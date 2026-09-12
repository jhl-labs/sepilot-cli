import { Box, Text, useInput } from 'ink'
import { colors } from '../../theme.js'

export interface InfoDialogProps {
  title: string
  body: string
  width: number
  onClose(): void
}

export function InfoDialog({ title, body, width, onClose }: InfoDialogProps) {
  useInput((input, key) => {
    if (key.escape || key.return || input === 'q') onClose()
  })

  return (
    <Box
      flexDirection="column"
      width={width}
      borderStyle="round"
      borderColor={colors.primary}
      paddingX={1}
    >
      <Text color={colors.primary} bold>{title}</Text>
      <Text>{body}</Text>
      <Text color={colors.dimText}>enter/esc/q close</Text>
    </Box>
  )
}
