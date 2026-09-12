import { Box, Text } from 'ink'
import { loadCliVersion } from '../../entrypoint.js'
import { colors } from '../theme.js'

const LARGE_MARK = [
  '███████╗███████╗██████╗ ██╗██╗      ██████╗ ████████╗',
  '██╔════╝██╔════╝██╔══██╗██║██║     ██╔═══██╗╚══██╔══╝',
  '███████╗█████╗  ██████╔╝██║██║     ██║   ██║   ██║   ',
  '╚════██║██╔══╝  ██╔═══╝ ██║██║     ██║   ██║   ██║   ',
  '███████║███████╗██║     ██║███████╗╚██████╔╝   ██║   ',
  '╚══════╝╚══════╝╚═╝     ╚═╝╚══════╝ ╚═════╝    ╚═╝   ',
]

/** A terminal wordmark that remains usable in narrow terminal windows. */
export function BrandMark({ width }: { width: number }) {
  const useLargeMark = width >= 64
  const version = loadCliVersion()

  return (
    <Box flexDirection="column" alignItems="center">
      {useLargeMark ? LARGE_MARK.map((line) => (
        <Text key={line} color={colors.primary} bold>{line}</Text>
      )) : (
        <>
          <Text color={colors.primary} bold>╭── S ──╮</Text>
          <Text color={colors.primary} bold>│ SEPILOT │</Text>
          <Text color={colors.primary} bold>╰───────╯</Text>
        </>
      )}
      <Text color={colors.dimText}>v{version}</Text>
    </Box>
  )
}
