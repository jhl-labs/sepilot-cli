import { Box, Text } from 'ink'
import type { DebateDecision, DebateRoundSummary } from '@sepilotd/api-client'

interface DebatePanelProps {
  rounds: DebateRoundSummary[]
  height?: number
  maxRows?: number
}

const DECISION_COLOR: Record<DebateDecision, string> = {
  accept: 'green',
  revise: 'yellow',
  reject: 'red',
}

const DECISION_GLYPH: Record<DebateDecision, string> = {
  accept: '✓',
  revise: '↺',
  reject: '✗',
}

function truncate(text: string, max = 80): string {
  if (text.length <= max) return text
  return `${text.slice(0, max - 1)}…`
}

export function DebatePanel({
  rounds,
  height,
  maxRows = 4,
}: DebatePanelProps) {
  const recent = [...rounds].slice(-maxRows).reverse()
  if (recent.length === 0) return null
  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor="magenta"
      paddingX={1}
      height={height}
      width="100%"
      minWidth={0}
    >
      <Text color="magenta" bold>
        Debate
      </Text>
      {recent.map((round) => {
        const proposer = round.entries.find((e) => e.role === 'proposer')
        const critic = round.entries.find((e) => e.role === 'critic')
        return (
          <Box key={round.roundId} flexDirection="column" marginTop={1} width="100%" minWidth={0}>
            <Box width="100%" minWidth={0}>
              <Text color={DECISION_COLOR[round.finalDecision]} bold>
                {DECISION_GLYPH[round.finalDecision]} {round.finalDecision.toUpperCase()}{' '}
              </Text>
              <Box flexGrow={1} flexShrink={1} minWidth={0}>
                <Text color="gray" wrap="truncate-end">
                  {truncate(round.topic, 60)}
                </Text>
              </Box>
            </Box>
            {proposer
              ? (
                  <Text color="cyan" wrap="truncate-end">
                    proposer: {truncate(proposer.content)}
                  </Text>
                )
              : null}
            {critic
              ? (
                  <Text color="yellow" wrap="truncate-end">
                    critic: {truncate(critic.content)}
                  </Text>
                )
              : null}
            {round.rationale
              ? (
                  <Text wrap="truncate-end">
                    final: {truncate(round.rationale)}
                  </Text>
                )
              : null}
          </Box>
        )
      })}
    </Box>
  )
}
