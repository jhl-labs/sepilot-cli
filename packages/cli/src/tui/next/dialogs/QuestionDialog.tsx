import type { DaemonPendingQuestion } from '@sepilotd/api-client'
import { Box, Text } from 'ink'
import { useState } from 'react'
import { ControlSafeTextInput } from '../../components/ControlSafeTextInput.js'
import { colors } from '../../theme.js'

export interface QuestionDialogProps {
  question: DaemonPendingQuestion
  width: number
  busy?: boolean
  error?: string | null
  onAnswer(answer: string): void
}

export function QuestionDialog({
  question,
  width,
  busy = false,
  error,
  onAnswer,
}: QuestionDialogProps) {
  const [answer, setAnswer] = useState('')
  const submit = (value: string) => {
    const trimmed = value.trim()
    if (!trimmed || busy) return
    onAnswer(trimmed)
  }

  return (
    <Box
      flexDirection="column"
      width={width}
      borderStyle="round"
      borderColor={colors.warning}
      paddingX={1}
    >
      <Text color={colors.warning} bold>Agent question</Text>
      <Text>{question.prompt}</Text>
      {question.choices?.length ? (
        <Text color={colors.dimText}>{`Choices: ${question.choices.join(' · ')}`}</Text>
      ) : null}
      <Box>
        <Text color={colors.primary}>{'› '}</Text>
        <ControlSafeTextInput
          value={answer}
          placeholder="type an answer…"
          focus={!busy}
          onChange={setAnswer}
          onSubmit={submit}
        />
      </Box>
      {busy ? <Text color={colors.dimText}>sending answer…</Text> : null}
      {error ? <Text color={colors.error}>{error}</Text> : null}
      <Text color={colors.dimText}>enter answer · the run resumes automatically</Text>
    </Box>
  )
}
