import { Box, Text, useInput } from 'ink'
import { useState } from 'react'
import { colors } from '../../theme.js'

export interface ChoiceOption {
  value: string
  label: string
}

export interface ChoiceDialogProps {
  title: string
  options: ChoiceOption[]
  current: string
  width: number
  onSelect(value: string): void
  onClose(): void
}

export function ChoiceDialog({
  title,
  options,
  current,
  width,
  onSelect,
  onClose,
}: ChoiceDialogProps) {
  const initialIndex = Math.max(0, options.findIndex(({ value }) => value === current))
  const [selectedIndex, setSelectedIndex] = useState(initialIndex)

  useInput((input, key) => {
    if (key.upArrow) setSelectedIndex((index) => Math.max(0, index - 1))
    else if (key.downArrow) setSelectedIndex((index) => Math.min(options.length - 1, index + 1))
    else if (key.escape || input === 'q') onClose()
    else if (key.return && options[selectedIndex]) onSelect(options[selectedIndex].value)
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
      {options.map((option, index) => (
        <Text key={option.value} color={index === selectedIndex ? colors.primary : colors.text}>
          {`${index === selectedIndex ? '› ' : '  '}${option.label}${option.value === current ? ' [current]' : ''}`}
        </Text>
      ))}
      <Text color={colors.dimText}>↑↓ move · enter select · esc close</Text>
    </Box>
  )
}
