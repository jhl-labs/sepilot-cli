import { Box, Text } from 'ink'

export interface TodoProgressItem {
  content: string
  status: string
}

interface TodoProgressPanelProps {
  todos: TodoProgressItem[]
  maxRows?: number
}

const STATUS_GLYPH: Record<string, string> = {
  pending: '○',
  in_progress: '◐',
  completed: '●',
  cancelled: '⊘',
}

const STATUS_COLOR: Record<string, string> = {
  pending: 'gray',
  in_progress: 'cyan',
  completed: 'green',
  cancelled: 'yellow',
}

/**
 * Live todo checklist for the current run, fed by the `state_board` stream
 * frame (structured items from the agent's own todowrite list — no text
 * parsing). Gives the user a per-step "what is it doing now / how much is
 * left" view during long runs.
 */
export function TodoProgressPanel({ todos, maxRows = 8 }: TodoProgressPanelProps) {
  if (todos.length === 0) {
    return null
  }
  const visible = todos.slice(0, maxRows)
  const overflow = todos.length - visible.length
  const done = todos.filter((todo) => todo.status === 'completed').length
  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor="cyan"
      paddingX={1}
      width="100%"
      minWidth={0}
    >
      <Text color="cyan" bold>
        Todo {done}/{todos.length}
      </Text>
      {visible.map((todo, index) => (
        <Text key={`${index}-${todo.content}`} wrap="truncate-end">
          <Text color={STATUS_COLOR[todo.status] ?? 'white'}>
            {STATUS_GLYPH[todo.status] ?? '·'}
          </Text>
          <Text
            color={todo.status === 'in_progress' ? 'cyan' : todo.status === 'completed' ? 'gray' : 'white'}
            dimColor={todo.status === 'completed'}
          >
            {' '}
            {todo.content}
          </Text>
        </Text>
      ))}
      {overflow > 0 && <Text dimColor>(+{overflow} more)</Text>}
    </Box>
  )
}
