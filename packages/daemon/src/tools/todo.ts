import { randomUUID } from 'node:crypto'
import type { TodoItem, TodoListEvent } from '@sepilotd/core'
import type { ToolDefinitionRuntime, ToolResult, ToolExecutionContext } from './registry.js'
import { hashProjectPath } from '../memory/scope.js'
import type { TaskBoardStore } from '../memory/task-board.js'

export interface TodoToolDeps {
  appendEvent: (sessionId: string, event: TodoListEvent) => Promise<void>
  /**
   * Optional cross-session task board (P022-T6). When present and the tool call
   * carries a cwd, the todo list is mirrored into the project-scoped durable
   * store so a later session can reload the still-open tasks.
   */
  taskBoard?: TaskBoardStore
}

const TODO_STATUSES = new Set<string>([
  'pending',
  'in_progress',
  'completed',
  'blocked',
  'cancelled',
])
const CLOSED_TODO_STATUSES = new Set<TodoItem['status']>(['completed', 'cancelled'])

/**
 * Structurally validate a `todowrite` items payload into TodoItem[].
 * Used by the tool itself and by the agent loop, which mirrors successful
 * todowrite calls into `AgentState.todoList` (first-class loop state) instead
 * of scraping the tool's text output. Returns null when the payload is not a
 * well-formed todo list.
 */
export function parseTodoItems(value: unknown): TodoItem[] | null {
  if (!Array.isArray(value)) return null
  const items: TodoItem[] = []
  for (const entry of value) {
    if (typeof entry !== 'object' || entry === null) return null
    const record = entry as Record<string, unknown>
    const id = typeof record.id === 'string' ? record.id.trim() : ''
    const contentValue =
      record.content ?? record.title ?? record.text ?? record.task ?? record.description
    const status = record.status
    if (!id || typeof contentValue !== 'string' || !contentValue.trim()) return null
    if (typeof status !== 'string' || !TODO_STATUSES.has(status)) return null
    items.push({ id, content: contentValue.trim(), status: status as TodoItem['status'] })
  }
  return items
}

function formatTodoOutput(items: TodoItem[]): string {
  const incomplete = items.filter((item) => !CLOSED_TODO_STATUSES.has(item.status))
  const lines = [`todo list updated (${items.length} item(s))`]
  if (incomplete.length === 0) {
    lines.push(
      items.every((item) => item.status === 'completed')
        ? '[todo-list] all items completed'
        : '[todo-list] all items closed',
    )
    return lines.join('\n')
  }

  lines.push('[todo-list]', 'incomplete:')
  for (const item of incomplete.slice(0, 8)) {
    lines.push(`- [${item.status}] ${item.content}`)
  }
  if (incomplete.length > 8) {
    lines.push(`- ... ${incomplete.length - 8} more incomplete item(s)`)
  }
  lines.push('[/todo-list]')
  return lines.join('\n')
}

export function createTodoWriteTool(deps: TodoToolDeps): ToolDefinitionRuntime {
  return {
    name: 'todowrite',
    description:
      'Replace the current session todo list. Use this to plan and mark progress. ' +
      'Items are {id, content, status} where status is pending | in_progress | completed | blocked | cancelled.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'todo-list' },
    inputSchema: {
      type: 'object',
      properties: {
        items: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              id: { type: 'string' },
              content: { type: 'string' },
              title: { type: 'string', description: 'Accepted alias for content when a model emits title instead of content.' },
              text: { type: 'string', description: 'Accepted alias for content.' },
              task: { type: 'string', description: 'Accepted alias for content.' },
              status: {
                type: 'string',
                enum: ['pending', 'in_progress', 'completed', 'blocked', 'cancelled'],
              },
            },
            required: ['id', 'status'],
          },
        },
      },
      required: ['items'],
    },
    async execute(input, ctx?: ToolExecutionContext): Promise<ToolResult> {
      const start = Date.now()
      const items = input.items
      if (!Array.isArray(items)) {
        return {
          output: 'items must be an array',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      if (!ctx?.sessionId) {
        return {
          output: 'no session context',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      const parsed = parseTodoItems(items)
      if (!parsed) {
        return {
          output: 'items must be an array of {id, content, status}; allowed statuses are pending, in_progress, completed, blocked, cancelled; title/text/task are accepted aliases for content',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      const event: TodoListEvent = {
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        type: 'todo_list',
        items: parsed,
      }
      await deps.appendEvent(ctx.sessionId, event)
      // Mirror into the durable project task board so tasks survive the session
      // (best-effort — a task-board failure must not fail the todo write).
      if (deps.taskBoard && ctx.cwd) {
        try {
          await deps.taskBoard.upsertTasks(hashProjectPath(ctx.cwd), event.items)
        } catch {
          // Durable mirroring is best-effort; the in-session list already saved.
        }
      }
      return {
        output: formatTodoOutput(event.items),
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }
}
