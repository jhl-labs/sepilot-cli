import type { A2UIComponent } from '@sepilotd/core'
export { extractA2UI } from '@sepilotd/core'

/** Generate an A2UI table from structured data */
export function createTable(title: string, headers: string[], rows: string[][]): A2UIComponent {
  return { type: 'table', title, headers, rows }
}

/** Generate an A2UI progress tracker */
export function createProgress(title: string, steps: Array<{ label: string; status: 'done' | 'active' | 'pending' }>): A2UIComponent {
  return { type: 'progress', title, steps }
}
