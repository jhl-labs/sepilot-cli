import type { A2UIComponent } from '@sepilotd/core'
export { extractA2UI } from '@sepilotd/core'

export function createTable(title: string, headers: string[], rows: string[][]): A2UIComponent {
  return { type: 'table', title, headers, rows }
}

export function createProgress(title: string, steps: Array<{ label: string; status: 'done' | 'active' | 'pending' }>): A2UIComponent {
  return { type: 'progress', title, steps }
}
