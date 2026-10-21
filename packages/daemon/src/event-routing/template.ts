import { resolvePath } from './match.js'

const TOKEN_PATTERN = /\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}/g
export const TEMPLATE_VALUE_MAX_CHARS = 4000
export const RESULT_VALUE_MAX_CHARS = 16_000
export const DEFAULT_PASSTHROUGH_TEMPLATE = '[{{event.source}}] {{event.type}}'

function stringify(value: unknown): string {
  if (value === undefined || value === null) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  try {
    return JSON.stringify(value) ?? ''
  } catch {
    return ''
  }
}

function cap(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, max)}…` : value
}

export function renderTemplate(
  template: string,
  vars: Record<string, unknown>,
  options: { valueMaxChars?: number; maxCharsByPath?: Record<string, number> } = {},
): string {
  return template
    .replace(TOKEN_PATTERN, (_match, path: string) => {
      const max = options.maxCharsByPath?.[path] ?? options.valueMaxChars ?? TEMPLATE_VALUE_MAX_CHARS
      return cap(stringify(resolvePath(vars, path)), max)
    })
    .trim()
}
