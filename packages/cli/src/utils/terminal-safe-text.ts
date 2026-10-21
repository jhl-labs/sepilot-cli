const ESCAPE_SEQUENCE = /\u001b(?:\[[0-?]*[ -/]*[@-~]|[\]PX^_][\s\S]*?(?:\u0007|\u001b\\|$)|[@-Z\\-_])/g
const C1_SEQUENCE = /\u009b[0-?]*[ -/]*[@-~]|[\u0090\u0098\u009d\u009e\u009f][\s\S]*?(?:\u0007|\u009c|$)/g
const CONTROL_CHARACTER = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/g

export function terminalSafeText(text: string): string {
  if (!/[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/.test(text)) return text
  return text
    .replace(ESCAPE_SEQUENCE, '')
    .replace(C1_SEQUENCE, '')
    .replace(/\r\n?/g, '\n')
    .replace(CONTROL_CHARACTER, '')
}

export function terminalSafeValue<T>(value: T): T {
  if (typeof value === 'string') return terminalSafeText(value) as T
  if (Array.isArray(value)) {
    let changed = false
    const next = value.map((item) => {
      const safe = terminalSafeValue(item)
      if (safe !== item) changed = true
      return safe
    })
    return (changed ? next : value) as T
  }
  if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype) {
    let changed = false
    const next: Record<string, unknown> = {}
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      const safe = terminalSafeValue(item)
      if (safe !== item) changed = true
      next[key] = safe
    }
    return (changed ? next : value) as T
  }
  return value
}
