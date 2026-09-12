export interface Frontmatter {
  data: Record<string, unknown>
  body: string
}

export function parseFrontmatter(raw: string): Frontmatter {
  const m = /^---\n([\s\S]*?)\n---\n?/.exec(raw)
  if (!m) return { data: {}, body: raw }
  const body = raw.slice(m[0].length)
  const data: Record<string, unknown> = {}
  for (const line of m[1].split('\n')) {
    const mm = /^([A-Za-z0-9_]+):\s*(.*)$/.exec(line)
    if (!mm) continue
    const key = mm[1]
    const value = mm[2].trim()
    if (value.startsWith('[') && value.endsWith(']')) {
      data[key] = value
        .slice(1, -1)
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
    } else if (value === 'true' || value === 'false') {
      data[key] = value === 'true'
    } else if (/^-?\d+$/.test(value)) {
      data[key] = Number(value)
    } else {
      data[key] = value.replace(/^['"]|['"]$/g, '')
    }
  }
  return { data, body }
}
