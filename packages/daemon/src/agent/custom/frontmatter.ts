import { parse as parseYaml } from 'yaml'

export interface Frontmatter {
  data: Record<string, unknown>
  body: string
}

export function parseFrontmatter(raw: string): Frontmatter {
  const m = /^\uFEFF?---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(raw)
  if (!m) {
    if (/^\uFEFF?---(?:\r?\n|$)/.test(raw)) throw new Error('Unterminated agent/command frontmatter')
    return { data: {}, body: raw }
  }
  const body = raw.slice(m[0].length)
  const data: unknown = parseYaml(m[1], { maxAliasCount: 20, uniqueKeys: true }) ?? {}
  if (typeof data !== 'object' || Array.isArray(data)) throw new Error('Frontmatter must be a YAML mapping')
  return { data: data as Record<string, unknown>, body }
}
