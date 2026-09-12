import type { SepilotReadableAppSummary } from './apps.js'

// These are explicit product namespaces, never natural-language synonyms.
const RESERVED = new Set(['app', 'skill', 'wiki', 'task'])
const SHORT_NAME = /^[a-z][a-z0-9_-]{0,63}$/u

export function appReferenceAliases(app: Pick<SepilotReadableAppSummary, 'id' | 'kind'>): string[] {
  return [...new Set([app.id, app.kind])]
    .filter(name => SHORT_NAME.test(name) && !RESERVED.has(name))
    .map(name => `$${name}`)
}

export function canonicalAppReference(id: string): string {
  return `$app:${id}`
}

function referenceProse(input: string): string {
  const literal = /```[^]*?(?:```|$)|~~~[^]*?(?:~~~|$)|(`+)[^]*?(?:\1|$)|(?<![\p{L}\p{N}])"[^"\n]*(?:"|$)|(?<![\p{L}\p{N}])'[^'\n]*(?:'|$)|“[^”\n]*(?:”|$)|‘[^’\n]*(?:’|$)/gu
  return input.replace(literal, text => ' '.repeat(text.length))
}

/** Let a composer search partial names without activating inside quoted/code content. */
export function isAppReferencePosition(input: string, offset: number): boolean {
  return referenceProse(input)[offset] === '$' && (offset === 0 || /[\s([{]/u.test(input[offset - 1]!))
}

/** Parse explicit syntax only. Quoted examples, code and escaped dollars are data. */
export function parseAppReferences(input: string): string[] {
  const prose = referenceProse(input)
  const references = new Set<string>()
  const token = /(?:^|[\s([{])\$(app:[a-z0-9][a-z0-9._-]{0,63}|[a-z][a-z0-9_-]{0,63})(?=$|[\s,!?;:)\]}]|\.(?=$|\s))/gu
  for (const match of prose.matchAll(token)) {
    const name = match[1]!
    if (!RESERVED.has(name)) references.add(`$${name}`)
  }
  return [...references]
}

export interface ResolvedAppReference {
  reference: string
  status: 'resolved' | 'ambiguous' | 'unavailable'
  appIds: string[]
}

/** Exact catalog identity resolution; never guess a target from request wording. */
export function resolveAppReferences(input: string, apps: readonly SepilotReadableAppSummary[]): ResolvedAppReference[] {
  return parseAppReferences(input).map(reference => {
    const appIds = apps.filter(app => reference === canonicalAppReference(app.id)
      || (!reference.startsWith('$app:') && appReferenceAliases(app).includes(reference)))
      .map(app => app.id)
    return { reference, appIds, status: appIds.length === 1 ? 'resolved' : appIds.length ? 'ambiguous' : 'unavailable' }
  })
}
