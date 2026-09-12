// Provider-specific JSON Schema normalization for tool/function definitions.
//
// Different providers accept different subsets of JSON Schema on the wire:
//   - Gemini functionDeclarations reject `$ref`, `oneOf`/`anyOf`/`allOf`,
//     `additionalProperties`, and unusual type unions (HTTP 400 or silently
//     ignored parameters).
//   - Ollama's OpenAI-compatible relay is similarly brittle with `$ref` and
//     schema-composition keywords.
//   - OpenAI / Anthropic tolerate most drafts but still choke on unresolved
//     `$ref` pointing at a `$defs`/`definitions` block that we drop on the wire.
//
// This module inlines `$ref` for every provider (correctness) and, for the
// restrictive providers, collapses composition keywords into a single
// permissive shape while preserving nested objects/arrays/enums/required.
//
// It is deliberately structural: no dataset/model/graphId branching, only the
// provider capability class decides how aggressively the schema is reduced.

export type SchemaProvider = 'openai' | 'anthropic' | 'gemini' | 'ollama'

// Providers that require aggressive reduction (drop composition + open maps).
const RESTRICTIVE: ReadonlySet<SchemaProvider> = new Set<SchemaProvider>(['gemini', 'ollama'])

type JsonObject = Record<string, unknown>

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

// Resolve a local JSON Pointer like `#/$defs/Foo` or `#/definitions/Foo`.
function resolvePointer(root: JsonObject, ref: string): unknown {
  if (!ref.startsWith('#/')) return undefined
  const parts = ref
    .slice(2)
    .split('/')
    .map((p) => p.replace(/~1/g, '/').replace(/~0/g, '~'))
  let cur: unknown = root
  for (const part of parts) {
    if (!isObject(cur) && !Array.isArray(cur)) return undefined
    cur = (cur as JsonObject)[part]
    if (cur === undefined) return undefined
  }
  return cur
}

// Recursively inline `$ref` against the root's definition blocks. `seen`
// guards against self-referential schemas (which we cannot fully inline).
function inlineRefs(node: unknown, root: JsonObject, seen: ReadonlySet<string>): unknown {
  if (Array.isArray(node)) {
    return node.map((item) => inlineRefs(item, root, seen))
  }
  if (!isObject(node)) return node

  const ref = node.$ref
  if (typeof ref === 'string') {
    if (seen.has(ref)) {
      // Cyclic reference — fall back to a permissive object to avoid infinite
      // expansion. The rest of the sibling keys (if any) are still emitted.
      const rest: JsonObject = {}
      for (const [k, v] of Object.entries(node)) {
        if (k === '$ref') continue
        rest[k] = inlineRefs(v, root, seen)
      }
      return { type: 'object', ...rest }
    }
    const target = resolvePointer(root, ref)
    if (isObject(target) || Array.isArray(target)) {
      const nextSeen = new Set(seen)
      nextSeen.add(ref)
      const resolved = inlineRefs(target, root, nextSeen)
      // Merge sibling keys (e.g. description) that sit next to the $ref.
      if (isObject(resolved)) {
        const merged: JsonObject = { ...resolved }
        for (const [k, v] of Object.entries(node)) {
          if (k === '$ref') continue
          merged[k] = inlineRefs(v, root, seen)
        }
        return merged
      }
      return resolved
    }
    // Unresolvable ref — degrade to a permissive object rather than leak it.
    return { type: 'object' }
  }

  const out: JsonObject = {}
  for (const [key, value] of Object.entries(node)) {
    out[key] = inlineRefs(value, root, seen)
  }
  return out
}

// Pick a single permissive `type` when a schema uses oneOf/anyOf/allOf.
function collapseComposition(node: JsonObject): JsonObject {
  const branches = node.oneOf ?? node.anyOf ?? node.allOf
  if (!Array.isArray(branches) || branches.length === 0) return node

  const rest: JsonObject = {}
  for (const [k, v] of Object.entries(node)) {
    if (k === 'oneOf' || k === 'anyOf' || k === 'allOf') continue
    rest[k] = v
  }

  // Prefer an object/array branch (keeps structure) else the first branch.
  const objectBranch = branches.find((b) => isObject(b) && (b as JsonObject).type === 'object')
  const arrayBranch = branches.find((b) => isObject(b) && (b as JsonObject).type === 'array')
  const chosen = objectBranch ?? arrayBranch ?? branches.find((b) => isObject(b))

  if (isObject(chosen)) {
    return { ...(chosen as JsonObject), ...rest }
  }
  // No structural branch — leave a typeless permissive schema.
  return rest.type ? rest : { ...rest }
}

// Reduce a schema tree to what restrictive providers accept.
function reduceForRestrictive(node: unknown): unknown {
  if (Array.isArray(node)) {
    return node.map((item) => reduceForRestrictive(item))
  }
  if (!isObject(node)) return node

  let current: JsonObject = node
  if ('oneOf' in current || 'anyOf' in current || 'allOf' in current) {
    current = collapseComposition(current)
  }

  const out: JsonObject = {}
  for (const [key, value] of Object.entries(current)) {
    // Drop keywords the restrictive providers reject outright.
    if (key === 'additionalProperties' || key === '$schema' || key === '$id') continue
    if (key === '$defs' || key === 'definitions') continue
    out[key] = reduceForRestrictive(value)
  }
  return out
}

// Strip only the definition blocks (already inlined) for tolerant providers.
function stripDefs(node: unknown): unknown {
  if (Array.isArray(node)) return node.map((item) => stripDefs(item))
  if (!isObject(node)) return node
  const out: JsonObject = {}
  for (const [key, value] of Object.entries(node)) {
    if (key === '$defs' || key === 'definitions') continue
    out[key] = stripDefs(value)
  }
  return out
}

/**
 * Normalize a tool inputSchema for a specific provider before it goes on the
 * wire. Inlines `$ref` for everyone; for Gemini/Ollama it additionally
 * collapses oneOf/anyOf/allOf and drops open-map / draft-meta keywords.
 * The input is never mutated.
 */
export function normalizeToolSchema(schema: unknown, provider: SchemaProvider): unknown {
  if (!isObject(schema)) return schema
  const inlined = inlineRefs(schema, schema, new Set())
  if (RESTRICTIVE.has(provider)) {
    return reduceForRestrictive(inlined)
  }
  return stripDefs(inlined)
}
