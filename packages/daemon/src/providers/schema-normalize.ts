export type SchemaProvider = 'openai' | 'anthropic' | 'gemini' | 'ollama'

const RESTRICTIVE: ReadonlySet<SchemaProvider> = new Set<SchemaProvider>(['gemini', 'ollama'])

type JsonObject = Record<string, unknown>

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

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

function inlineRefs(node: unknown, root: JsonObject, seen: ReadonlySet<string>): unknown {
  if (Array.isArray(node)) {
    return node.map((item) => inlineRefs(item, root, seen))
  }
  if (!isObject(node)) return node

  const ref = node.$ref
  if (typeof ref === 'string') {
    if (seen.has(ref)) {
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
    return { type: 'object' }
  }

  const out: JsonObject = {}
  for (const [key, value] of Object.entries(node)) {
    out[key] = inlineRefs(value, root, seen)
  }
  return out
}

function collapseComposition(node: JsonObject): JsonObject {
  const branches = node.oneOf ?? node.anyOf ?? node.allOf
  if (!Array.isArray(branches) || branches.length === 0) return node

  const rest: JsonObject = {}
  for (const [k, v] of Object.entries(node)) {
    if (k === 'oneOf' || k === 'anyOf' || k === 'allOf') continue
    rest[k] = v
  }

  const objectBranch = branches.find((b) => isObject(b) && (b as JsonObject).type === 'object')
  const arrayBranch = branches.find((b) => isObject(b) && (b as JsonObject).type === 'array')
  const chosen = objectBranch ?? arrayBranch ?? branches.find((b) => isObject(b))

  if (isObject(chosen)) {
    return { ...(chosen as JsonObject), ...rest }
  }
  return rest.type ? rest : { ...rest }
}

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
    if (key === 'additionalProperties' || key === '$schema' || key === '$id') continue
    if (key === '$defs' || key === 'definitions') continue
    out[key] = reduceForRestrictive(value)
  }
  return out
}

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

export function normalizeToolSchema(schema: unknown, provider: SchemaProvider): unknown {
  if (!isObject(schema)) return schema
  const inlined = inlineRefs(schema, schema, new Set())
  if (RESTRICTIVE.has(provider)) {
    return reduceForRestrictive(inlined)
  }
  return stripDefs(inlined)
}
