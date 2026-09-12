import type { SemanticMemoryStore } from './types.js'

type AuditListener = Parameters<SemanticMemoryStore['subscribeAudit']>[0]
type FileRegistry = Parameters<NonNullable<SemanticMemoryStore['configureFileMemory']>>[0]
type Generation = { store: SemanticMemoryStore; active: number; retired: boolean; unsubscribe: () => void }
const replacements = new WeakMap<SemanticMemoryStore, (next: SemanticMemoryStore) => Promise<void>>()

/** Keep the identity captured by tools, routers and background services stable
 * across configuration changes. An in-flight call owns its old DB until it
 * settles; subsequent calls use the replacement. */
export function rebindableSemanticStore(initial: SemanticMemoryStore): SemanticMemoryStore {
  const listeners = new Set<AuditListener>()
  let fileRegistry: FileRegistry | undefined
  let closed = false
  const generation = (store: SemanticMemoryStore): Generation => ({
    store, active: 0, retired: false,
    unsubscribe: store.subscribeAudit((entry) => {
      for (const listener of [...listeners]) listener(entry)
    }),
  })
  let current = generation(initial)
  const dispose = (entry: Generation) => {
    if (!entry.retired || entry.active !== 0) return
    entry.unsubscribe()
    entry.store.close()
  }
  const close = () => {
    if (closed) return
    closed = true
    listeners.clear()
    current.retired = true
    dispose(current)
  }
  const facade = new Proxy({} as SemanticMemoryStore, {
    get(_target, key) {
      if (key === 'close') return close
      if (key === 'subscribeAudit') return (listener: AuditListener) => {
        if (closed) throw new Error('Semantic memory store is closed')
        listeners.add(listener)
        return () => { listeners.delete(listener) }
      }
      const value: unknown = Reflect.get(current.store, key)
      if (typeof value !== 'function') return value
      return (...args: unknown[]) => {
        if (closed) throw new Error('Semantic memory store is closed')
        const entry = current
        const method = Reflect.get(entry.store, key) as (...values: unknown[]) => unknown
        entry.active += 1
        const release = () => { entry.active -= 1; dispose(entry) }
        try {
          const result = method.apply(entry.store, args)
          if (key === 'configureFileMemory') fileRegistry = args[0] as FileRegistry
          if (result instanceof Promise) return result.finally(release)
          release()
          return result
        } catch (error) {
          release()
          throw error
        }
      }
    },
  })
  replacements.set(facade, async (next) => {
    let replacement: Generation
    try {
      if (closed) throw new Error('Semantic memory store is closed')
      if (fileRegistry) await next.configureFileMemory?.(fileRegistry)
      if (closed) throw new Error('Semantic memory store is closed')
      replacement = generation(next)
    } catch (error) {
      next.close()
      throw error
    }
    const previous = current
    current = replacement
    previous.retired = true
    dispose(previous)
  })
  return facade
}

export async function replaceSemanticStore(
  previous: SemanticMemoryStore,
  next: SemanticMemoryStore,
): Promise<SemanticMemoryStore> {
  const replace = replacements.get(previous)
  if (replace) {
    await replace(next)
    return previous
  }
  // Lightweight/injected runtimes may not have assembled a storage layer.
  const facade = rebindableSemanticStore(next)
  previous?.close()
  return facade
}
