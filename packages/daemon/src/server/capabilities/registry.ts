export interface CapabilityMethod {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH' | 'WS'
  path: string
}

export interface CapabilityInfo {
  name: string
  version: string
  methods: CapabilityMethod[]
  available: boolean
  description?: string
}

const items = new Map<string, CapabilityInfo>()

export function registerCapability(info: CapabilityInfo): void {
  const existing = items.get(info.name)
  if (existing) {
    items.set(info.name, {
      ...existing,
      ...info,
      methods: dedupe([...existing.methods, ...info.methods]),
    })
    return
  }
  items.set(info.name, { ...info, methods: dedupe(info.methods) })
}

export function listCapabilities(): CapabilityInfo[] {
  return Array.from(items.values()).sort((a, b) =>
    a.name.localeCompare(b.name),
  )
}

export function getCapability(name: string): CapabilityInfo | undefined {
  return items.get(name)
}

export function resetCapabilitiesForTests(): void {
  items.clear()
}

function dedupe(methods: CapabilityMethod[]): CapabilityMethod[] {
  const seen = new Set<string>()
  const out: CapabilityMethod[] = []
  for (const m of methods) {
    const k = `${m.method} ${m.path}`
    if (seen.has(k)) continue
    seen.add(k)
    out.push(m)
  }
  return out
}
