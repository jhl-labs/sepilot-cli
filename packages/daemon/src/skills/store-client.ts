import type { SkillMetadata } from '@sepilotd/core'

export interface SkillStoreConfig {
  baseUrl: string
  apiKey?: string
}

export interface RemoteSkill {
  metadata: SkillMetadata
  content: string
  downloads: number
  rating: number
}

export class SkillStoreClient {
  private baseUrl: string
  private apiKey?: string

  constructor(config: SkillStoreConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, '')
    this.apiKey = config.apiKey
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { 'Content-Type': 'application/json' }
    if (this.apiKey) h.Authorization = `Bearer ${this.apiKey}`
    return h
  }

  async search(query: string, limit?: number): Promise<SkillMetadata[]> {
    const res = await fetch(`${this.baseUrl}/api/v1/skills/search?q=${encodeURIComponent(query)}&limit=${limit ?? 20}`, { headers: this.headers() })
    if (!res.ok) throw new Error(`Skill store search failed: ${res.status}`)
    const data = await res.json() as { data: SkillMetadata[] }
    return data.data
  }

  async get(id: string): Promise<RemoteSkill> {
    const res = await fetch(`${this.baseUrl}/api/v1/skills/${encodeURIComponent(id)}`, { headers: this.headers() })
    if (!res.ok) throw new Error(`Skill not found: ${id}`)
    const data = await res.json() as { data: RemoteSkill }
    return data.data
  }

  async install(id: string): Promise<RemoteSkill> {
    const skill = await this.get(id)
    // Download count tracked server-side
    return skill
  }

  async publish(metadata: SkillMetadata, content: string): Promise<void> {
    const res = await fetch(`${this.baseUrl}/api/v1/skills`, {
      method: 'POST', headers: this.headers(),
      body: JSON.stringify({ metadata, content }),
    })
    if (!res.ok) throw new Error(`Skill publish failed: ${res.status}`)
  }

  async list(page?: number): Promise<{ skills: SkillMetadata[]; total: number }> {
    const res = await fetch(`${this.baseUrl}/api/v1/skills?page=${page ?? 1}`, { headers: this.headers() })
    if (!res.ok) throw new Error(`Skill store list failed: ${res.status}`)
    return res.json() as Promise<{ skills: SkillMetadata[]; total: number }>
  }
}
