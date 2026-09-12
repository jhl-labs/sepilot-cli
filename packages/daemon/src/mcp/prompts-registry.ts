import type { PromptDescriptor } from './client.js'

export class McpPromptsRegistry {
  private entries = new Map<string, PromptDescriptor[]>()

  set(server: string, prompts: PromptDescriptor[]): void {
    this.entries.set(server, prompts)
  }

  remove(server: string): void {
    this.entries.delete(server)
  }

  list(server: string): PromptDescriptor[] | null {
    return this.entries.get(server) ?? null
  }

  listAll(): Array<{ server: string; prompts: PromptDescriptor[] }> {
    return Array.from(this.entries.entries()).map(([server, prompts]) => ({ server, prompts }))
  }

  resolveCallsite(ref: string): { server: string; prompt: string } | null {
    if (ref.includes('/')) {
      const [server, prompt] = ref.split('/', 2)
      const prompts = this.entries.get(server)
      if (!prompts?.some((p) => p.name === prompt)) return null
      return { server, prompt }
    }
    for (const [server, prompts] of this.entries) {
      if (prompts.some((p) => p.name === ref)) return { server, prompt: ref }
    }
    return null
  }
}
