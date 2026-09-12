import type { SkillSource, SkillRef, FetchedSkill } from './types.js'
import { parseSkillMd } from '../loader.js'
import { SkillFetchError } from '../errors.js'
import type { SkillSourceUrlPolicy } from '../source-url-policy.js'
import {
  fetchPublicSkillText,
  type PublicSkillFetchRuntime,
} from './public-http.js'

export interface UrlSourceDeps extends PublicSkillFetchRuntime {
  urlPolicy?: SkillSourceUrlPolicy
}

export class UrlSource implements SkillSource {
  constructor(private deps: UrlSourceDeps = {}) {}

  async fetch(ref: SkillRef): Promise<FetchedSkill[]> {
    if (ref.type !== 'url') throw new SkillFetchError(`UrlSource cannot handle ${ref.type}`)
    const res = await fetchPublicSkillText(ref.url, {
      fetch: this.deps.fetch,
      resolveUrl: this.deps.resolveUrl,
      createDispatcher: this.deps.createDispatcher,
      assertAllowed: (url) => this.deps.urlPolicy?.assertAllowed(url, 'url'),
    })
    if (!res.ok) {
      throw new SkillFetchError(`Failed to fetch ${ref.url}: ${res.status}`)
    }
    const body = res.text
    const last = ref.url.split('/').filter(Boolean).pop() ?? 'skill'
    const fallbackId = last
      .replace(/\.md$/i, '')
      .replace(/[^a-z0-9_-]/gi, '-')
      .toLowerCase()
    const parsed = parseSkillMd(body, fallbackId)
    return [
      {
        metadata: parsed.metadata,
        content: parsed.content,
        source: { type: 'url', ref: ref.url },
      },
    ]
  }
}
