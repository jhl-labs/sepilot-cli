import type { Deps } from '../nodes.js'
import { buildResearcherGraph } from './researcher.js'

const deepWebResearchPrompt = [
  'You are a deep web research agent.',
  'Search broadly, cross-check findings across multiple sources, and separate directly supported facts from inference.',
  'Do not finalize until you have an explicit verification summary with remaining uncertainty.',
].join(' ')

export function buildDeepWebResearchGraph(deps: Deps) {
  return buildResearcherGraph({
    ...deps,
    systemPrompt: [deps.systemPrompt ?? '', deepWebResearchPrompt]
      .filter(Boolean)
      .join('\n\n'),
  }, { depth: 'deep' })
}
