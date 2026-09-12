import type { Deps } from '../nodes.js'
import { buildFocusedLoopGraph } from './focused-loop.js'

const browserAgentPrompt = [
  'You are a browser agent.',
  'Prefer browser navigation, extraction, and screenshot tools when the task involves websites or rendered UI.',
  'For an already-running local service, process.sessions and the read-only process observation tools may be used to rediscover its managed id or inspect readiness; do not start, stop, or signal processes from this browser-validation graph.',
  'Use web search only to find the target page; once on-page, continue with browser-centric tools before falling back to terminal or file tools.',
].join(' ')

export const BROWSER_AGENT_TOOL_ALLOWLIST = [
  'browser.*',
  'process.list',
  'process.sessions',
  'process.read',
  'process.follow',
  'process.wait',
  'web.search',
  'webfetch',
] as const

export function buildBrowserAgentGraph(deps: Deps) {
  return buildFocusedLoopGraph(deps, {
    systemPrompt: browserAgentPrompt,
    // Web/UI scope: browser tools, read-only managed-process observation for
    // local service readiness/log context, plus search/fetch to locate targets.
    toolAllowlist: BROWSER_AGENT_TOOL_ALLOWLIST,
  })
}
