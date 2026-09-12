import type { Deps } from '../nodes.js'
import { buildFocusedLoopGraph } from './focused-loop.js'

const terminalAgentPrompt = [
  'You are a terminal agent.',
  'Prefer terminal.run for command-driven work, keep commands intentional, and avoid unbounded command loops.',
  'Use system.info with sampleCount and sampleIntervalMs for bounded host CPU or memory observation windows; it works without a TTY and is read-only.',
  'Use process.start followed by process.follow or process.wait for a genuinely concurrent child process. For a requested observation window on a continuously updating PTY, use one bounded process.follow; it waits for the full timeout by default, unlike ordinary log follow. For a full-screen/TUI observer, request a headless PTY and read its clean screen snapshot; raw ANSI is opt-in and no interactive stdin is available. Stop temporary sessions before finalizing.',
  'Summarize important command outcomes and switch back to reasoning as soon as enough evidence is collected.',
].join(' ')

export const TERMINAL_AGENT_TOOL_ALLOWLIST = [
  'terminal.run',
  'system.info',
  'process.list',
  'process.sessions',
  'process.start',
  'process.read',
  'process.follow',
  'process.wait',
  'process.stop',
  'fs.read',
  'fs.glob',
  'fs.search',
  'code.*',
] as const

export function buildTerminalAgentGraph(deps: Deps) {
  return buildFocusedLoopGraph(deps, {
    systemPrompt: terminalAgentPrompt,
    // Command-driven scope: foreground shell, audited host observation, managed
    // background processes, plus read-only file/code inspection.
    // No fs.write/apply_patch/browser — the registry description's "tighter
    // shell discipline" is now an actual tool boundary, not just prose.
    toolAllowlist: TERMINAL_AGENT_TOOL_ALLOWLIST,
  })
}
