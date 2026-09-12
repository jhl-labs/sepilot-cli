import type { DesktopAgentId } from '@sepilotd/api-client'
import { DESKTOP_AGENT_PRESETS } from './desktop-agent.js'
import { resolveAgentExecutable, terminalEnvironment } from './pty.js'

export function probeDesktopAgent(
  id: DesktopAgentId,
  env = terminalEnvironment(process.env),
): Promise<{ available: boolean; reason?: string }> {
  try {
    resolveAgentExecutable(DESKTOP_AGENT_PRESETS[id].command, env)
    return Promise.resolve({ available: true })
  } catch {
    return Promise.resolve({ available: false, reason: 'CLI 미설치 또는 실행 불가' })
  }
}
