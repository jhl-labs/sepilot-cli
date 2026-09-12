import type { SwarmAgentName } from '@sepilotd/core'

export interface AgentLaunchConfig {
  command: string
  args: string[]
  autoApproveFlag: string | null
  autoApproveArgs?: string[]
  startupWaitMs: number
  idlePatterns: RegExp[]
  busyPatterns: RegExp[]
  minWaitMs: number
  stableSeconds: number
  exitCommand: string
  env: Record<string, string>
  /**
   * Number of Enter key presses needed to submit a prompt. claude and
   * gemini submit on the first Enter; codex and opencode use multi-line input boxes
   * where the first Enter inserts a newline and the second Enter (on an
   * empty line) actually sends. Verified by direct tmux testing on the
   * binaries shipped at the time of writing.
   */
  submitEnters: number
}

export const AGENT_CONFIGS: Record<SwarmAgentName, AgentLaunchConfig> = {
  claude: {
    command: 'claude',
    args: [],
    autoApproveFlag: '--dangerously-skip-permissions',
    startupWaitMs: 5_000,
    idlePatterns: [/[\$>]\s*$/m, /bypass permissions on/i],
    busyPatterns: [/[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]/, /Thinking/, /Working/],
    minWaitMs: 2_000,
    stableSeconds: 5,
    exitCommand: '/exit',
    env: { NO_COLOR: '1' },
    submitEnters: 1,
  },
  codex: {
    command: 'codex',
    args: [],
    autoApproveFlag: null,
    startupWaitMs: 3_000,
    idlePatterns: [/[\$>]\s*$/m, /^›\s*$/m],
    busyPatterns: [],
    minWaitMs: 1_000,
    stableSeconds: 3,
    exitCommand: 'exit',
    env: {},
    submitEnters: 2,
  },
  gemini: {
    command: 'gemini',
    args: [],
    autoApproveFlag: null,
    // Gemini CLI deprecated --yolo in favor of --approval-mode=yolo.
    // --skip-trust avoids an interactive folder-trust gate for daemon-created
    // worktrees while preserving the existing autoApproveAgents switch.
    autoApproveArgs: ['--skip-trust', '--approval-mode=yolo'],
    startupWaitMs: 6_000,
    idlePatterns: [/[\$>]\s*$/m, /╭─/, /Type your message/i],
    busyPatterns: [
      /[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]/,
      /Thinking/,
      /Working/,
      /Attempting to automatically update/i,
    ],
    minWaitMs: 1_000,
    stableSeconds: 4,
    exitCommand: '/quit',
    env: { NO_COLOR: '1' },
    submitEnters: 1,
  },
  opencode: {
    command: 'opencode',
    args: [],
    autoApproveFlag: null,
    startupWaitMs: 3_000,
    idlePatterns: [/>\s*$/m],
    busyPatterns: [/[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]/],
    minWaitMs: 1_000,
    stableSeconds: 4,
    exitCommand: '/exit',
    env: {},
    submitEnters: 2,
  },
}

export function getAgentConfig(name: SwarmAgentName): AgentLaunchConfig {
  const cfg = AGENT_CONFIGS[name]
  if (!cfg) throw new Error(`unknown swarm agent: ${String(name)}`)
  return cfg
}
