import { setTimeout as timerSleep } from 'node:timers/promises'
import type { SwarmAgentHandle, SwarmAgentName } from '@sepilotd/core'
import { getAgentConfig } from '../config/agents.js'
import { formatStartupCommand } from '../run/startup-evidence.js'
import type { TmuxSessionPool } from '../tmux/pool.js'

export interface LaunchOpts {
  runId: string
  handle: string
  agent: SwarmAgentName
  cwd: string
  role?: string
  autoApprove: boolean
}

export class AgentLauncher {
  constructor(
    private pool: TmuxSessionPool,
    private sleep: (ms: number) => Promise<void> = timerSleep,
  ) {}

  async launch(opts: LaunchOpts): Promise<SwarmAgentHandle> {
    const startedAt = Date.now()
    const cfg = getAgentConfig(opts.agent)
    const tmuxName = `sepilotd_swarm_${opts.runId.replace(/^swarm_/, '')}_${opts.handle}`
    const args = [...cfg.args]
    if (opts.autoApprove) {
      args.push(...(cfg.autoApproveArgs ?? (cfg.autoApproveFlag ? [cfg.autoApproveFlag] : [])))
    }
    const command = [cfg.command, ...args]
    // Run the agent as the pane's process (NOT via shell+send-keys) so its
    // TUI takes the PTY directly. Otherwise claude/codex see no terminal
    // and fail to render their UI.
    await this.pool.create({
      name: tmuxName,
      cwd: opts.cwd,
      env: cfg.env,
      command,
    })
    await this.sleep(cfg.startupWaitMs)
    const readyAt = Date.now()
    return {
      handle: opts.handle,
      agent: opts.agent,
      role: opts.role,
      tmuxSessionName: tmuxName,
      cwd: opts.cwd,
      status: 'idle',
      spawnedAt: readyAt,
      runtime: 'tmux',
      startupEvidence: {
        handle: opts.handle,
        agent: opts.agent,
        runtime: 'tmux',
        lifecycleState: 'ready_for_prompt',
        cwd: opts.cwd,
        paneCommand: formatStartupCommand(command),
        startedAt,
        readyAt,
        promptAccepted: false,
        trustPromptDetected: false,
        toolPermissionPromptDetected: false,
        transportHealthy: true,
        elapsedMs: Math.max(0, readyAt - startedAt),
      },
    }
  }

  async stop(handle: SwarmAgentHandle): Promise<void> {
    const cfg = getAgentConfig(handle.agent)
    try {
      await this.pool.sendKeys(handle.tmuxSessionName, cfg.exitCommand)
    } catch {
      /* ignore */
    }
    await this.sleep(500)
    await this.pool.destroy(handle.tmuxSessionName)
  }
}
