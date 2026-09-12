import { constants, statSync, accessSync } from 'node:fs'
import { delimiter, isAbsolute, join } from 'node:path'
import type {
  SwarmAgentName,
  SwarmAgentRuntimeName,
  SwarmPreflightCheck,
  SwarmPreflightReport,
} from '@sepilotd/core'
import {
  isExternalAcpAgentName,
  resolveExternalAcpAgentPreset,
} from '../../../acp/external-agent.js'
import { getAgentConfig } from '../config/agents.js'
import { redactStartupEvidenceText } from '../run/startup-evidence.js'

function envKeyForAgent(agent: SwarmAgentName, suffix: string): string {
  return `SEPILOTD_SWARM_${agent.toUpperCase()}_${suffix}`
}

function addCheck(checks: SwarmPreflightCheck[], check: SwarmPreflightCheck): void {
  checks.push(check)
}

function runtimeFromEnv(agent: SwarmAgentName, env: NodeJS.ProcessEnv): {
  runtime: SwarmAgentRuntimeName
  check?: SwarmPreflightCheck
} {
  const raw = env[envKeyForAgent(agent, 'RUNTIME')] ?? env.SEPILOTD_SWARM_AGENT_RUNTIME
  const normalized = raw?.trim().toLowerCase()
  if (!normalized) return { runtime: 'tmux' }
  if (normalized === 'tmux' || normalized === 'acp' || normalized === 'a2a') {
    return { runtime: normalized }
  }
  return {
    runtime: 'unknown',
    check: {
      id: 'runtime',
      status: 'blocked',
      summary: `Unsupported swarm runtime: ${raw}`,
      detail: 'Use tmux, acp, or a2a.',
    },
  }
}

function executableExists(command: string, env: NodeJS.ProcessEnv): boolean {
  const candidates = command.includes('/') || isAbsolute(command)
    ? [command]
    : (env.PATH ?? process.env.PATH ?? '')
      .split(delimiter)
      .filter(Boolean)
      .map((dir) => join(dir, command))
  for (const candidate of candidates) {
    try {
      accessSync(candidate, constants.X_OK)
      return true
    } catch {
      /* try next path */
    }
  }
  return false
}

function a2aAgentCardUrlForAgent(
  env: NodeJS.ProcessEnv,
  agent: SwarmAgentName,
): string | undefined {
  return env[envKeyForAgent(agent, 'A2A_AGENT_CARD_URL')]
    ?? env[envKeyForAgent(agent, 'A2A_URL')]
    ?? env.SEPILOTD_SWARM_A2A_AGENT_CARD_URL
    ?? env.SEPILOTD_SWARM_A2A_URL
}

function validateA2AUrl(raw: string | undefined, checks: SwarmPreflightCheck[]): void {
  const value = raw?.trim()
  if (!value) {
    addCheck(checks, {
      id: 'a2a_url',
      status: 'blocked',
      summary: 'A2A runtime selected but no Agent Card URL is configured.',
      detail: 'Set SEPILOTD_SWARM_A2A_AGENT_CARD_URL or the agent-specific A2A URL env var.',
    })
    return
  }
  try {
    const parsed = new URL(value)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      throw new Error('URL must use http or https')
    }
    if (parsed.username || parsed.password) {
      throw new Error('URL must not contain credentials')
    }
    addCheck(checks, {
      id: 'a2a_url',
      status: 'ok',
      summary: 'A2A Agent Card URL is configured.',
      detail: redactStartupEvidenceText(parsed.toString()),
    })
  } catch (error) {
    addCheck(checks, {
      id: 'a2a_url',
      status: 'blocked',
      summary: 'A2A Agent Card URL is invalid.',
      detail: error instanceof Error ? error.message : String(error),
    })
  }
}

function broadCwdWarning(cwd: string, env: NodeJS.ProcessEnv): SwarmPreflightCheck | null {
  const home = env.HOME || process.env.HOME
  const broad = new Set(['/', '/tmp', '/var/tmp'])
  if (home) broad.add(home)
  if (!broad.has(cwd)) return null
  return {
    id: 'cwd_scope',
    status: 'warn',
    summary: `Swarm cwd is broad: ${cwd}`,
    detail: 'Prefer a repository worktree or project subdirectory before launching external agents.',
  }
}

export function runSwarmAgentPreflight(input: {
  agent: SwarmAgentName
  cwd: string
  env?: NodeJS.ProcessEnv
}): SwarmPreflightReport {
  const env = input.env ?? process.env
  const checks: SwarmPreflightCheck[] = []
  const runtimeResult = runtimeFromEnv(input.agent, env)
  if (runtimeResult.check) checks.push(runtimeResult.check)

  try {
    const stat = statSync(input.cwd)
    addCheck(checks, {
      id: 'cwd',
      status: stat.isDirectory() ? 'ok' : 'blocked',
      summary: stat.isDirectory()
        ? 'Working directory exists.'
        : 'Working directory is not a directory.',
      detail: input.cwd,
    })
  } catch (error) {
    addCheck(checks, {
      id: 'cwd',
      status: 'blocked',
      summary: 'Working directory does not exist.',
      detail: error instanceof Error ? error.message : String(error),
    })
  }
  const cwdScope = broadCwdWarning(input.cwd, env)
  if (cwdScope) checks.push(cwdScope)

  if (runtimeResult.runtime === 'tmux') {
    const cfg = getAgentConfig(input.agent)
    const tmuxAvailable = executableExists('tmux', env)
    const agentBinaryAvailable = executableExists(cfg.command, env)
    addCheck(checks, {
      id: 'tmux',
      status: tmuxAvailable ? 'ok' : 'blocked',
      summary: tmuxAvailable
        ? 'tmux binary is available.'
        : 'tmux binary is missing from PATH.',
    })
    addCheck(checks, {
      id: 'agent_binary',
      status: agentBinaryAvailable ? 'ok' : 'blocked',
      summary: agentBinaryAvailable
        ? `${cfg.command} binary is available.`
        : `${cfg.command} binary is missing from PATH.`,
    })
  } else if (runtimeResult.runtime === 'acp') {
    if (!isExternalAcpAgentName(input.agent)) {
      addCheck(checks, {
        id: 'acp_agent',
        status: 'blocked',
        summary: `ACP runtime does not support ${input.agent}.`,
      })
    } else {
      const resolved = resolveExternalAcpAgentPreset(input.agent, env)
      const acpBinaryAvailable = executableExists(resolved.command, env)
      addCheck(checks, {
        id: 'acp_binary',
        status: acpBinaryAvailable ? 'ok' : 'blocked',
        summary: acpBinaryAvailable
          ? `${resolved.command} ACP command is available.`
          : `${resolved.command} ACP command is missing from PATH.`,
      })
    }
  } else if (runtimeResult.runtime === 'a2a') {
    validateA2AUrl(a2aAgentCardUrlForAgent(env, input.agent), checks)
  }

  const blocked = checks.some((check) => check.status === 'blocked')
  const warned = checks.some((check) => check.status === 'warn')
  return {
    status: blocked ? 'blocked' : warned ? 'degraded' : 'ready',
    agent: input.agent,
    runtime: runtimeResult.runtime,
    cwd: input.cwd,
    checks,
  }
}

export function summarizeSwarmPreflight(report: SwarmPreflightReport): string {
  const failed = report.checks.filter((check) => check.status === 'blocked')
  if (failed.length > 0) {
    return failed.map((check) => `${check.id}: ${check.summary}`).join('; ')
  }
  const warnings = report.checks.filter((check) => check.status === 'warn')
  if (warnings.length > 0) {
    return warnings.map((check) => `${check.id}: ${check.summary}`).join('; ')
  }
  return 'swarm preflight ready'
}
