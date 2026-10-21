import type { ToolExecutionPosture } from '@sepilotd/core'

export interface FileToolPostureOptions {
  sandboxMode?: 'local' | 'docker' | 'bubblewrap'
}

export function buildFileToolPosture(
  options: FileToolPostureOptions | undefined,
  cwd: string | undefined,
): ToolExecutionPosture | undefined {
  const mode = options?.sandboxMode
  if (mode !== 'docker' && mode !== 'bubblewrap') return undefined
  return {
    sandbox: {
      requested: true,
      active: false,
      mode: 'host',
      fallbackReason: `security.sandbox=${mode} isolates terminal.run only; this file tool writes directly to the host filesystem.`,
    },
    filesystem: {
      boundary: 'workspace_policy',
      cwd,
      isolated: false,
      note: `Host filesystem write gated by approval policy and deny_paths, not by the ${mode} sandbox.`,
    },
    network: {
      isolated: false,
      mode: 'host',
    },
  }
}
