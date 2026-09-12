import type { ToolExecutionPosture } from '@sepilotd/core'

export interface FileToolPostureOptions {
  /** Daemon `security.sandbox` mode the tool registry was built with. */
  sandboxMode?: 'local' | 'docker' | 'bubblewrap'
}

/**
 * Honest execution posture for host-side file tools when the daemon is
 * configured with a `terminal.run`-only sandbox (`docker` or
 * `bubblewrap`). Those sandboxes isolate command execution only —
 * fs.write/fs.edit/fs.append/apply_patch still write directly to the
 * host filesystem. Without this note a run looks fully sandboxed while
 * file writes are not, which is a false sense of safety. Returns
 * undefined for local mode (nothing was claimed, so nothing to correct).
 */
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
