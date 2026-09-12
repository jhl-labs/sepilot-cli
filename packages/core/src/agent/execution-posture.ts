import type { ToolExecutionPosture } from './types.js'

export function toolExecutionPostureFilesystemBoundaryLabel(
  boundary: ToolExecutionPosture['filesystem']['boundary'],
): string {
  switch (boundary) {
    case 'working-directory':
      return 'working directory'
    case 'session_cwd':
      return 'session cwd'
    case 'process_cwd':
      return 'process cwd'
    case 'workspace_policy':
      return 'workspace policy'
    case 'tool_specific':
      return 'tool cwd'
    case 'unknown':
      return 'unknown filesystem boundary'
    default:
      return String(boundary).replace(/_/g, ' ')
  }
}

export function toolExecutionPostureLabel(posture?: ToolExecutionPosture): string | undefined {
  if (!posture) return undefined

  const sandbox = posture.sandbox.active
    ? `${posture.sandbox.mode} sandbox`
    : 'host execution, no sandbox'
  const filesystem = posture.filesystem.isolated
    ? 'fs isolated'
    : `fs: ${toolExecutionPostureFilesystemBoundaryLabel(posture.filesystem.boundary)}`
  const network = posture.network.isolated ? `network: ${posture.network.mode}` : 'network: host'

  return `${sandbox} (${filesystem}, ${network})`
}
