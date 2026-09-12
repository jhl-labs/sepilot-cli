import {
  dockerAvailable as realDockerAvailable,
  listManagedContainers as realListManagedContainers,
  type ManagedContainer,
} from '../../utils/docker.js'

export interface LeftoverDeps {
  dockerAvailable: () => Promise<boolean>
  listManagedContainers: () => Promise<ManagedContainer[]>
}

const realDeps: LeftoverDeps = {
  dockerAvailable: realDockerAvailable,
  listManagedContainers: realListManagedContainers,
}

const MAX_NAMED = 3

/**
 * Returns 0–3 banner lines describing sepilot-managed Docker containers left
 * over from previous sessions. Empty when Docker is unavailable, when there
 * are none, or on any error — querying Docker must never block the TUI.
 */
export async function buildLeftoverContainerBanner(deps: LeftoverDeps = realDeps): Promise<string[]> {
  try {
    if (!(await deps.dockerAvailable())) return []
    const list = await deps.listManagedContainers()
    if (list.length === 0) return []
    const named = list.slice(0, MAX_NAMED).map((c) => `${c.name} (${c.state})`).join(', ')
    const more = list.length > MAX_NAMED ? `, +${list.length - MAX_NAMED} more` : ''
    return [
      `${list.length} container(s) from previous sessions still around: ${named}${more}`,
      '  /containers to view · run `sepilot containers prune` to clean up',
    ]
  } catch {
    return []
  }
}
