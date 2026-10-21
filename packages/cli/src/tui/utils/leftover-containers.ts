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
