import {
  dockerAvailable as realDockerAvailable,
  listManagedContainers as realListManagedContainers,
  removeContainers as realRemoveContainers,
  pruneContainerLog as realPruneContainerLog,
  type ManagedContainer,
} from '../utils/docker.js'
import { detectCliLocale } from '../utils/locale.js'

export interface ContainersDeps {
  dockerAvailable: () => Promise<boolean>
  listManagedContainers: () => Promise<ManagedContainer[]>
  removeContainers: (idsOrNames: string[]) => Promise<void>
  pruneContainerLog: (removedNames: Set<string>) => Promise<void>
}

const realDeps: ContainersDeps = {
  dockerAvailable: realDockerAvailable,
  listManagedContainers: realListManagedContainers,
  removeContainers: realRemoveContainers,
  pruneContainerLog: realPruneContainerLog,
}

const CONTAINERS_COPY = {
  en: {
    dockerUnavailable:
      'Docker is not installed or its daemon is not running. Install/start Docker and try again.',
    noContainers: 'No sepilot-managed containers.',
    tableHeader:
      'NAME                              STATE     STATUS                  IMAGE            PURPOSE',
    removeHint:
      '\nRemove with: sepilot containers prune  (stopped)  |  sepilot containers prune --all  |  sepilot containers rm <name>',
    noContainersToRemove: 'No sepilot-managed containers to remove.',
    noStoppedContainers: 'No stopped sepilot-managed containers.',
    removedContainers: (count: number, names: string) => `Removed ${count} container(s): ${names}`,
    removed: (name: string) => `Removed: ${name}`,
  },
  ko: {
    dockerUnavailable:
      'Docker가 설치되어 있지 않거나 daemon이 실행 중이 아닙니다. Docker를 설치/시작한 뒤 다시 시도하세요.',
    noContainers: 'sepilot이 관리하는 컨테이너가 없습니다.',
    tableHeader:
      '이름                              상태      상태 설명               이미지           용도',
    removeHint:
      '\n삭제: sepilot containers prune  (중지된 컨테이너)  |  sepilot containers prune --all  |  sepilot containers rm <name>',
    noContainersToRemove: '삭제할 sepilot 관리 컨테이너가 없습니다.',
    noStoppedContainers: '중지된 sepilot 관리 컨테이너가 없습니다.',
    removedContainers: (count: number, names: string) =>
      `컨테이너 ${count}개를 삭제했습니다: ${names}`,
    removed: (name: string) => `삭제했습니다: ${name}`,
  },
} as const

function containersCopy() {
  return CONTAINERS_COPY[detectCliLocale()] ?? CONTAINERS_COPY.en
}

async function requireDocker(deps: ContainersDeps): Promise<boolean> {
  if (await deps.dockerAvailable()) return true
  console.error(containersCopy().dockerUnavailable)
  process.exit(1)
  return false
}

export async function containersLsImpl(deps: ContainersDeps): Promise<void> {
  const copy = containersCopy()
  if (!(await requireDocker(deps))) return
  const list = await deps.listManagedContainers()
  if (list.length === 0) {
    console.log(copy.noContainers)
    return
  }
  console.log(copy.tableHeader)
  for (const c of list) {
    console.log(
      `${c.name.padEnd(33)} ${c.state.padEnd(9)} ${(c.status || '—').padEnd(23)} ${c.image.padEnd(16)} ${c.purpose ?? ''}`,
    )
  }
  console.log(copy.removeHint)
}

export async function containersPruneImpl(
  deps: ContainersDeps,
  opts: { all: boolean },
): Promise<void> {
  const copy = containersCopy()
  if (!(await requireDocker(deps))) return
  const list = await deps.listManagedContainers()
  const targets = opts.all ? list : list.filter((c) => c.state !== 'running')
  if (targets.length === 0) {
    console.log(opts.all ? copy.noContainersToRemove : copy.noStoppedContainers)
    return
  }
  await deps.removeContainers(targets.map((c) => c.id))
  await deps.pruneContainerLog(new Set(targets.map((c) => c.name)))
  console.log(copy.removedContainers(targets.length, targets.map((c) => c.name).join(', ')))
}

export async function containersRmImpl(deps: ContainersDeps, idOrName: string): Promise<void> {
  const copy = containersCopy()
  if (!(await requireDocker(deps))) return
  // Resolve the container's name up front: `idOrName` may be a short ID, but
  // the containers.jsonl metadata is keyed by name.
  const match = (await deps.listManagedContainers()).find(
    (c) => c.id === idOrName || c.name === idOrName,
  )
  await deps.removeContainers([idOrName])
  await deps.pruneContainerLog(new Set([match?.name ?? idOrName]))
  console.log(copy.removed(match?.name ?? idOrName))
}

export async function containersLsCommand(): Promise<void> {
  await containersLsImpl(realDeps)
}

export async function containersPruneCommand(options: { all?: boolean } = {}): Promise<void> {
  await containersPruneImpl(realDeps, { all: !!options.all })
}

export async function containersRmCommand(idOrName: string): Promise<void> {
  await containersRmImpl(realDeps, idOrName)
}
