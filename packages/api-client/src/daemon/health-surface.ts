import type {
  DaemonHealthComponent,
  DaemonHealthReadinessCheck,
  DaemonHealthReportSnapshot,
} from './types.js'

export interface DaemonHealthSurfaceItem {
  key: string
  status: string
  details?: string
  optional?: boolean
  core?: boolean
}

function statusWeight(status: string): number {
  switch (status) {
    case 'error':
    case 'unreachable':
    case 'not_ready':
      return 0
    case 'degraded':
    case 'reindex_required':
      return 1
    case 'ok':
      return 3
    default:
      return 2
  }
}

function compareByStatus(
  left: DaemonHealthComponent | DaemonHealthReadinessCheck,
  right: DaemonHealthComponent | DaemonHealthReadinessCheck,
): number {
  const leftWeight = statusWeight(left.status)
  const rightWeight = statusWeight(right.status)
  if (leftWeight !== rightWeight) {
    return leftWeight - rightWeight
  }
  return 0
}

function isOptionalHealthComponent(
  key: string,
  component: DaemonHealthComponent,
): boolean {
  return component.optional === true || key === 'gateway'
}

function isCoreHealthComponent(
  key: string,
  component: DaemonHealthComponent,
): boolean {
  return !isOptionalHealthComponent(key, component) && component.core !== false
}

function healthComponentCategory(
  key: string,
  component: DaemonHealthComponent,
): number {
  if (isOptionalHealthComponent(key, component)) return 2
  if (component.core === false) return 1
  return 0
}

function isProblemComponent(component: DaemonHealthComponent): boolean {
  return component.status !== 'ok' && component.status !== 'disabled'
}

export function getDaemonHealthMonitorLabel(
  snapshot: DaemonHealthReportSnapshot | null,
): string {
  if (!snapshot) {
    return 'Unavailable'
  }
  if (snapshot.readiness.status !== 'ok') {
    return 'Readiness Attention'
  }
  const componentEntries = Object.entries(snapshot.health.components)
  const coreComponents = componentEntries
    .filter(([key, component]) => isCoreHealthComponent(key, component))
  const coreComponentNeedsAttention = coreComponents.some(([, component]) =>
    isProblemComponent(component),
  )
  if (
    snapshot.health.status !== 'ok'
    && (componentEntries.length === 0 || coreComponentNeedsAttention)
  ) {
    return 'Degraded'
  }
  return 'Healthy'
}

export function getDaemonHealthMonitorMeta(
  snapshot: DaemonHealthReportSnapshot | null,
): string | null {
  if (!snapshot) {
    return null
  }
  return `Updated ${new Date(snapshot.generatedAt).toLocaleTimeString()}`
}

export function getDaemonHealthMonitorDetail(
  snapshot: DaemonHealthReportSnapshot | null,
): string {
  if (!snapshot) {
    return 'Health snapshot unavailable.'
  }

  const componentEntries = Object.entries(snapshot.health.components)
  const degradedComponents = componentEntries.filter(
    ([key, component]) =>
      isCoreHealthComponent(key, component) && isProblemComponent(component),
  ).length
  const degradedFeatureComponents = componentEntries.filter(
    ([key, component]) =>
      !isOptionalHealthComponent(key, component)
      && component.core === false
      && isProblemComponent(component),
  ).length
  const degradedOptionalComponents = componentEntries.filter(
    ([key, component]) =>
      isOptionalHealthComponent(key, component) && isProblemComponent(component),
  ).length
  const failingReadinessChecks = Object.values(snapshot.readiness.checks).filter(
    (check) => check.status !== 'ok',
  ).length

  if (
    degradedComponents === 0
    && failingReadinessChecks === 0
    && degradedFeatureComponents === 0
    && degradedOptionalComponents === 0
  ) {
    return 'All health components and readiness checks are passing.'
  }

  const parts: string[] = []
  if (degradedComponents > 0) {
    parts.push(`${degradedComponents} degraded component${degradedComponents === 1 ? '' : 's'}`)
  }
  if (failingReadinessChecks > 0) {
    parts.push(
      `${failingReadinessChecks} readiness check${
        failingReadinessChecks === 1 ? ' needs' : 's need'
      } attention`,
    )
  }
  if (degradedFeatureComponents > 0) {
    parts.push(
      `${degradedFeatureComponents} feature component${
        degradedFeatureComponents === 1 ? '' : 's'
      } ${degradedFeatureComponents === 1 ? 'needs' : 'need'} attention`,
    )
  }
  if (degradedOptionalComponents > 0) {
    parts.push(
      `${degradedOptionalComponents} optional component${
        degradedOptionalComponents === 1 ? '' : 's'
      } ${degradedOptionalComponents === 1 ? 'needs' : 'need'} attention`,
    )
  }
  return parts.join(', ')
}

export function listDaemonHealthSurfaceItems(
  snapshot: DaemonHealthReportSnapshot | null,
  limit = 6,
): DaemonHealthSurfaceItem[] {
  if (!snapshot) {
    return []
  }

  return Object.entries(snapshot.health.components)
    .sort(([leftKey, left], [rightKey, right]) => {
      const leftCategory = healthComponentCategory(leftKey, left)
      const rightCategory = healthComponentCategory(rightKey, right)
      if (leftCategory !== rightCategory) {
        return leftCategory - rightCategory
      }
      const byStatus = compareByStatus(left, right)
      if (byStatus !== 0) {
        return byStatus
      }
      return leftKey.localeCompare(rightKey)
    })
    .slice(0, limit)
    .map(([key, component]) => {
      const item: DaemonHealthSurfaceItem = {
        key,
        status: component.status,
      }
      if (component.details !== undefined) {
        item.details = component.details
      }
      if (isOptionalHealthComponent(key, component)) {
        item.optional = true
      }
      if (component.core === false) {
        item.core = false
      }
      return item
    })
}
