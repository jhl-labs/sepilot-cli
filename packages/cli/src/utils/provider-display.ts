import type { DaemonProviderInfo } from '@sepilotd/api-client'

type DaemonProviderModel = DaemonProviderInfo['models'][number]

export function formatProviderModelBadges(model: DaemonProviderModel): string {
  const badges: string[] = []

  if (model.capabilities.embedding) {
    badges.push('embed')
  }

  badges.push(model.capabilities.toolUse ? 'native-tools' : 'prompt-tools')

  if (model.capabilities.thinking) {
    badges.push('thinking')
  }

  return badges.map((badge) => ` [${badge}]`).join('')
}
