import { AutonomyLevel } from '@sepilotd/core'
import { ALLOWED_DOWNGRADES } from './autonomy.js'

export type TrustLevel = 'owner' | 'trusted' | 'untrusted'

export interface ChannelAclConfig {
  channelType: string
  mode: 'pairing' | 'allowlist' | 'open'
  allowedUsers: string[]
  maxAutonomy: AutonomyLevel
  rateLimitPerMinute?: number
}

export class ChannelAcl {
  private configs = new Map<string, ChannelAclConfig>()

  configure(config: ChannelAclConfig): void {
    this.configs.set(config.channelType, config)
  }

  getTrustLevel(channelType: string, userId: string): TrustLevel {
    const config = this.configs.get(channelType)
    if (!config) return 'untrusted'

    if (config.mode === 'open') return 'trusted'
    if (config.allowedUsers.includes(userId)) return 'owner'
    return 'untrusted'
  }

  getEffectiveAutonomy(channelType: string, globalAutonomy: AutonomyLevel): AutonomyLevel {
    const config = this.configs.get(channelType)
    if (!config) return globalAutonomy

    const ceiling = config.maxAutonomy
    if ((ALLOWED_DOWNGRADES[ceiling] ?? []).includes(globalAutonomy)) {
      return globalAutonomy
    }
    if ((ALLOWED_DOWNGRADES[globalAutonomy] ?? []).includes(ceiling)) {
      return ceiling
    }
    return AutonomyLevel.ReadOnly
  }

  isAllowed(channelType: string, userId: string): boolean {
    const trust = this.getTrustLevel(channelType, userId)
    return trust !== 'untrusted'
  }

  addAllowedUser(channelType: string, userId: string): void {
    const config = this.configs.get(channelType)
    if (config && !config.allowedUsers.includes(userId)) {
      config.allowedUsers.push(userId)
    }
  }

  removeAllowedUser(channelType: string, userId: string): void {
    const config = this.configs.get(channelType)
    if (config) {
      config.allowedUsers = config.allowedUsers.filter(u => u !== userId)
    }
  }

  getConfig(channelType: string): ChannelAclConfig | undefined {
    return this.configs.get(channelType)
  }
}
