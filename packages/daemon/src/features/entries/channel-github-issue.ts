import { join } from 'node:path'
import { GitHubIssueChannel } from '../../channels/github-issue.js'
import type { ChannelFactoryRegistry } from '../../server/runtime/channels.js'

export function registerChannelFactory(registry: ChannelFactoryRegistry): void {
  registry.register('github-issue', (channel, deps) => {
    const channelConfig = channel.config as Record<string, unknown> | undefined
    return {
      channel: new GitHubIssueChannel(
        {
          owner: typeof channelConfig?.owner === 'string' ? channelConfig.owner : '',
          repo: typeof channelConfig?.repo === 'string' ? channelConfig.repo : '',
          labels: Array.isArray(channelConfig?.labels)
            ? channelConfig.labels.filter((label): label is string => typeof label === 'string')
            : ['ai-task'],
          gatewayUrl: deps.gatewayUrl,
          processingStorePath: deps.dataDir
            ? join(deps.dataDir, 'channels', 'github-issue-processing.db')
            : undefined,
          claimEnabled: process.env.SEPILOTD_GITHUB_TICKET_CLAIM === '1',
        },
        deps.gatewayClient,
      ),
    }
  })
}
