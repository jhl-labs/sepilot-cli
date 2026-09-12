export interface SchedulerDeliveryRoute {
  channelType: string | null
  channelTarget: string | null
  replyToMessageId: string | null
}

export function deliveryRouteError(input: {
  channelType?: string | null
  channelTarget?: string | null
  replyToMessageId?: string | null
}): string | null {
  const hasType = Boolean(input.channelType)
  const hasTarget = Boolean(input.channelTarget)
  if (hasType !== hasTarget) return 'channelType and channelTarget must be configured together'
  if (input.replyToMessageId && !hasType) {
    return 'replyToMessageId requires channelType and channelTarget'
  }
  return null
}
