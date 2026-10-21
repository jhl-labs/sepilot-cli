export const EVENT_ROUTING_SURFACE = 'event-routing'

export const EVENT_ROUTING_SESSION_TAG = 'event-routing'

export const EVENT_ROUTING_ORIGIN_HEADER = 'x-sepilot-origin'
export const EVENT_ROUTING_ORIGIN = 'event-routing'

export function hasEventRoutingOrigin(value: string | string[] | undefined): boolean {
  if (value === undefined) return false
  const values = Array.isArray(value) ? value : [value]
  return values.some((entry) => entry
    .split(',')
    .some((token) => token.trim().toLowerCase() === EVENT_ROUTING_ORIGIN))
}
