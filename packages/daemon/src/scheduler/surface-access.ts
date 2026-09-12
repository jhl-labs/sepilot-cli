import type { FastifyReply } from 'fastify'
import { schedulerSurfaceAccessSchema, type SepilotdConfig } from '../config/schema.js'

export type SchedulerSurface = 'cli' | 'desktop' | 'mobile'

export const SCHEDULER_SURFACE_DISABLED_CODE = 'SCHEDULER_SURFACE_DISABLED'

/**
 * `scheduler` is optional in the config schema, so a config file that never
 * mentions it — every fresh install — leaves the key undefined. Fall back to
 * the schema's declared per-surface defaults instead of reading "absent" as
 * "every surface disabled", which silently contradicts them and rejects the
 * surfaces the schema enables by default.
 */
export function isSchedulerSurfaceEnabled(
  config: Pick<SepilotdConfig, 'scheduler'> | undefined,
  surface: SchedulerSurface,
): boolean {
  if (!config) return true
  const surfaces = config.scheduler?.surfaces ?? schedulerSurfaceAccessSchema.parse(undefined)
  return surfaces[surface] === true
}

export function schedulerSurfaceDisabledMessage(surface: SchedulerSurface): string {
  return `Scheduler ${surface} surface is disabled. Enable scheduler.surfaces.${surface} in config.yaml or with "sepilot config-set scheduler.surfaces.${surface} true".`
}

export function sendSchedulerSurfaceDisabled(
  reply: FastifyReply,
  surface: SchedulerSurface,
): FastifyReply {
  return reply.status(403).send({
    error: {
      code: SCHEDULER_SURFACE_DISABLED_CODE,
      message: schedulerSurfaceDisabledMessage(surface),
    },
  })
}

export function sendSchedulerCapabilityDisabled(
  reply: FastifyReply,
  surface: SchedulerSurface,
): FastifyReply {
  return reply.status(403).send({
    code: SCHEDULER_SURFACE_DISABLED_CODE,
    message: schedulerSurfaceDisabledMessage(surface),
    retriable: false,
  })
}
