import type { FastifyReply } from 'fastify'
import { schedulerSurfaceAccessSchema, type SepilotdConfig } from '../config/schema.js'

export type SchedulerSurface = 'cli' | 'desktop' | 'mobile'

export const SCHEDULER_SURFACE_DISABLED_CODE = 'SCHEDULER_SURFACE_DISABLED'

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
