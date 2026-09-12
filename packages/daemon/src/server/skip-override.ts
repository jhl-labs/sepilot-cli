import type { FastifyPluginCallback } from 'fastify'

const SKIP_OVERRIDE = Symbol.for('skip-override')

export function skipOverride(plugin: FastifyPluginCallback): void {
  ;(plugin as { [SKIP_OVERRIDE]?: boolean })[SKIP_OVERRIDE] = true
}