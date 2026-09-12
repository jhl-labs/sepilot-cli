import type { FastifyInstance } from 'fastify'
import { registerCapability, type CapabilityMethod } from './registry.js'

export interface CapabilityMeta {
  name: string
  version: string
  description?: string
  methods: CapabilityMethod[]
}

export type CapabilityRegister = (
  app: FastifyInstance,
) => Promise<void> | void

export async function bindCapability(
  app: FastifyInstance,
  meta: CapabilityMeta,
  register: CapabilityRegister,
): Promise<void> {
  await register(app)
  registerCapability({ ...meta, available: true })
}
