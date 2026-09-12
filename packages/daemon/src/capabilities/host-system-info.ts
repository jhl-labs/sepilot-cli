import type { SepilotdConfig } from '../config/schema.js'
import type { ToolRegistry } from '../tools/registry.js'
import { createSystemInfoTool } from '../tools/system-info.js'

export const HOST_SYSTEM_INFO_CAPABILITY_ID = 'host-system-info'

export interface HostSystemInfoCapabilityConfig {
  agent?: {
    capabilities?: {
      hostSystemInfo?: boolean
    }
  }
}

export function isHostSystemInfoCapabilityEnabled(
  config?: HostSystemInfoCapabilityConfig,
): boolean {
  return config?.agent?.capabilities?.hostSystemInfo ?? true
}

export function registerHostSystemInfoCapability(options: {
  config?: SepilotdConfig | HostSystemInfoCapabilityConfig
  tools: ToolRegistry
  registerTool?: boolean
}): boolean {
  if (!isHostSystemInfoCapabilityEnabled(options.config)) {
    return false
  }

  if (options.registerTool !== false && !options.tools.get('system.info')) {
    options.tools.register(createSystemInfoTool())
  }

  return true
}
