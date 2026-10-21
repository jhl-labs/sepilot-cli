import { join } from 'node:path'
import { PluginLoader } from '../../plugins/loader.js'
import type { SepilotdConfig } from '../../config/schema.js'

export {
  pluginOpenApiComponents,
  pluginOpenApiOverrides,
} from '../../server/routes/plugins-openapi.js'

export function createPluginLoader(dataDir: string, config?: SepilotdConfig): PluginLoader {
  return new PluginLoader(join(dataDir, 'plugins'), {
    strict: config?.plugins.strict,
    loadTimeoutMs: config?.plugins.loadTimeoutMs,
    trustedSignatureKeys: config?.plugins.trustedSignatureKeys,
  })
}
