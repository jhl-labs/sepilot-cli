import { join } from 'node:path'
import { PluginLoader } from '../../plugins/loader.js'
import type { SepilotdConfig } from '../../config/schema.js'

// Re-export the plugin OpenAPI schema so the manifest can surface it to
// server/app.ts only when the plugins feature is enabled.
export {
  pluginOpenApiComponents,
  pluginOpenApiOverrides,
} from '../../server/routes/plugins-openapi.js'

/**
 * Builds a PluginLoader. Only imported (and thus only bundled) when the
 * `plugins` feature is enabled; when disabled the manifest returns null and
 * consumers skip plugin loading gracefully.
 */
export function createPluginLoader(dataDir: string, config?: SepilotdConfig): PluginLoader {
  return new PluginLoader(join(dataDir, 'plugins'), {
    strict: config?.plugins.strict,
    loadTimeoutMs: config?.plugins.loadTimeoutMs,
    trustedSignatureKeys: config?.plugins.trustedSignatureKeys,
  })
}
