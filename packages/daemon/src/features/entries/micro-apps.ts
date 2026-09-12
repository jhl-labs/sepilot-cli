import { createAppsTools } from '../../tools/apps.js'
import type { FeatureRuntimeToolRegistrar, FeatureToolRegistrar } from '../types.js'

export const registerTools: FeatureToolRegistrar = (registry, deps) => {
  for (const tool of createAppsTools({ dataDir: deps.dataDir })) {
    registry.register(tool)
  }
}

/**
 * Second-pass registration inside buildRuntime: re-registers the apps tools
 * (overriding the buildToolRegistry ones) now that the semantic index exists so
 * app content is searchable.
 */
export const registerRuntimeTools: FeatureRuntimeToolRegistrar = (registry, deps) => {
  for (const tool of createAppsTools({
    dataDir: deps.dataDir,
    semanticIndex: deps.semanticIndex,
  })) {
    registry.register(tool)
  }
}
