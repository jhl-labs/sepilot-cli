import { createAppsTools } from '../../tools/apps.js'
import type { FeatureRuntimeToolRegistrar, FeatureToolRegistrar } from '../types.js'

export const registerTools: FeatureToolRegistrar = (registry, deps) => {
  for (const tool of createAppsTools({ dataDir: deps.dataDir })) {
    registry.register(tool)
  }
}

export const registerRuntimeTools: FeatureRuntimeToolRegistrar = (registry, deps) => {
  for (const tool of createAppsTools({
    dataDir: deps.dataDir,
    semanticIndex: deps.semanticIndex,
  })) {
    registry.register(tool)
  }
}
