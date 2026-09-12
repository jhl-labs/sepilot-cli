import { createMediaExtractTextTool, createMediaInspectTool } from '../../tools/media.js'
import type { FeatureToolRegistrar } from '../types.js'

export const registerTools: FeatureToolRegistrar = (registry) => {
  registry.register(createMediaInspectTool())
  registry.register(createMediaExtractTextTool())
}
