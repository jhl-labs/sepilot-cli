import {
  createImageGenCancelTool,
  createImageGenCreateTool,
  createImageGenFileTool,
  createImageGenJobTool,
  createImageGenProvidersTool,
} from '../../tools/image-gen.js'
import type { FeatureToolRegistrar } from '../types.js'

export const registerTools: FeatureToolRegistrar = (registry) => {
  registry.register(createImageGenProvidersTool())
  registry.register(createImageGenCreateTool())
  registry.register(createImageGenJobTool())
  registry.register(createImageGenFileTool())
  registry.register(createImageGenCancelTool())
}
