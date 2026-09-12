import { createMediaSpeakTool, createMediaTranscribeTool } from '../../tools/media.js'
import type { FeatureToolRegistrar } from '../types.js'

export const registerTools: FeatureToolRegistrar = (registry) => {
  registry.register(createMediaTranscribeTool())
  registry.register(createMediaSpeakTool())
}
