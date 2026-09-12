import { createOfficeTools } from '../../tools/office.js'
import type { FeatureToolRegistrar } from '../types.js'

export const registerTools: FeatureToolRegistrar = (registry) => {
  for (const tool of createOfficeTools()) {
    registry.register(tool)
  }
}
