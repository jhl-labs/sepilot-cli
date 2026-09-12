import { createComputerUseTools } from '../../tools/computer-use.js'
import type { FeatureToolRegistrar } from '../types.js'

export const registerTools: FeatureToolRegistrar = (registry) => {
  for (const tool of createComputerUseTools()) {
    registry.register(tool)
  }
}
