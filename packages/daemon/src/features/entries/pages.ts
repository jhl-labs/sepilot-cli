import {
  createPagesScaffoldTool,
  createPagesScanTool,
  createPagesStatusTool,
} from '../../tools/pages-studio.js'
import type { FeatureToolRegistrar } from '../types.js'

export const registerTools: FeatureToolRegistrar = (registry) => {
  registry.register(createPagesScaffoldTool())
  registry.register(createPagesStatusTool())
  registry.register(createPagesScanTool())
}
