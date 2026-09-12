import { createCodeDependenciesTool, createCodeSymbolsTool } from '../../tools/code-analysis.js'
import type { FeatureToolRegistrar } from '../types.js'

export const registerTools: FeatureToolRegistrar = (registry) => {
  registry.register(createCodeSymbolsTool())
  registry.register(createCodeDependenciesTool())
}
