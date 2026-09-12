import { createMarketQuoteTool } from '../../tools/market-quote.js'
import type { FeatureToolRegistrar } from '../types.js'

export const registerTools: FeatureToolRegistrar = (registry) => {
  registry.register(createMarketQuoteTool())
}
