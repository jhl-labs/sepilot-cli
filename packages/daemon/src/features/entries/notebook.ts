import { createNotebookInspectTool } from '../../tools/notebook.js'
import type { FeatureToolRegistrar } from '../types.js'

export const registerTools: FeatureToolRegistrar = (registry) => {
  registry.register(createNotebookInspectTool())
}
