import type { Deps } from '../nodes.js'
import { buildFocusedLoopGraph } from './focused-loop.js'

const editorAgentPrompt = [
  'You are an editor agent.',
  'Prefer fs.read, apply_patch, fs.write, and lsp-style code inspection tools for source changes.',
  'Read the smallest useful scope first, edit precisely, and avoid shelling out when the editing tool family can do the job directly.',
].join(' ')

export function buildEditorAgentGraph(deps: Deps) {
  return buildFocusedLoopGraph(deps, {
    systemPrompt: editorAgentPrompt,
    toolAllowlist: ['fs.*', 'apply_patch', 'code.*'],
    enablePostEditAnalysis: true,
  })
}
