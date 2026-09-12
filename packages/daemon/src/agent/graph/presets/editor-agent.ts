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
    // Source-editing scope: file + patch tools plus post-edit code analysis.
    // No terminal.run or browser tools — an editor agent edits, it does not shell out.
    toolAllowlist: ['fs.*', 'apply_patch', 'code.*'],
    // Editing-heavy preset — analyse blast radius + diagnostics after
    // every tool batch so the next turn revises with grounded signal.
    enablePostEditAnalysis: true,
  })
}
