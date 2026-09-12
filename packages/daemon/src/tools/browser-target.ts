import { ToolRegistry } from './registry.js'

const HEADLESS_BROWSER_TOOLS = new Set([
  'browser.navigate', 'browser.extract', 'browser.screenshot', 'browser.click', 'browser.evaluate',
])

/** Retain the selected tab as the default across transfers without removing explicit headless work.
 * Selection is an execution target, not an authorization grant. Search and fetch stay independent.
 */
export function withSelectedBrowserTarget(source: ToolRegistry, attached: boolean): ToolRegistry {
  if (!attached) return source
  const selected = new ToolRegistry()
  for (const tool of source.list()) {
    const registration = source.registrationSource(tool.name)
    selected.register(HEADLESS_BROWSER_TOOLS.has(tool.name) ? {
      ...tool,
      description: 'A visible Chrome/Edge tab is attached to this chat. Use browser.remote_snapshot and browser.remote_action for that tab. This tool operates a SEPARATE headless browser without the attached profile or login. Only set browserTarget=headless when the user explicitly requests separate/headless execution; never use it as a silent fallback from an attached-tab failure. web.search and webfetch remain independent. ' + tool.description,
      inputSchema: {
        ...tool.inputSchema,
        properties: {
          ...(tool.inputSchema.properties as Record<string, unknown> ?? {}),
          browserTarget: { type: 'string', enum: ['headless'], description: 'Explicit separate/headless browser target requested by the user. Omit for connected-tab work and use browser.remote_* instead.' },
        },
      },
      execute: async (input, context) => {
        if (input.browserTarget !== 'headless') return {
          status: 'error', durationMs: 0, code: 'BROWSER_TARGET_REQUIRED',
          output: 'This chat has a selected visible browser. Inspect it with browser.remote_snapshot and act with browser.remote_action. This headless tool did not run. Set browserTarget=headless only for an explicit user request to use a separate headless browser; do not silently switch after a connection or control error.',
        }
        const { browserTarget: _target, ...args } = input
        return tool.execute(args, context)
      },
    } : tool, registration ? { source: registration } : undefined)
  }
  return selected
}
