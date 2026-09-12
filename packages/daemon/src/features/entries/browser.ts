import { createRemoteBrowserTools, remoteBrowserBridge } from '../../tools/browser-remote.js'
import {
  createBrowserClickTool,
  createBrowserEvaluateTool,
  createBrowserExtractTool,
  createBrowserNavigateTool,
  createBrowserScreenshotTool,
} from '../../tools/browser.js'
import type { FeatureToolRegistrar } from '../types.js'

export const registerTools: FeatureToolRegistrar = (registry, deps) => {
  for (const tool of createRemoteBrowserTools(remoteBrowserBridge(registry))) registry.register(tool)
  const browserOptions = {
    egressAllowlist: deps.config.security.egressAllowlist,
    isLocalhostAllowed: deps.isManagedLoopbackUrl,
    managedLoopbackSocketForUrl: deps.managedLoopbackSocketForUrl,
  }
  try {
    registry.register(createBrowserNavigateTool(browserOptions))
    registry.register(createBrowserScreenshotTool(browserOptions))
    registry.register(createBrowserClickTool(browserOptions))
    registry.register(createBrowserEvaluateTool(browserOptions))
    registry.register(createBrowserExtractTool(browserOptions))
  } catch {
    // Playwright is optional at runtime.
  }
}
