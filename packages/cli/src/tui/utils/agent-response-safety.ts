// Thin cli wrapper over the shared api-client sanitizer. Kept so existing
// tui callers/tests keep their import path; all logic lives in api-client
// so web/desktop/mobile apply the same protocol stripping.
import {
  sanitizeAssistantContent,
  type SanitizeAssistantContentOptions,
} from '@sepilotd/api-client'

export function sanitizeTuiAgentResponse(
  responseText: string,
  options?: SanitizeAssistantContentOptions,
): {
  text: string
  internalFallback: boolean
} {
  return sanitizeAssistantContent(responseText, options)
}
