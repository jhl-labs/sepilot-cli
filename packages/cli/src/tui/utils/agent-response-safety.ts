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
