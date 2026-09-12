export type ErrorCode =
  | 'INVALID_REQUEST'
  | 'NOT_FOUND'
  | 'UNAUTHORIZED'
  | 'FORBIDDEN'
  | 'CONFLICT'
  | 'RATE_LIMITED'
  | 'CONTEXT_LENGTH'
  | 'CONTENT_FILTER'
  | 'PROVIDER_ERROR'
  | 'TIMEOUT'
  | 'INTERNAL_ERROR'
  | 'SERVICE_UNAVAILABLE'
  | 'SPEND_BUDGET_EXCEEDED'
  | 'WORKSPACE_BOUNDARY'

export interface ApiError {
  code: ErrorCode
  message: string
  details?: Record<string, unknown>
  requestId?: string
}
