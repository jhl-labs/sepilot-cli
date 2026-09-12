export const A2A_PROTOCOL_VERSION = '1.0'
export const A2A_SUPPORTED_VERSIONS = new Set(['0.3', A2A_PROTOCOL_VERSION])

export type A2ARole = 'ROLE_USER' | 'ROLE_AGENT'

export type A2ATaskState =
  | 'TASK_STATE_SUBMITTED'
  | 'TASK_STATE_WORKING'
  | 'TASK_STATE_COMPLETED'
  | 'TASK_STATE_FAILED'
  | 'TASK_STATE_CANCELED'
  | 'TASK_STATE_INPUT_REQUIRED'
  | 'TASK_STATE_REJECTED'
  | 'TASK_STATE_AUTH_REQUIRED'

export type A2APart =
  | { text: string; mediaType?: string; metadata?: Record<string, unknown> }
  | { data: unknown; mediaType?: string; metadata?: Record<string, unknown> }
  | { url: string; filename?: string; mediaType?: string; metadata?: Record<string, unknown> }
  | { raw: string; filename?: string; mediaType?: string; metadata?: Record<string, unknown> }

export interface A2AMessage {
  messageId: string
  contextId?: string
  taskId?: string
  role: A2ARole
  parts: A2APart[]
  metadata?: Record<string, unknown>
  extensions?: string[]
  referenceTaskIds?: string[]
}

export interface A2AArtifact {
  artifactId: string
  name?: string
  description?: string
  parts: A2APart[]
  metadata?: Record<string, unknown>
  extensions?: string[]
}

export interface A2ATaskStatus {
  state: A2ATaskState
  message?: A2AMessage
  timestamp?: string
}

export interface A2ATask {
  id: string
  contextId?: string
  status: A2ATaskStatus
  artifacts?: A2AArtifact[]
  history?: A2AMessage[]
  metadata?: Record<string, unknown>
}

export interface A2ASendMessageRequest {
  tenant?: string
  message: A2AMessage
  configuration?: {
    acceptedOutputModes?: string[]
    historyLength?: number
    returnImmediately?: boolean
    taskPushNotificationConfig?: unknown
  }
  metadata?: Record<string, unknown>
}

export interface A2ASendMessageResponse {
  task?: A2ATask
  message?: A2AMessage
}

export interface A2AListTasksRequest {
  tenant?: string
  contextId?: string
  status?: A2ATaskState
  pageSize?: number
  pageToken?: string
  historyLength?: number
  statusTimestampAfter?: string
  includeArtifacts?: boolean
}

export interface A2AListTasksResponse {
  tasks: A2ATask[]
  nextPageToken: string
  pageSize: number
  totalSize: number
}

export interface A2ATaskStatusUpdateEvent {
  taskId: string
  contextId: string
  status: A2ATaskStatus
  final?: boolean
  metadata?: Record<string, unknown>
}

export interface A2ATaskArtifactUpdateEvent {
  taskId: string
  contextId: string
  artifact: A2AArtifact
  append?: boolean
  lastChunk?: boolean
  metadata?: Record<string, unknown>
}

export type A2AStreamResponse =
  | { task: A2ATask }
  | { message: A2AMessage }
  | { statusUpdate: A2ATaskStatusUpdateEvent }
  | { artifactUpdate: A2ATaskArtifactUpdateEvent }

export interface A2AAuthenticationInfo {
  scheme: string
  credentials?: string
}

export interface A2APushNotificationConfig {
  id: string
  url: string
  authentication?: A2AAuthenticationInfo
  metadata?: Record<string, unknown>
}

export interface A2ATaskPushNotificationConfig {
  taskId: string
  pushNotificationConfig: A2APushNotificationConfig
}

export interface A2AListPushNotificationConfigsRequest {
  tenant?: string
  taskId: string
  pageSize?: number
  pageToken?: string
}

export interface A2AListPushNotificationConfigsResponse {
  configs: A2ATaskPushNotificationConfig[]
  nextPageToken: string
}

export interface A2AAgentCard {
  name: string
  description: string
  supportedInterfaces: Array<{
    url: string
    protocolBinding: 'JSONRPC' | 'GRPC' | 'HTTP+JSON' | string
    protocolVersion: string
    tenant?: string
  }>
  provider?: {
    url: string
    organization: string
  }
  version: string
  documentationUrl?: string
  capabilities: {
    streaming?: boolean
    pushNotifications?: boolean
    extensions?: Array<{
      uri?: string
      description?: string
      required?: boolean
      params?: Record<string, unknown>
    }>
    extendedAgentCard?: boolean
  }
  securitySchemes?: Record<string, unknown>
  securityRequirements?: Array<Record<string, string[]>>
  defaultInputModes: string[]
  defaultOutputModes: string[]
  skills: Array<{
    id: string
    name: string
    description: string
    tags: string[]
    examples?: string[]
    inputModes?: string[]
    outputModes?: string[]
    securityRequirements?: Array<Record<string, string[]>>
  }>
  signatures?: unknown[]
  iconUrl?: string
}

export interface A2AJsonRpcRequest {
  jsonrpc: '2.0'
  id?: string | number | null
  method: string
  params?: unknown
}

export interface A2AJsonRpcResponse {
  jsonrpc: '2.0'
  id?: string | number | null
  result?: unknown
  error?: {
    code: number
    message: string
    data?: unknown
  }
}
