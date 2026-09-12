import type { ChatStreamOptions, DaemonClient } from './http.js'
import type {
  ActivityItem,
  AgentState,
  Message as SurfaceMessage,
  StreamEventControllerBindings,
  StreamEventControllerOptions,
} from './chat-surface-types.js'
import type {
  DaemonArtifact,
  DaemonChatStreamPayload,
  DaemonSessionsWatchPayload,
  DaemonSessionWatchPayload,
} from './types.js'
import type { TokenSpeedTracker } from './token-speed.js'

export type DaemonStreamEventHandler<T> = (
  event: T,
) => void | Promise<void>

export type DaemonChatEventHandler = DaemonStreamEventHandler<DaemonChatStreamPayload>
export type DaemonSessionsWatchEventHandler =
  DaemonStreamEventHandler<DaemonSessionsWatchPayload>
export type DaemonSessionWatchEventHandler =
  DaemonStreamEventHandler<DaemonSessionWatchPayload>

export interface SurfaceStreamRunParams {
  bindings: StreamEventControllerBindings
  controllerOptions: Omit<StreamEventControllerOptions, 'assistantId' | 'createId'>
  createId: () => string
  createSeedMessages?: (assistantId: string) => SurfaceMessage[]
  pushActivity: (item: ActivityItem) => void
  run: (onEvent: DaemonChatEventHandler) => Promise<void>
  onSession?: (sessionId: string) => void | Promise<void>
  onArtifacts?: (artifacts: DaemonArtifact[]) => void | Promise<void>
  tokenSpeedTracker?: TokenSpeedTracker
  formatRunError: (error: unknown) => string
  errorAssistantFallbackText: string
  rethrow?: boolean
}

export interface ManagedSurfaceStreamLifecycleParams {
  setLoading: (loading: boolean) => void
  beforeStart?: () => void | Promise<void>
  afterFinish?: () => void | Promise<void>
  statusText: string | null
  agentState: AgentState
  activityLabel: string
  activityDetail: string
  activityKind?: ActivityItem['kind']
  activityStatus?: ActivityItem['status']
  clearError?: boolean
  resetStatusOnFinish?: boolean
}

export interface ManagedSurfaceStreamRunParams extends SurfaceStreamRunParams {
  lifecycle: ManagedSurfaceStreamLifecycleParams
}

export interface ManagedSurfaceStreamRunnerBaseParams {
  bindings: StreamEventControllerBindings
  createId: () => string
  pushActivity: (item: ActivityItem) => void
  onSession?: (sessionId: string) => void | Promise<void>
  onArtifacts?: (artifacts: DaemonArtifact[]) => void | Promise<void>
  tokenSpeedTracker?: TokenSpeedTracker
  maxActivityItems?: number
  formatRunError: (error: unknown) => string
  formatErrorMessage?: (message: string) => string
}

export interface ManagedSurfaceStreamRunnerParams {
  lifecycle: ManagedSurfaceStreamLifecycleParams
  doneFallbackText: string
  errorFallbackText: string
  errorLabel: string
  /** See StreamEventControllerOptions.onEmptyCompletion. */
  onEmptyCompletion?: () => void
  approvalResumeAvailable?: boolean
  createSeedMessages?: (assistantId: string) => SurfaceMessage[]
  errorAssistantFallbackText: string
  rethrow?: boolean
  run: (onEvent: DaemonChatEventHandler) => Promise<void>
}

export interface BrowserWebSocketMessageEvent {
  data: unknown
}

export interface BrowserWebSocketCloseEvent {
  code?: number
  reason?: string
  wasClean?: boolean
}

export interface BrowserWebSocketErrorEvent {
  message?: string
}

export type BrowserWebSocketEventMap = {
  open: Event
  message: BrowserWebSocketMessageEvent
  error: BrowserWebSocketErrorEvent
  close: BrowserWebSocketCloseEvent
}

export interface BrowserWebSocketLike {
  addEventListener<K extends keyof BrowserWebSocketEventMap>(
    type: K,
    listener: (event: BrowserWebSocketEventMap[K]) => void,
  ): void
  removeEventListener<K extends keyof BrowserWebSocketEventMap>(
    type: K,
    listener: (event: BrowserWebSocketEventMap[K]) => void,
  ): void
  close(): void
  send(data: string): void
}

export interface BrowserWebSocketRef {
  current: BrowserWebSocketLike | null
}

export type BrowserWebSocketFactory = (url: string) => BrowserWebSocketLike

export interface BrowserWebSocketChatParams {
  baseUrl: string
  message: string
  sessionId?: string
  options?: ChatStreamOptions
  socketRef?: BrowserWebSocketRef
  socketFactory?: BrowserWebSocketFactory
  signal?: AbortSignal
  onEvent: DaemonChatEventHandler
}

export interface SseChatParams {
  client: Pick<DaemonClient, 'chatStream'>
  message: string
  sessionId?: string
  options?: ChatStreamOptions
  signal?: AbortSignal
  onEvent: DaemonChatEventHandler
}

export interface SessionWatchParams {
  client: Pick<DaemonClient, 'watchSessionStream'>
  sessionId: string
  onEvent: DaemonSessionWatchEventHandler
}

export interface SessionsWatchParams {
  client: Pick<DaemonClient, 'watchSessionsStream'>
  query?: string
  options?: { page?: number; perPage?: number }
  onEvent: DaemonSessionsWatchEventHandler
}
