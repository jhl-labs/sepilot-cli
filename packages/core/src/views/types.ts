export type LiveViewKind = 'browser_tab' | 'embedded_browser' | 'screen' | 'app_window'
export type LiveViewController = 'agent' | 'human'

export interface LiveView {
  id: string
  kind: LiveViewKind
  title: string
  subtitle?: string
  sessionId?: string
  controller: LiveViewController
  createdAt: number
  lastFrameAt?: number
}

export interface LiveViewStill {
  viewId: string
  image: string
  width: number
  height: number
  capturedAt: number
}

export interface LiveViewRecord {
  id: string
  kind: LiveViewKind
  title: string
  subtitle?: string
  still: LiveViewStill
}
