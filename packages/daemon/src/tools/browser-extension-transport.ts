import type {
  BrowserTransport,
  RemoteAction,
  RemoteFrame,
  RemoteSnapshot,
} from './browser-transport.js'

const EXTENSION_ACTIONS: ReadonlySet<string> = new Set([
  'snapshot',
  'navigate',
  'click',
  'type',
  'scroll',
  'key',
])

export class ExtensionTransport implements BrowserTransport {
  readonly kind = 'extension' as const
  readonly capabilities = { refs: false, console: false, network: false }
  private frame?: RemoteFrame
  private readonly frameListeners = new Set<(frame: RemoteFrame) => void>()
  private readonly closedListeners = new Set<(reason: string) => void>()

  constructor(
    private readonly dispatch: (action: RemoteAction, signal?: AbortSignal) => Promise<string>,
  ) {}

  currentUrl() {
    return this.frame?.url
  }

  latestFrame() {
    return this.frame
  }

  receiveFrame(frame: RemoteFrame) {
    this.frame = frame
    for (const listener of this.frameListeners) listener(frame)
  }

  async snapshot(signal?: AbortSignal): Promise<RemoteSnapshot> {
    const text = await this.dispatch({ action: 'snapshot' }, signal)
    return {
      url: this.frame?.url ?? '',
      title: this.frame?.title ?? '',
      text,
      elements: [],
      image: this.frame?.image,
      snapshotId: '',
    }
  }

  act(action: RemoteAction, signal?: AbortSignal) {
    if (action.action === 'click' && (action.x === undefined || action.y === undefined) && !action.ref)
      return Promise.reject(new Error('click through the extension needs x and y from the snapshot'))
    if (action.action === 'scroll' && action.deltaY === undefined && !action.ref)
      return Promise.reject(new Error('scroll through the extension needs deltaY'))
    if (!EXTENSION_ACTIONS.has(action.action) || ('ref' in action && action.ref)) {
      return Promise.reject(
        new Error(
          `UNSUPPORTED_ACTION: this tab is attached through the browser extension, which supports ${[...EXTENSION_ACTIONS].join(', ')} with viewport coordinates only. Use coordinates from browser.remote_snapshot, or ask the user to connect Chrome directly for element refs, console and network.`,
        ),
      )
    }
    return this.dispatch(action, signal)
  }

  onFrame(listener: (frame: RemoteFrame) => void) {
    this.frameListeners.add(listener)
    return () => {
      this.frameListeners.delete(listener)
    }
  }

  onUserInput() {
    return () => {}
  }

  onClosed(listener: (reason: string) => void) {
    this.closedListeners.add(listener)
    return () => {
      this.closedListeners.delete(listener)
    }
  }

  async close() {}
}
