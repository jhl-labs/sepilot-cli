import { randomUUID } from 'node:crypto'
import { WebSocketCdpClient, type CdpClient, type CdpConnectFn } from './browser-cdp-client.js'
import type {
  BrowserTransport,
  RemoteAction,
  RemoteElement,
  RemoteFrame,
  RemoteSnapshot,
  StillFormat,
} from './browser-transport.js'

export type { CdpConnectFn } from './browser-cdp-client.js'

export interface CdpTab {
  targetId: string
  url: string
  title: string
}

const USER_INPUT_BINDING = '__sepilotUserInput'
const USER_INPUT_MASK_MS = 300
const MAX_FRAME_BASE64 = Math.ceil((1_000_000 * 4) / 3)
const MAX_ELEMENTS = 120
const MAX_TEXT_CHARS = 6_000
const MAX_CONSOLE = 100
const MAX_NETWORK = 50
const WAIT_POLL_MS = 200
const SCREENSHOT_TIMEOUT_MS = 5_000
const NAVIGATION_TIMEOUT_MS = 10_000
const NAVIGATION_QUIET_MS = 300
const MAX_OPENED_TABS = 10
const SECRET_PARAM = /\b(token|access_token|refresh_token|id_token|code|key|api_key|apikey|secret|password|passwd|auth|authorization|session|sig|signature)=([^\s&"']+)/gi
const SECRET_BARE = /\b(sk|pk|rk|ak)_(live|test|prod)?_?[A-Za-z0-9]{8,}\b/g
const INTERACTIVE_ROLES = new Set([
  'button', 'link', 'textbox', 'searchbox', 'combobox', 'listbox', 'option', 'checkbox', 'radio',
  'switch', 'slider', 'spinbutton', 'menuitem', 'menuitemcheckbox', 'menuitemradio', 'tab',
  'treeitem', 'heading', 'cell', 'gridcell', 'row', 'textarea', 'img', 'image', 'progressbar',
  'scrollbar', 'tabpanel', 'dialog', 'alert', 'status', 'listitem', 'article', 'region',
])
const USER_INPUT_INIT_SCRIPT = `(() => {
  const report = (kind) => (event) => {
    if (event.isTrusted && typeof window.${USER_INPUT_BINDING} === 'function') window.${USER_INPUT_BINDING}(kind)
  }
  for (const kind of ['keydown', 'mousedown', 'wheel'])
    window.addEventListener(kind, report(kind), { capture: true, passive: true })
})()`

export function redactConsoleText(text: string): string {
  return text.replace(SECRET_PARAM, '$1=[REDACTED]').replace(SECRET_BARE, '[REDACTED]')
}

export function stripUrlSecrets(url: string): string {
  try {
    const parsed = new URL(url)
    parsed.search = ''
    parsed.hash = ''
    return parsed.toString()
  } catch {
    return url.split(/[?#]/)[0] ?? url
  }
}

function isSharablePageUrl(url: string): boolean {
  return /^https?:\/\//i.test(url)
}

interface KeyDescriptor {
  key: string
  code: string
  keyCode: number
  text?: string
}
const NAMED_KEYS: Record<string, KeyDescriptor> = {
  Enter: { key: 'Enter', code: 'Enter', keyCode: 13, text: '\r' },
  Tab: { key: 'Tab', code: 'Tab', keyCode: 9 },
  Escape: { key: 'Escape', code: 'Escape', keyCode: 27 },
  Backspace: { key: 'Backspace', code: 'Backspace', keyCode: 8 },
  Delete: { key: 'Delete', code: 'Delete', keyCode: 46 },
  ArrowUp: { key: 'ArrowUp', code: 'ArrowUp', keyCode: 38 },
  ArrowDown: { key: 'ArrowDown', code: 'ArrowDown', keyCode: 40 },
  ArrowLeft: { key: 'ArrowLeft', code: 'ArrowLeft', keyCode: 37 },
  ArrowRight: { key: 'ArrowRight', code: 'ArrowRight', keyCode: 39 },
  Home: { key: 'Home', code: 'Home', keyCode: 36 },
  End: { key: 'End', code: 'End', keyCode: 35 },
  PageUp: { key: 'PageUp', code: 'PageUp', keyCode: 33 },
  PageDown: { key: 'PageDown', code: 'PageDown', keyCode: 34 },
  Space: { key: ' ', code: 'Space', keyCode: 32, text: ' ' },
}
const MODIFIER_BITS: Record<string, number> = { Alt: 1, Control: 2, Meta: 4, Shift: 8 }

export function describeKey(spec: string): { descriptor: KeyDescriptor; modifiers: number } {
  const parts = spec.split('+').filter(Boolean)
  const keyName = parts.pop() ?? ''
  let modifiers = 0
  for (const part of parts) {
    const bit = MODIFIER_BITS[part] ?? MODIFIER_BITS[part.charAt(0).toUpperCase() + part.slice(1).toLowerCase()]
    if (bit === undefined) throw new Error(`Unknown modifier '${part}'`)
    modifiers |= bit
  }
  const named = NAMED_KEYS[keyName]
  if (named) return { descriptor: named, modifiers }
  if ([...keyName].length !== 1) throw new Error(`Unknown key '${keyName}'`)
  const upper = keyName.toUpperCase()
  const descriptor: KeyDescriptor = {
    key: keyName,
    code: /^[a-z]$/i.test(keyName) ? `Key${upper}` : /^[0-9]$/.test(keyName) ? `Digit${keyName}` : '',
    keyCode: upper.charCodeAt(0),
    text: modifiers & (MODIFIER_BITS.Control! | MODIFIER_BITS.Meta! | MODIFIER_BITS.Alt!) ? undefined : keyName,
  }
  return { descriptor, modifiers }
}

async function defaultConnect(wsEndpoint: string): Promise<CdpClient> {
  return WebSocketCdpClient.connect(wsEndpoint)
}

export class CdpConnection {
  private constructor(
    private readonly client: CdpClient,
    readonly wsEndpoint: string,
  ) {}

  static async open(wsEndpoint: string, connect: CdpConnectFn = defaultConnect): Promise<CdpConnection> {
    return new CdpConnection(await connect(wsEndpoint), wsEndpoint)
  }

  async listTabs(): Promise<CdpTab[]> {
    const result = (await this.client.send('Target.getTargets')) as {
      targetInfos?: { targetId: string; type: string; url: string; title?: string }[]
    }
    return (result.targetInfos ?? [])
      .filter((target) => target.type === 'page' && isSharablePageUrl(target.url))
      .map((target) => ({ targetId: target.targetId, url: target.url, title: (target.title ?? '').slice(0, 500) }))
  }

  async attach(targetId: string): Promise<CdpTransport> {
    const tab = (await this.listTabs()).find((candidate) => candidate.targetId === targetId)
    if (!tab) {
      await this.close()
      throw new Error('TAB_NOT_FOUND: the selected tab moved or closed; list tabs again')
    }
    try {
      return await CdpTransport.create(this.client, tab)
    } catch (error) {
      await this.close()
      throw error
    }
  }

  async close(): Promise<void> {
    await this.client.close().catch(() => {})
  }
}

interface ConsoleEntry {
  level: string
  text: string
}
interface NetworkEntry {
  method: string
  url: string
  status: number
  resourceType: string
}

export class CdpTransport implements BrowserTransport {
  readonly kind = 'cdp' as const
  readonly capabilities = { refs: true, console: true, network: true }
  private url: string
  private title: string
  private sessionId = ''
  private snapshotId: string | null = null
  private refs = new Map<string, number>()
  private lastAgentActionAt = 0
  private actionInFlight = 0
  private closed = false
  private screencastRunning = false
  private screencastQueue: Promise<void> = Promise.resolve()
  private readonly consoleLog: ConsoleEntry[] = []
  private readonly networkLog: NetworkEntry[] = []
  private readonly requestMethods = new Map<string, string>()
  private readonly openedTabs: string[] = []
  private readonly frameListeners = new Set<(frame: RemoteFrame) => void>()
  private readonly inputListeners = new Set<() => void>()
  private readonly closedListeners = new Set<(reason: string) => void>()
  private readonly unsubscribers: (() => void)[] = []
  private readonly navigationListeners = new Set<(url: string, error: boolean) => void>()

  private constructor(
    private readonly client: CdpClient,
    tab: CdpTab,
    private readonly targetId = tab.targetId,
  ) {
    this.url = tab.url
    this.title = tab.title
  }

  static async create(client: CdpClient, tab: CdpTab): Promise<CdpTransport> {
    const transport = new CdpTransport(client, tab)
    await transport.install()
    return transport
  }

  private page(method: string, params: Record<string, unknown> = {}, timeoutMs?: number): Promise<any> {
    return this.client.send(method, params, this.sessionId, timeoutMs)
  }

  private async activate(): Promise<void> {
    await this.client.send('Target.activateTarget', { targetId: this.targetId }).catch(() => {})
  }

  private listen(event: string, listener: (params: Record<string, unknown>) => void): void {
    this.unsubscribers.push(
      this.client.on(event, (params, sessionId) => {
        if (sessionId === this.sessionId) listener(params)
      }),
    )
  }

  private async install(): Promise<void> {
    const attached = (await this.client.send('Target.attachToTarget', { targetId: this.targetId, flatten: true })) as {
      sessionId: string
    }
    this.sessionId = attached.sessionId
    this.unsubscribers.push(this.client.onClose(() => this.end('Browser disconnected')))
    this.unsubscribers.push(
      this.client.on('Target.detachedFromTarget', (params) => {
        if (params.sessionId === this.sessionId || params.targetId === this.targetId) this.end('Tab closed')
      }),
    )
    this.unsubscribers.push(
      this.client.on('Target.targetDestroyed', (params) => {
        if (params.targetId === this.targetId) this.end('Tab closed')
      }),
    )
    this.unsubscribers.push(
      this.client.on('Target.targetCreated', (params) => {
        const info = params.targetInfo as { type?: string; url?: string; openerId?: string } | undefined
        if (info?.openerId !== this.targetId || info.type !== 'page' || !info.url) return
        this.openedTabs.push(info.url.slice(0, 500))
        if (this.openedTabs.length > MAX_OPENED_TABS) this.openedTabs.splice(0, this.openedTabs.length - MAX_OPENED_TABS)
      }),
    )
    this.listen('Page.frameNavigated', (params) => {
      const frame = params.frame as { parentId?: string; url?: string } | undefined
      if (!frame || frame.parentId || !frame.url) return
      if (frame.url.startsWith('chrome-error://')) {
        for (const listener of this.navigationListeners) listener(frame.url, true)
        return
      }
      this.url = frame.url
      for (const listener of this.navigationListeners) listener(frame.url, false)
      if (!isSharablePageUrl(this.url)) this.end('Tab navigated to a restricted page')
    })
    this.listen('Page.loadEventFired', () => {
      void this.refreshTitle()
    })
    this.listen('Runtime.consoleAPICalled', (params) => {
      const args = (params.args as { type?: string; value?: unknown; description?: string }[] | undefined) ?? []
      const text = args
        .map((arg) => (arg.value !== undefined ? String(arg.value) : (arg.description ?? arg.type ?? '')))
        .join(' ')
      this.consoleLog.push({ level: String(params.type ?? 'log'), text: redactConsoleText(text.slice(0, 1000)) })
      if (this.consoleLog.length > MAX_CONSOLE) this.consoleLog.splice(0, this.consoleLog.length - MAX_CONSOLE)
    })
    this.listen('Network.requestWillBeSent', (params) => {
      const request = params.request as { method?: string } | undefined
      if (typeof params.requestId === 'string') this.requestMethods.set(params.requestId, request?.method ?? 'GET')
    })
    this.listen('Network.responseReceived', (params) => {
      const response = params.response as { url?: string; status?: number } | undefined
      const requestId = typeof params.requestId === 'string' ? params.requestId : ''
      this.networkLog.push({
        method: this.requestMethods.get(requestId) ?? 'GET',
        url: stripUrlSecrets(response?.url ?? '').slice(0, 2000),
        status: Number(response?.status ?? 0),
        resourceType: String(params.type ?? ''),
      })
      this.requestMethods.delete(requestId)
      if (this.networkLog.length > MAX_NETWORK) this.networkLog.splice(0, this.networkLog.length - MAX_NETWORK)
    })
    this.listen('Runtime.bindingCalled', (params) => {
      if (params.name === USER_INPUT_BINDING) this.reportUserInput()
    })
    this.listen('Page.screencastFrame', (params) => {
      void this.page('Page.screencastFrameAck', { sessionId: params.sessionId }).catch(() => {})
      const data = typeof params.data === 'string' ? params.data : ''
      if (!data || data.length > MAX_FRAME_BASE64) return
      const frame: RemoteFrame = {
        url: this.url,
        title: this.title,
        image: `data:image/jpeg;base64,${data}`,
        capturedAt: Date.now(),
      }
      for (const listener of this.frameListeners) listener(frame)
    })
    await this.page('Page.enable')
    await this.page('Runtime.enable')
    await this.page('Network.enable')
    await this.page('DOM.enable')
    await this.page('Accessibility.enable')
    await this.page('Runtime.addBinding', { name: USER_INPUT_BINDING })
    await this.page('Page.addScriptToEvaluateOnNewDocument', { source: USER_INPUT_INIT_SCRIPT })
    await this.page('Runtime.evaluate', { expression: USER_INPUT_INIT_SCRIPT }).catch(() => {})
    await this.client.send('Target.setDiscoverTargets', { discover: true }).catch(() => {})
  }

  private async refreshTitle(): Promise<void> {
    try {
      const info = (await this.client.send('Target.getTargetInfo', { targetId: this.targetId })) as {
        targetInfo?: { title?: string; url?: string }
      }
      if (info.targetInfo?.title !== undefined) this.title = info.targetInfo.title.slice(0, 500)
      if (info.targetInfo?.url) this.url = info.targetInfo.url
    } catch {
    }
  }

  private reportUserInput(): void {
    if (this.actionInFlight > 0 || Date.now() - this.lastAgentActionAt < USER_INPUT_MASK_MS) return
    for (const listener of this.inputListeners) listener()
  }

  private end(reason: string): void {
    if (this.closed) return
    this.closed = true
    for (const listener of this.closedListeners) listener(reason)
    this.frameListeners.clear()
    this.inputListeners.clear()
    this.closedListeners.clear()
    for (const unsubscribe of this.unsubscribers.splice(0)) unsubscribe()
    void this.client.close().catch(() => {})
  }

  currentUrl(): string | undefined {
    return this.closed ? undefined : this.url
  }

  private syncScreencast(): Promise<void> {
    this.screencastQueue = this.screencastQueue
      .then(async () => {
        const wanted = this.frameListeners.size > 0 && !this.closed
        if (wanted === this.screencastRunning) return
        if (wanted) {
          await this.page('Page.startScreencast', {
            format: 'jpeg',
            quality: 60,
            maxWidth: 1280,
            maxHeight: 1280,
            everyNthFrame: 2,
          })
        } else {
          await this.page('Page.stopScreencast')
        }
        this.screencastRunning = wanted
      })
      .catch(() => {})
    return this.screencastQueue
  }

  onFrame(listener: (frame: RemoteFrame) => void): () => void {
    this.frameListeners.add(listener)
    void this.syncScreencast()
    return () => {
      this.frameListeners.delete(listener)
      void this.syncScreencast()
    }
  }

  onUserInput(listener: () => void): () => void {
    this.inputListeners.add(listener)
    return () => {
      this.inputListeners.delete(listener)
    }
  }

  onClosed(listener: (reason: string) => void): () => void {
    this.closedListeners.add(listener)
    return () => {
      this.closedListeners.delete(listener)
    }
  }

  private nodeFor(ref: string, snapshotId: string | undefined): number {
    const backendNodeId = this.refs.get(ref)
    if (!this.snapshotId || snapshotId !== this.snapshotId || backendNodeId === undefined) {
      throw new Error('Page changed; inspect again before acting')
    }
    return backendNodeId
  }

  private async hrefOf(backendNodeId: number): Promise<string | undefined> {
    try {
      const described = (await this.page('DOM.describeNode', { backendNodeId })) as {
        node?: { attributes?: string[] }
      }
      const attributes = described.node?.attributes ?? []
      const index = attributes.findIndex((value, i) => i % 2 === 0 && value.toLowerCase() === 'href')
      const raw = index >= 0 ? attributes[index + 1] : undefined
      if (!raw) return undefined
      return new URL(raw, this.url).toString().slice(0, 2000)
    } catch {
      return undefined
    }
  }

  private async centerOf(backendNodeId: number): Promise<{ x: number; y: number }> {
    const result = (await this.page('DOM.getBoxModel', { backendNodeId })) as {
      model?: { content?: number[] }
    }
    const quad = result.model?.content
    if (!quad || quad.length < 8) throw new Error('Element is not rendered; inspect again')
    const xs = [quad[0]!, quad[2]!, quad[4]!, quad[6]!]
    const ys = [quad[1]!, quad[3]!, quad[5]!, quad[7]!]
    return {
      x: (Math.min(...xs) + Math.max(...xs)) / 2,
      y: (Math.min(...ys) + Math.max(...ys)) / 2,
    }
  }

  async still(maxWidth: number, format: StillFormat = 'jpeg'): Promise<RemoteFrame | undefined> {
    try {
      const metrics = (await this.page('Page.getLayoutMetrics', {}, SCREENSHOT_TIMEOUT_MS)) as {
        cssVisualViewport?: { clientWidth?: number; clientHeight?: number; pageX?: number; pageY?: number }
      }
      const x = metrics.cssVisualViewport?.pageX ?? 0
      const y = metrics.cssVisualViewport?.pageY ?? 0
      const width = Math.max(1, Math.round(metrics.cssVisualViewport?.clientWidth ?? 1280))
      const height = Math.max(1, Math.round(metrics.cssVisualViewport?.clientHeight ?? 800))
      const scale = Math.min(1, maxWidth / width)
      const shot = (await this.page(
        'Page.captureScreenshot',
        { format, ...(format === 'jpeg' ? { quality: 55 } : {}), clip: { x, y, width, height, scale } },
        SCREENSHOT_TIMEOUT_MS,
      )) as { data?: string }
      if (!shot.data || shot.data.length > MAX_FRAME_BASE64) return undefined
      return {
        url: this.url,
        title: this.title,
        image: `data:image/${format};base64,${shot.data}`,
        capturedAt: Date.now(),
        width: Math.max(1, Math.round(width * scale)),
        height: Math.max(1, Math.round(height * scale)),
      }
    } catch {
      return undefined
    }
  }

  async snapshot(): Promise<RemoteSnapshot> {
    await this.activate()
    const tree = (await this.page('Accessibility.getFullAXTree')) as {
      nodes?: {
        ignored?: boolean
        role?: { value?: string }
        name?: { value?: string }
        value?: { value?: unknown }
        backendDOMNodeId?: number
      }[]
    }
    const snapshotId = randomUUID()
    const refs = new Map<string, number>()
    const elements: RemoteElement[] = []
    for (const node of tree.nodes ?? []) {
      if (elements.length >= MAX_ELEMENTS) break
      if (node.ignored || node.backendDOMNodeId === undefined) continue
      const role = String(node.role?.value ?? '')
      const name = String(node.name?.value ?? '')
      if (!INTERACTIVE_ROLES.has(role) || (!name && role !== 'textbox' && role !== 'searchbox' && role !== 'combobox')) continue
      let center: { x: number; y: number }
      try {
        center = await this.centerOf(node.backendDOMNodeId)
      } catch {
        continue
      }
      const ref = `e${elements.length + 1}`
      refs.set(ref, node.backendDOMNodeId)
      const element: RemoteElement = { ref, role, name, x: center.x, y: center.y }
      if (node.value?.value !== undefined) element.value = String(node.value.value)
      if (role === 'link') {
        const href = await this.hrefOf(node.backendDOMNodeId)
        if (href) element.href = href
      }
      elements.push(element)
    }
    this.snapshotId = snapshotId
    this.refs = refs
    let text = ''
    try {
      const evaluated = (await this.page('Runtime.evaluate', {
        expression: 'document.body ? document.body.innerText : ""',
        returnByValue: true,
      })) as { result?: { value?: unknown } }
      text = String(evaluated.result?.value ?? '')
    } catch {
      text = ''
    }
    let image: string | undefined
    try {
      const shot = (await this.page(
        'Page.captureScreenshot',
        { format: 'jpeg', quality: 60 },
        SCREENSHOT_TIMEOUT_MS,
      )) as { data?: string }
      if (shot.data && shot.data.length <= MAX_FRAME_BASE64) image = `data:image/jpeg;base64,${shot.data}`
    } catch {
      image = undefined
    }
    await this.refreshTitle()
    const opened = this.openedTabs.map(
      (url) => `Opened by this page since attach (not shared; navigate here to follow): ${url}`,
    )
    return {
      url: this.url,
      title: this.title,
      text: [...opened, text.slice(0, MAX_TEXT_CHARS)].join('\n'),
      elements,
      image,
      snapshotId,
    }
  }

  async act(action: RemoteAction, signal?: AbortSignal): Promise<string> {
    if (this.closed) throw new Error('Tab closed')
    if (action.action === 'console') return this.formatConsole(action.level)
    if (action.action === 'network') return this.formatNetwork(action.filter)
    if (action.action === 'snapshot' || action.action === 'handover') {
      throw new Error(`${action.action} is handled by the session hub`)
    }
    if (signal?.aborted) throw new Error('Browser action cancelled before it reached the tab')
    this.actionInFlight += 1
    this.lastAgentActionAt = Date.now()
    try {
      await this.activate()
      return await this.dispatch(action, signal)
    } finally {
      this.actionInFlight -= 1
      this.lastAgentActionAt = Date.now()
    }
  }

  private async mouse(type: 'mouseMoved' | 'mousePressed' | 'mouseReleased', x: number, y: number): Promise<void> {
    await this.page('Input.dispatchMouseEvent', {
      type,
      x,
      y,
      button: type === 'mouseMoved' ? 'none' : 'left',
      clickCount: type === 'mouseMoved' ? 0 : 1,
    })
  }

  private async clickAt(x: number, y: number, signal?: AbortSignal): Promise<void> {
    await this.mouse('mouseMoved', x, y)
    if (signal?.aborted) throw new Error('Browser action cancelled; inspect the page before continuing')
    await this.mouse('mousePressed', x, y)
    await this.mouse('mouseReleased', x, y)
  }

  private async pressKey(spec: string): Promise<void> {
    const { descriptor, modifiers } = describeKey(spec)
    const base = {
      key: descriptor.key,
      code: descriptor.code,
      windowsVirtualKeyCode: descriptor.keyCode,
      nativeVirtualKeyCode: descriptor.keyCode,
      modifiers,
    }
    await this.page('Input.dispatchKeyEvent', {
      ...base,
      type: descriptor.text ? 'keyDown' : 'rawKeyDown',
      ...(descriptor.text ? { text: descriptor.text, unmodifiedText: descriptor.text } : {}),
    })
    await this.page('Input.dispatchKeyEvent', { ...base, type: 'keyUp' })
  }

  private async callOnNode(backendNodeId: number, functionDeclaration: string, args: unknown[] = []): Promise<unknown> {
    const resolved = (await this.page('DOM.resolveNode', { backendNodeId })) as { object?: { objectId?: string } }
    if (!resolved.object?.objectId) throw new Error('Element is gone; inspect again')
    const result = (await this.page('Runtime.callFunctionOn', {
      objectId: resolved.object.objectId,
      functionDeclaration,
      arguments: args.map((value) => ({ value })),
      returnByValue: true,
    })) as { result?: { value?: unknown }; exceptionDetails?: { text?: string } }
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text ?? 'Element action failed')
    return result.result?.value
  }

  private waitForNavigation(): { promise: Promise<'navigated' | 'error' | 'timeout'>; cancel: () => void } {
    let quiet: ReturnType<typeof setTimeout> | undefined
    let settle: (outcome: 'navigated' | 'error' | 'timeout') => void = () => {}
    const listener = (_url: string, error: boolean) => {
      if (quiet) clearTimeout(quiet)
      if (error) {
        settle('error')
        return
      }
      quiet = setTimeout(() => settle('navigated'), NAVIGATION_QUIET_MS)
    }
    const deadline = setTimeout(() => settle('timeout'), NAVIGATION_TIMEOUT_MS)
    const promise = new Promise<'navigated' | 'error' | 'timeout'>((resolve) => {
      settle = (outcome) => {
        clearTimeout(deadline)
        if (quiet) clearTimeout(quiet)
        this.navigationListeners.delete(listener)
        resolve(outcome)
      }
    })
    this.navigationListeners.add(listener)
    return { promise, cancel: () => settle('timeout') }
  }

  private async waitUntil(check: () => Promise<boolean>, timeoutMs: number, label: string): Promise<void> {
    const deadline = Date.now() + timeoutMs
    for (;;) {
      if (await check()) return
      if (Date.now() >= deadline) throw new Error(`wait_for timed out after ${timeoutMs} ms waiting for ${label}`)
      await new Promise((resolve) => setTimeout(resolve, WAIT_POLL_MS))
    }
  }

  private async dispatch(action: RemoteAction, signal?: AbortSignal): Promise<string> {
    const checkAbort = () => {
      if (signal?.aborted) throw new Error('Browser action cancelled; inspect the page before continuing')
    }
    switch (action.action) {
      case 'navigate': {
        const settled = this.waitForNavigation()
        const result = (await this.page('Page.navigate', { url: action.url })) as { errorText?: string }
        if (result.errorText) {
          settled.cancel()
          throw new Error(`Navigation failed: ${result.errorText}`)
        }
        const outcome = await settled.promise
        if (outcome === 'error') throw new Error(`Navigation failed: ${action.url} returned a browser error page (network or DNS problem); the tab stays shared`)
        if (outcome === 'timeout') throw new Error(`Navigation to ${action.url} did not finish within ${NAVIGATION_TIMEOUT_MS / 1000} s; inspect the page before retrying`)
        return `Navigated to ${this.url}; inspect the page before the next action.`
      }
      case 'click':
        if (action.ref) {
          const backendNodeId = this.nodeFor(action.ref, action.snapshotId)
          await this.page('DOM.scrollIntoViewIfNeeded', { backendNodeId })
          const center = await this.centerOf(backendNodeId)
          checkAbort()
          await this.clickAt(center.x, center.y, signal)
        } else if (action.x !== undefined && action.y !== undefined) await this.clickAt(action.x, action.y, signal)
        else throw new Error('click needs ref or x and y')
        break
      case 'fill': {
        const backendNodeId = this.nodeFor(action.ref, action.snapshotId)
        await this.page('DOM.scrollIntoViewIfNeeded', { backendNodeId })
        await this.page('DOM.focus', { backendNodeId })
        await this.callOnNode(
          backendNodeId,
          `function () {
            if (typeof this.select === 'function') { this.select(); return }
            const selection = window.getSelection()
            if (selection) { const range = document.createRange(); range.selectNodeContents(this); selection.removeAllRanges(); selection.addRange(range) }
          }`,
        )
        checkAbort()
        await this.page('Input.insertText', { text: action.text })
        break
      }
      case 'type':
        await this.page('Input.insertText', { text: action.text })
        break
      case 'press':
      case 'key':
        await this.pressKey(action.key)
        break
      case 'scroll':
        if (action.ref) {
          await this.page('DOM.scrollIntoViewIfNeeded', { backendNodeId: this.nodeFor(action.ref, action.snapshotId) })
        } else {
          await this.page('Input.dispatchMouseEvent', {
            type: 'mouseWheel',
            x: 100,
            y: 100,
            deltaX: 0,
            deltaY: action.deltaY ?? 0,
          })
        }
        break
      case 'select': {
        const backendNodeId = this.nodeFor(action.ref, action.snapshotId)
        const picked = await this.callOnNode(
          backendNodeId,
          `function (values) {
            if (this.tagName !== 'SELECT') throw new Error('Not a select element')
            const wanted = new Set(values)
            let count = 0
            for (const option of this.options) { option.selected = wanted.has(option.value) || wanted.has(option.textContent.trim()); if (option.selected) count += 1 }
            this.dispatchEvent(new Event('input', { bubbles: true }))
            this.dispatchEvent(new Event('change', { bubbles: true }))
            return count
          }`,
          [action.values],
        )
        if (!picked) throw new Error('No option matched the requested values')
        break
      }
      case 'hover': {
        const backendNodeId = this.nodeFor(action.ref, action.snapshotId)
        await this.page('DOM.scrollIntoViewIfNeeded', { backendNodeId })
        const center = await this.centerOf(backendNodeId)
        await this.mouse('mouseMoved', center.x, center.y)
        break
      }
      case 'wait_for':
        if (action.ref) {
          const backendNodeId = this.nodeFor(action.ref, action.snapshotId)
          await this.waitUntil(
            () => this.centerOf(backendNodeId).then(() => true, () => false),
            action.timeoutMs,
            `element ${action.ref}`,
          )
        } else if (action.text) {
          const needle = action.text
          await this.waitUntil(
            async () => {
              const evaluated = (await this.page('Runtime.evaluate', {
                expression: `Boolean(document.body && document.body.innerText.includes(${JSON.stringify(needle)}))`,
                returnByValue: true,
              })) as { result?: { value?: unknown } }
              return evaluated.result?.value === true
            },
            action.timeoutMs,
            `text ${JSON.stringify(needle)}`,
          )
        } else throw new Error('wait_for needs text or ref')
        break
    }
    checkAbort()
    return `${action.action} completed; inspect the page before the next action.`
  }

  private formatConsole(level?: string): string {
    const entries = level ? this.consoleLog.filter((entry) => entry.level === level) : this.consoleLog
    if (entries.length === 0) return 'No console messages captured since the tab was attached.'
    return entries.map((entry) => `[${entry.level}] ${entry.text}`).join('\n')
  }

  private formatNetwork(filter?: string): string {
    const entries = filter ? this.networkLog.filter((entry) => entry.url.includes(filter)) : this.networkLog
    if (entries.length === 0) return 'No network responses captured since the tab was attached.'
    return entries
      .map((entry) => `${entry.method} ${entry.url} ${entry.status} ${entry.resourceType}`)
      .join('\n')
  }

  async close(): Promise<void> {
    if (this.closed) return
    this.frameListeners.clear()
    await this.syncScreencast()
    await this.page('Target.detachFromTarget', { sessionId: this.sessionId }).catch(() => {})
    this.end('Disconnected')
  }
}
