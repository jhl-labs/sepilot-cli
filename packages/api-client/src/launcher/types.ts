export interface DaemonInvocation {
  command: string
  args: string[]
  kind: 'binary' | 'module'
}

export interface DaemonResolveOptions {
  moduleSearchRoots?: string[]
  resourceSearchRoots?: string[]
  scanPnpmLayout?: boolean
  env?: NodeJS.ProcessEnv
  execPath?: string
  homeDir?: string
}

export interface DaemonLaunchOptions {
  env?: Record<string, string>
  /**
   * Merge `env` over the parent process environment by default. Set false
   * when the caller has already built a complete least-privilege child
   * environment and ambient variables must not be reintroduced here.
   */
  inheritEnv?: boolean
  detached?: boolean
  stdio?: 'ignore' | 'inherit'
}

export interface DaemonProcess {
  pid: number
  /** Lifecycle of the child we launched, without probing a potentially reused PID. */
  isRunning?(): boolean
  stop(): void
}
