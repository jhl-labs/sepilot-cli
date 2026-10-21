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
  inheritEnv?: boolean
  detached?: boolean
  stdio?: 'ignore' | 'inherit'
}

export interface DaemonProcess {
  pid: number
  isRunning?(): boolean
  stop(): void
}
