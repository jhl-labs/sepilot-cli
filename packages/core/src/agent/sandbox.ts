export interface RunOptions {
  cwd?: string
  env?: Record<string, string>
  timeoutMs?: number
}

export interface RunResult {
  stdout: string
  stderr: string
  exitCode: number
  durationMs: number
}

export interface ISandbox {
  run(command: string, options?: RunOptions): Promise<RunResult>
  writeFile(path: string, content: string): Promise<void>
  readFile(path: string): Promise<string>
  cleanup(): Promise<void>
}
