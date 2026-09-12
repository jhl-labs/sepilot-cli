import { execFile } from 'node:child_process'
import { mkdir, readFile, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { randomUUID } from 'node:crypto'
import type { ToolDefinitionRuntime, ToolResult, ToolResultImage } from './registry.js'

const execFileAsync = promisify(execFile)
const DEFAULT_TIMEOUT_MS = 15_000
const MAX_OBSERVE_IMAGE_BYTES = 5 * 1024 * 1024

type ComputerButton = 'left' | 'right' | 'middle'
type ComputerLaunchApp = 'notepad' | 'calculator' | 'paint' | 'explorer'
type ComputerPointParseMode = 'required' | 'optional'

export type ComputerOpenUrlParseResult =
  | { ok: true; url: string }
  | { ok: false; message: string }

type ComputerPointParseResult =
  | { ok: true; hasPoint: true; x: number; y: number }
  | { ok: true; hasPoint: false; x: 0; y: 0 }
  | { ok: false; message: string }
type ComputerDragParseResult =
  | { ok: true; fromX: number; fromY: number; toX: number; toY: number }
  | { ok: false; message: string }

const LAUNCH_APPS: Record<ComputerLaunchApp, string> = {
  notepad: 'notepad.exe',
  calculator: 'calc.exe',
  paint: 'mspaint.exe',
  explorer: 'explorer.exe',
}

function isWindows(): boolean {
  return process.platform === 'win32'
}

function windowsOnly(start: number): ToolResult {
  return {
    output: 'computer.* tools are available only on Windows desktop sessions.',
    status: 'error',
    code: 'WINDOWS_ONLY_PERMANENT',
    durationMs: Date.now() - start,
  }
}

// Handing a URL to the desktop's default browser is a plain OS primitive
// everywhere, unlike the rest of computer.* which drives the Windows-only
// automation backend (mouse, keyboard, window handles). Keep the platform
// question per-tool so routing can advertise this capability where it really
// works instead of tying it to the GUI-automation backend.
const OPEN_URL_COMMANDS = new Map<NodeJS.Platform, string>([
  ['darwin', 'open'],
  ['linux', 'xdg-open'],
])

export function isComputerOpenUrlSupported(
  platform: NodeJS.Platform = process.platform,
): boolean {
  return platform === 'win32' || OPEN_URL_COMMANDS.has(platform)
}

function openUrlUnsupported(start: number): ToolResult {
  return {
    output: `computer.open_url is not available on ${process.platform}.`,
    status: 'error',
    code: 'UNSUPPORTED_PLATFORM_PERMANENT',
    durationMs: Date.now() - start,
  }
}

function waitWithSignal(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms)
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer)
        reject(new Error('computer.open_url aborted'))
      },
      { once: true },
    )
  })
}

function parseNumber(
  value: unknown,
  fallback: number,
  options: { min?: number; max?: number } = {},
): number {
  const next = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(next)) return fallback
  const min = options.min ?? Number.NEGATIVE_INFINITY
  const max = options.max ?? Number.POSITIVE_INFINITY
  return Math.max(min, Math.min(max, Math.trunc(next)))
}

function parseFiniteInteger(value: unknown): number | null {
  const next = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(next) ? Math.trunc(next) : null
}

export function parseComputerPointInput(
  input: Record<string, unknown>,
  mode: ComputerPointParseMode,
): ComputerPointParseResult {
  const hasX = input.x !== undefined
  const hasY = input.y !== undefined

  if (mode === 'optional' && !hasX && !hasY) {
    return { ok: true, hasPoint: false, x: 0, y: 0 }
  }
  if (!hasX || !hasY) {
    return { ok: false, message: 'x and y coordinates must be provided together.' }
  }

  const x = parseFiniteInteger(input.x)
  const y = parseFiniteInteger(input.y)
  if (x === null || y === null) {
    return { ok: false, message: 'x and y coordinates must be finite numbers.' }
  }

  return { ok: true, hasPoint: true, x, y }
}

function parseComputerDragInput(input: Record<string, unknown>): ComputerDragParseResult {
  const fromX = parseFiniteInteger(input.fromX)
  const fromY = parseFiniteInteger(input.fromY)
  const toX = parseFiniteInteger(input.toX)
  const toY = parseFiniteInteger(input.toY)
  if (fromX === null || fromY === null || toX === null || toY === null) {
    return {
      ok: false,
      message: 'fromX, fromY, toX, and toY coordinates must be finite numbers.',
    }
  }
  return { ok: true, fromX, fromY, toX, toY }
}

function parseString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

function parseButton(value: unknown): ComputerButton {
  return value === 'right' || value === 'middle' ? value : 'left'
}

function parseLaunchApp(value: unknown): ComputerLaunchApp | null {
  return typeof value === 'string' && value in LAUNCH_APPS ? (value as ComputerLaunchApp) : null
}

/**
 * Validate the narrow URL surface that may be handed to the user's default
 * browser. Keeping this separate from headless browser navigation makes the
 * user-visible side effect explicit and prevents file/custom-protocol launch.
 */
export function parseComputerOpenUrl(value: unknown): ComputerOpenUrlParseResult {
  if (typeof value !== 'string' || !value.trim()) {
    return { ok: false, message: 'url must be a non-empty HTTP(S) URL.' }
  }
  const candidate = value.trim()
  if (candidate.length > 4096) {
    return { ok: false, message: 'url must be 4096 characters or fewer.' }
  }

  try {
    const parsed = new URL(candidate)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return { ok: false, message: 'Only http and https URLs can be opened.' }
    }
    if (parsed.username || parsed.password) {
      return { ok: false, message: 'URLs containing credentials cannot be opened.' }
    }
    return { ok: true, url: parsed.toString() }
  } catch {
    return { ok: false, message: 'url must be a valid absolute HTTP(S) URL.' }
  }
}

function psJson(value: unknown): string {
  return JSON.stringify(value).replace(/'/g, "''")
}

function encodePowerShell(script: string): string {
  return Buffer.from(script, 'utf16le').toString('base64')
}

function outputText(value: unknown): string {
  if (Buffer.isBuffer(value)) return value.toString('utf8')
  return typeof value === 'string' ? value : value == null ? '' : String(value)
}

function tail(value: string, max = 2000): string {
  const normalized = value.replace(/\r\n/g, '\n').trim()
  if (normalized.length <= max) return normalized
  return `...${normalized.slice(normalized.length - max)}`
}

function powerShellRawOutputDetails(stdout: unknown, stderr: unknown): string {
  const stdoutText = tail(outputText(stdout))
  const stderrText = tail(outputText(stderr))
  const parts: string[] = []
  if (stdoutText) parts.push(`stdout tail:\n${stdoutText}`)
  if (stderrText) parts.push(`stderr tail:\n${stderrText}`)
  return parts.length > 0 ? `\nPowerShell raw output:\n${parts.join('\n')}` : ''
}

function errorRawOutput(error: unknown, key: 'stdout' | 'stderr'): unknown {
  return typeof error === 'object' && error !== null && key in error
    ? (error as Record<typeof key, unknown>)[key]
    : undefined
}

async function runPowerShellJson<T>(
  script: string,
  options?: { signal?: AbortSignal; timeoutMs?: number },
): Promise<T> {
  const encoded = encodePowerShell(script)
  let stdout: unknown
  let stderr: unknown
  try {
    const result = await execFileAsync(
      'powershell.exe',
      [
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-Sta',
        '-EncodedCommand',
        encoded,
      ],
      {
        windowsHide: true,
        timeout: options?.timeoutMs ?? DEFAULT_TIMEOUT_MS,
        maxBuffer: 10 * 1024 * 1024,
        signal: options?.signal,
      },
    )
    stdout = result.stdout
    stderr = result.stderr
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(
      `${message}${powerShellRawOutputDetails(
        errorRawOutput(error, 'stdout'),
        errorRawOutput(error, 'stderr'),
      )}`,
    )
  }

  try {
    return JSON.parse(outputText(stdout).trim()) as T
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(
      `PowerShell returned non-JSON output: ${message}${powerShellRawOutputDetails(stdout, stderr)}`,
    )
  }
}

function successOutput(value: unknown, start: number): ToolResult {
  return {
    output: JSON.stringify(value, null, 2),
    status: 'success',
    durationMs: Date.now() - start,
  }
}

async function observationImage(value: unknown): Promise<ToolResultImage[] | undefined> {
  if (typeof value !== 'object' || value === null) return undefined
  const record = value as { path?: unknown; mimeType?: unknown; mediaType?: unknown }
  if (typeof record.path !== 'string' || !record.path) return undefined
  const mediaType =
    typeof record.mimeType === 'string'
      ? record.mimeType
      : typeof record.mediaType === 'string'
        ? record.mediaType
        : 'image/png'
  if (!mediaType.startsWith('image/')) return undefined
  try {
    const info = await stat(record.path)
    if (info.size <= 0 || info.size > MAX_OBSERVE_IMAGE_BYTES) return undefined
    const data = await readFile(record.path)
    return [{ mediaType, data: data.toString('base64') }]
  } catch {
    return undefined
  }
}

function errorOutput(error: unknown, start: number): ToolResult {
  return {
    output: error instanceof Error ? error.message : String(error),
    status: 'error',
    durationMs: Date.now() - start,
  }
}

function normalizeKeys(input: unknown): string[] {
  if (Array.isArray(input)) {
    return input.map((key) => String(key).trim()).filter(Boolean)
  }
  if (typeof input === 'string') {
    return input
      .split(/[+\s]+/)
      .map((key) => key.trim())
      .filter(Boolean)
  }
  return []
}

function sendKeysToken(key: string): string {
  const normalized = key.trim().toUpperCase()
  const special: Record<string, string> = {
    ENTER: '{ENTER}',
    RETURN: '{ENTER}',
    ESC: '{ESC}',
    ESCAPE: '{ESC}',
    TAB: '{TAB}',
    SPACE: ' ',
    BACKSPACE: '{BACKSPACE}',
    BKSP: '{BACKSPACE}',
    DELETE: '{DELETE}',
    DEL: '{DELETE}',
    INSERT: '{INSERT}',
    HOME: '{HOME}',
    END: '{END}',
    PAGEUP: '{PGUP}',
    PAGEDOWN: '{PGDN}',
    UP: '{UP}',
    DOWN: '{DOWN}',
    LEFT: '{LEFT}',
    RIGHT: '{RIGHT}',
  }
  if (/^F(?:[1-9]|1[0-9]|2[0-4])$/.test(normalized)) {
    return `{${normalized}}`
  }
  if (special[normalized]) {
    return special[normalized]
  }
  if (key.length === 1) {
    return key.replace(/[+^%~()[\]{}]/g, '{$&}')
  }
  return `{${normalized}}`
}

function toSendKeys(keys: string[]): string {
  const modifiers = new Set(['CTRL', 'CONTROL', 'ALT', 'SHIFT'])
  const prefix = keys
    .slice(0, -1)
    .map((key) => key.trim().toUpperCase())
    .filter((key) => modifiers.has(key))
    .map((key) => {
      if (key === 'ALT') return '%'
      if (key === 'SHIFT') return '+'
      return '^'
    })
    .join('')
  const target = keys.findLast((key) => !modifiers.has(key.trim().toUpperCase()))
  return target ? `${prefix}${sendKeysToken(target)}` : ''
}

export function buildComputerListWindowsPowerShell(): string {
  return [
    'Add-Type @"',
    'using System;',
    'using System.Runtime.InteropServices;',
    'public static class Win32WindowList {',
    '  public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }',
    '  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();',
    '  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);',
    '}',
    '"@',
    '$foregroundHwnd = [Win32WindowList]::GetForegroundWindow()',
    '$windows = Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle } | ForEach-Object {',
    '  $bounds = $null',
    "  $rect = New-Object 'Win32WindowList+RECT'",
    '  $hwnd = [IntPtr]$_.MainWindowHandle',
    '  if ([Win32WindowList]::GetWindowRect($hwnd, [ref]$rect)) {',
    '    $bounds = [PSCustomObject]@{',
    '      x = [int]$rect.Left',
    '      y = [int]$rect.Top',
    '      width = [int]($rect.Right - $rect.Left)',
    '      height = [int]($rect.Bottom - $rect.Top)',
    '    }',
    '  }',
    '  [PSCustomObject]@{',
    '    pid = [int]$_.Id',
    '    processName = $_.ProcessName',
    '    title = $_.MainWindowTitle',
    '    hwnd = [Int64]$_.MainWindowHandle',
    '    bounds = $bounds',
    '    foreground = [bool]([Int64]$_.MainWindowHandle -eq $foregroundHwnd.ToInt64())',
    '  }',
    '}',
    '@($windows) | ConvertTo-Json -Compress -Depth 4',
  ].join('\n')
}

function buildComputerFocusPowerShellType(): string[] {
  return [
    'Add-Type @"',
    'using System;',
    'using System.Runtime.InteropServices;',
    'public static class Win32Focus {',
    '  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);',
    '  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);',
    '  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();',
    '  [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr hWnd);',
    '  [DllImport("user32.dll")] public static extern IntPtr SetActiveWindow(IntPtr hWnd);',
    '  [DllImport("user32.dll")] public static extern IntPtr SetFocus(IntPtr hWnd);',
    '  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);',
    '  [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();',
    '  [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint idAttach, uint idAttachTo, bool fAttach);',
    '}',
    '"@',
  ]
}

function buildComputerFocusPowerShellFunction(): string[] {
  return [
    'function Invoke-SepilotForeground {',
    '  param([IntPtr]$TargetHwnd)',
    '  if ($TargetHwnd -eq [IntPtr]::Zero) {',
    '    return [PSCustomObject]@{ focused = $false; setForegroundReturned = $false; foregroundHwnd = 0 }',
    '  }',
    '  [Win32Focus]::ShowWindow($TargetHwnd, 9) | Out-Null',
    '  Start-Sleep -Milliseconds 100',
    '  $beforeForeground = [Win32Focus]::GetForegroundWindow()',
    '  [uint32]$targetPid = 0',
    '  $targetThread = [Win32Focus]::GetWindowThreadProcessId($TargetHwnd, [ref]$targetPid)',
    '  [uint32]$foregroundPid = 0',
    '  $foregroundThread = 0',
    '  if ($beforeForeground -ne [IntPtr]::Zero) {',
    '    $foregroundThread = [Win32Focus]::GetWindowThreadProcessId($beforeForeground, [ref]$foregroundPid)',
    '  }',
    '  $currentThread = [Win32Focus]::GetCurrentThreadId()',
    '  $attachedCurrent = $false',
    '  $attachedForeground = $false',
    '  $setForegroundReturned = $false',
    '  try {',
    '    if ($targetThread -ne 0 -and $targetThread -ne $currentThread) {',
    '      $attachedCurrent = [Win32Focus]::AttachThreadInput($currentThread, $targetThread, $true)',
    '    }',
    '    if ($targetThread -ne 0 -and $foregroundThread -ne 0 -and $foregroundThread -ne $targetThread) {',
    '      $attachedForeground = [Win32Focus]::AttachThreadInput($foregroundThread, $targetThread, $true)',
    '    }',
    '    [Win32Focus]::BringWindowToTop($TargetHwnd) | Out-Null',
    '    [Win32Focus]::SetActiveWindow($TargetHwnd) | Out-Null',
    '    [Win32Focus]::SetFocus($TargetHwnd) | Out-Null',
    '    $setForegroundReturned = [Win32Focus]::SetForegroundWindow($TargetHwnd)',
    '  } finally {',
    '    if ($attachedForeground) { [Win32Focus]::AttachThreadInput($foregroundThread, $targetThread, $false) | Out-Null }',
    '    if ($attachedCurrent) { [Win32Focus]::AttachThreadInput($currentThread, $targetThread, $false) | Out-Null }',
    '  }',
    '  Start-Sleep -Milliseconds 150',
    '  $afterForeground = [Win32Focus]::GetForegroundWindow()',
    '  [PSCustomObject]@{',
    '    focused = [bool]($afterForeground -eq $TargetHwnd)',
    '    setForegroundReturned = [bool]$setForegroundReturned',
    '    foregroundHwnd = $afterForeground.ToInt64()',
    '  }',
    '}',
  ]
}

export function createComputerListWindowsTool(): ToolDefinitionRuntime {
  return {
    name: 'computer.list_windows',
    description:
      'List visible top-level Windows application windows. Use before GUI actions to choose a target process/window.',
    resumeSafety: 'replay-safe',
    inputSchema: {
      type: 'object',
      properties: {},
    },
    async execute(_input, context): Promise<ToolResult> {
      const start = Date.now()
      if (!isWindows()) return windowsOnly(start)
      try {
        const result = await runPowerShellJson<unknown>(buildComputerListWindowsPowerShell(), {
          signal: context?.signal,
        })
        return successOutput(result, start)
      } catch (error) {
        return errorOutput(error, start)
      }
    },
  }
}

export function createComputerFocusWindowTool(): ToolDefinitionRuntime {
  return {
    name: 'computer.focus_window',
    description:
      'Focus a visible Windows application window by hwnd, pid, or title substring. Call this before click/type actions so input goes to the intended app.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        hwnd: { type: 'number', description: 'Window handle from computer.list_windows.' },
        pid: { type: 'number', description: 'Process id from computer.list_windows.' },
        title: { type: 'string', description: 'Case-insensitive title substring.' },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      if (!isWindows()) return windowsOnly(start)
      const params = {
        hwnd: input.hwnd,
        pid: input.pid,
        title: parseString(input.title).trim(),
      }
      try {
        const result = await runPowerShellJson<unknown>(
          [
            `$params = ConvertFrom-Json -InputObject '${psJson(params)}'`,
            ...buildComputerFocusPowerShellType(),
            ...buildComputerFocusPowerShellFunction(),
            '$hwnd = [IntPtr]::Zero',
            '$targetProcess = $null',
            'if ($params.hwnd) {',
            '  $hwnd = [IntPtr]([Int64]$params.hwnd)',
            '  $targetProcess = Get-Process | Where-Object { [Int64]$_.MainWindowHandle -eq [Int64]$params.hwnd } | Select-Object -First 1',
            '}',
            'elseif ($params.pid) {',
            '  $targetProcess = Get-Process -Id ([int]$params.pid) -ErrorAction Stop',
            '  $hwnd = [IntPtr]$targetProcess.MainWindowHandle',
            '} elseif ($params.title) {',
            '  $targetProcess = Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -like ("*" + $params.title + "*") } | Select-Object -First 1',
            '  if ($targetProcess) { $hwnd = [IntPtr]$targetProcess.MainWindowHandle }',
            '}',
            'if ($hwnd -eq [IntPtr]::Zero) { throw "No matching window found" }',
            'if (-not $targetProcess) {',
            '  $targetProcess = Get-Process | Where-Object { [Int64]$_.MainWindowHandle -eq $hwnd.ToInt64() } | Select-Object -First 1',
            '}',
            '$focusResult = Invoke-SepilotForeground -TargetHwnd $hwnd',
            '$pidValue = $null',
            '$processName = $null',
            '$title = $null',
            'if ($targetProcess) {',
            '  $pidValue = [int]$targetProcess.Id',
            '  $processName = $targetProcess.ProcessName',
            '  $title = $targetProcess.MainWindowTitle',
            '}',
            '[PSCustomObject]@{ hwnd = $hwnd.ToInt64(); pid = $pidValue; processName = $processName; title = $title; focused = [bool]$focusResult.focused; focusVerified = [bool]$focusResult.focused; foregroundHwnd = [Int64]$focusResult.foregroundHwnd; setForegroundReturned = [bool]$focusResult.setForegroundReturned } | ConvertTo-Json -Compress',
          ].join('\n'),
          { signal: context?.signal },
        )
        return successOutput(result, start)
      } catch (error) {
        return errorOutput(error, start)
      }
    },
  }
}

export function createComputerLaunchAppTool(): ToolDefinitionRuntime {
  return {
    name: 'computer.launch_app',
    description:
      'Launch and focus a supported built-in Windows GUI app for a simple computer-use workflow. MVP supports notepad, calculator, paint, and explorer.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        app: {
          type: 'string',
          enum: Object.keys(LAUNCH_APPS),
          description: 'Built-in Windows app to launch.',
        },
        waitMs: {
          type: 'number',
          description:
            'Milliseconds to wait for the app main window before returning. Defaults to 1500 and caps at 10000.',
        },
        focus: {
          type: 'boolean',
          description: 'Bring the launched app window to the foreground. Defaults to true.',
        },
        path: {
          type: 'string',
          description: 'Optional folder path for explorer. Ignored by other apps.',
        },
      },
      required: ['app'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      if (!isWindows()) return windowsOnly(start)

      const app = parseLaunchApp(input.app)
      if (!app) {
        return {
          output: `Unsupported app '${String(input.app)}'. Supported apps: ${Object.keys(LAUNCH_APPS).join(', ')}`,
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          durationMs: Date.now() - start,
        }
      }

      const params = {
        app,
        executable: LAUNCH_APPS[app],
        waitMs: parseNumber(input.waitMs, 1500, { min: 0, max: 10_000 }),
        focus: input.focus !== false,
        path: app === 'explorer' ? parseString(input.path).trim() : '',
      }
      try {
        const result = await runPowerShellJson<unknown>(
          [
            `$params = ConvertFrom-Json -InputObject '${psJson(params)}'`,
            ...buildComputerFocusPowerShellType(),
            ...buildComputerFocusPowerShellFunction(),
            '$resolvedPath = $null',
            'if ($params.app -eq "explorer" -and $params.path) {',
            '  $resolvedPath = [System.IO.Path]::GetFullPath([string]$params.path)',
            '  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Container)) { throw "Explorer path is not a folder: $resolvedPath" }',
            '}',
            '$process = $null',
            'if ($params.app -eq "explorer" -and $resolvedPath) {',
            '  $process = Start-Process -FilePath ([string]$params.executable) -ArgumentList @($resolvedPath) -PassThru',
            '} else {',
            '  $process = Start-Process -FilePath ([string]$params.executable) -PassThru',
            '}',
            '$deadline = (Get-Date).AddMilliseconds([int]$params.waitMs)',
            'try { $process.WaitForInputIdle([int]$params.waitMs) | Out-Null } catch { }',
            'try { $process.Refresh() } catch { }',
            '$hwnd = [IntPtr]$process.MainWindowHandle',
            '$targetProcess = $process',
            'if ($params.app -eq "explorer" -and $resolvedPath) {',
            '  $shellApp = New-Object -ComObject Shell.Application',
            '  while ($hwnd -eq [IntPtr]::Zero -and (Get-Date) -lt $deadline) {',
            '    Start-Sleep -Milliseconds 200',
            '    foreach ($candidate in @($shellApp.Windows())) {',
            '      try {',
            '        $fullName = [string]$candidate.FullName',
            '        $folderPath = [string]$candidate.Document.Folder.Self.Path',
            '        if ($fullName -like "*explorer.exe" -and $folderPath -eq $resolvedPath) {',
            '          $hwnd = [IntPtr]([Int64]$candidate.HWND)',
            '          break',
            '        }',
            '      } catch { }',
            '    }',
            '  }',
            '  if ($hwnd -ne [IntPtr]::Zero) {',
            '    $targetProcess = Get-Process | Where-Object { [Int64]$_.MainWindowHandle -eq $hwnd.ToInt64() } | Select-Object -First 1',
            '  }',
            '} else {',
            '  while ($hwnd -eq [IntPtr]::Zero -and (Get-Date) -lt $deadline) {',
            '    Start-Sleep -Milliseconds 100',
            '    try { $process.Refresh(); $hwnd = [IntPtr]$process.MainWindowHandle } catch { break }',
            '  }',
            '}',
            '$focusResult = [PSCustomObject]@{ focused = $false; setForegroundReturned = $false; foregroundHwnd = 0 }',
            'if ($params.focus -and $hwnd -ne [IntPtr]::Zero) {',
            '  $focusResult = Invoke-SepilotForeground -TargetHwnd $hwnd',
            '}',
            '$pidValue = $null',
            '$processName = $null',
            '$title = $null',
            'if ($targetProcess) {',
            '  $pidValue = [int]$targetProcess.Id',
            '  $processName = $targetProcess.ProcessName',
            '  $title = $targetProcess.MainWindowTitle',
            '}',
            '[PSCustomObject]@{',
            '  app = $params.app',
            '  executable = $params.executable',
            '  path = $resolvedPath',
            '  pid = $pidValue',
            '  processName = $processName',
            '  hwnd = $hwnd.ToInt64()',
            '  title = $title',
            '  focused = [bool]$focusResult.focused',
            '  focusVerified = [bool]$focusResult.focused',
            '  foregroundHwnd = [Int64]$focusResult.foregroundHwnd',
            '  setForegroundReturned = [bool]$focusResult.setForegroundReturned',
            '  waitMs = [int]$params.waitMs',
            '  started = $true',
            '} | ConvertTo-Json -Compress',
          ].join('\n'),
          { signal: context?.signal, timeoutMs: 15_000 },
        )
        return successOutput(result, start)
      } catch (error) {
        return errorOutput(error, start)
      }
    },
  }
}

export function createComputerOpenUrlTool(): ToolDefinitionRuntime {
  return {
    name: 'computer.open_url',
    description:
      'Open one known http/https URL visibly in the user\'s default system browser. This is a user-facing desktop action, unlike browser.navigate which uses headless Chromium for inspection. Use it only when the user explicitly asks to open, show, or display a site/page/map in their browser. For research or search-only requests, use web.search/browser.navigate instead; when the destination is not yet known, find the URL first and then ask to open the selected result. A successful result confirms that the operating system accepted the open request, not that the page content or downstream service was verified.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        url: {
          type: 'string',
          description: 'Absolute http/https URL to open in the default browser.',
        },
        waitMs: {
          type: 'number',
          description:
            'Milliseconds to wait after handing the URL to Windows. Defaults to 1500 and caps at 10000.',
        },
      },
      required: ['url'],
    },
    validateInput: (input) => {
      const parsed = parseComputerOpenUrl(input.url)
      if (parsed.ok) return null
      return {
        output: parsed.message,
        status: 'error',
        code: 'INVALID_INPUT_PERMANENT',
        durationMs: 0,
      }
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()

      const parsed = parseComputerOpenUrl(input.url)
      if (!parsed.ok) {
        return {
          output: parsed.message,
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          durationMs: Date.now() - start,
        }
      }

      const waitMs = parseNumber(input.waitMs, 1500, { min: 0, max: 10_000 })
      const params = { url: parsed.url, waitMs }
      try {
        if (isWindows()) {
          const result = await runPowerShellJson<unknown>(
            [
              `$params = ConvertFrom-Json -InputObject '${psJson(params)}'`,
              'Start-Process -FilePath ([string]$params.url)',
              'if ([int]$params.waitMs -gt 0) { Start-Sleep -Milliseconds ([int]$params.waitMs) }',
              '[PSCustomObject]@{',
              '  app = "default-browser"',
              '  url = [string]$params.url',
              '  opened = $true',
              '  waitMs = [int]$params.waitMs',
              '} | ConvertTo-Json -Compress',
            ].join('\n'),
            { signal: context?.signal, timeoutMs: 15_000 },
          )
          return successOutput(result, start)
        }

        const command = OPEN_URL_COMMANDS.get(process.platform)
        if (!command) return openUrlUnsupported(start)
        // The URL travels as its own argv entry and never through a shell.
        // parseComputerOpenUrl has already pinned it to a credential-free
        // http(s) URL, so it can be read neither as shell syntax nor as an
        // option flag by the opener.
        await execFileAsync(command, [parsed.url], {
          signal: context?.signal,
          timeout: DEFAULT_TIMEOUT_MS,
        })
        if (waitMs > 0) await waitWithSignal(waitMs, context?.signal)
        return successOutput(
          { app: 'default-browser', url: parsed.url, opened: true, waitMs },
          start,
        )
      } catch (error) {
        return errorOutput(error, start)
      }
    },
  }
}

export function createComputerObserveTool(): ToolDefinitionRuntime {
  return {
    name: 'computer.observe',
    description:
      'Capture the Windows virtual desktop as a PNG screenshot and return its file path plus screen coordinate metadata. Coordinates for computer.click are absolute screen pixels relative to originX/originY.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        note: { type: 'string', description: 'Optional short reason for this observation.' },
        scope: {
          type: 'string',
          enum: ['desktop', 'foreground'],
          description:
            'Capture the full virtual desktop or only the foreground window. Defaults to desktop.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      if (!isWindows()) return windowsOnly(start)
      const observationId = randomUUID()
      const dir = join(tmpdir(), 'sepilotd-computer-use', context?.sessionId ?? 'default')
      await mkdir(dir, { recursive: true })
      const path = join(dir, `${observationId}.png`)
      const params = {
        path,
        observationId,
        note: parseString(input.note).slice(0, 240),
        scope: input.scope === 'foreground' ? 'foreground' : 'desktop',
      }
      try {
        const result = await runPowerShellJson<unknown>(
          [
            `$params = ConvertFrom-Json -InputObject '${psJson(params)}'`,
            'Add-Type -AssemblyName System.Windows.Forms',
            'Add-Type -AssemblyName System.Drawing',
            'Add-Type @"',
            'using System;',
            'using System.Runtime.InteropServices;',
            'using System.Text;',
            'public static class Win32Observation {',
            '  public struct POINT { public int X; public int Y; }',
            '  public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }',
            '  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();',
            '  [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);',
            '  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);',
            '  [DllImport("user32.dll")] public static extern bool GetCursorPos(out POINT point);',
            '  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);',
            '}',
            '"@',
            '$bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen',
            '$foregroundWindow = $null',
            '$foregroundBounds = $null',
            '$foregroundHwnd = [Win32Observation]::GetForegroundWindow()',
            'if ($foregroundHwnd -ne [IntPtr]::Zero) {',
            '  $foregroundTitleBuilder = New-Object System.Text.StringBuilder 1024',
            '  [Win32Observation]::GetWindowText($foregroundHwnd, $foregroundTitleBuilder, $foregroundTitleBuilder.Capacity) | Out-Null',
            '  [uint32]$foregroundPid = 0',
            '  [Win32Observation]::GetWindowThreadProcessId($foregroundHwnd, [ref]$foregroundPid) | Out-Null',
            '  $foregroundProcessName = $null',
            '  try {',
            '    if ($foregroundPid -gt 0) { $foregroundProcessName = (Get-Process -Id ([int]$foregroundPid) -ErrorAction Stop).ProcessName }',
            '  } catch {',
            '    $foregroundProcessName = $null',
            '  }',
            "  $foregroundRect = New-Object 'Win32Observation+RECT'",
            '  if ([Win32Observation]::GetWindowRect($foregroundHwnd, [ref]$foregroundRect)) {',
            '    $foregroundBounds = [PSCustomObject]@{',
            '      x = [int]$foregroundRect.Left',
            '      y = [int]$foregroundRect.Top',
            '      width = [int]($foregroundRect.Right - $foregroundRect.Left)',
            '      height = [int]($foregroundRect.Bottom - $foregroundRect.Top)',
            '    }',
            '  }',
            '  $foregroundWindow = [PSCustomObject]@{',
            '    hwnd = $foregroundHwnd.ToInt64()',
            '    pid = [int]$foregroundPid',
            '    processName = $foregroundProcessName',
            '    title = $foregroundTitleBuilder.ToString()',
            '    bounds = $foregroundBounds',
            '  }',
            '}',
            '$cursor = $null',
            "$cursorPoint = New-Object 'Win32Observation+POINT'",
            'if ([Win32Observation]::GetCursorPos([ref]$cursorPoint)) {',
            '  $cursor = [PSCustomObject]@{ x = [int]$cursorPoint.X; y = [int]$cursorPoint.Y }',
            '}',
            '$captureBounds = [PSCustomObject]@{ X = [int]$bounds.X; Y = [int]$bounds.Y; Width = [int]$bounds.Width; Height = [int]$bounds.Height }',
            '$captureScope = "desktop"',
            'if ($params.scope -eq "foreground") {',
            '  if ($null -eq $foregroundBounds) { throw "No foreground window bounds available for foreground observation" }',
            '  $left = [Math]::Max([int]$foregroundBounds.x, [int]$bounds.X)',
            '  $top = [Math]::Max([int]$foregroundBounds.y, [int]$bounds.Y)',
            '  $right = [Math]::Min([int]($foregroundBounds.x + $foregroundBounds.width), [int]($bounds.X + $bounds.Width))',
            '  $bottom = [Math]::Min([int]($foregroundBounds.y + $foregroundBounds.height), [int]($bounds.Y + $bounds.Height))',
            '  if ($right -le $left -or $bottom -le $top) { throw "Foreground window bounds are outside the virtual screen" }',
            '  $captureBounds = [PSCustomObject]@{ X = $left; Y = $top; Width = [int]($right - $left); Height = [int]($bottom - $top) }',
            '  $captureScope = "foreground"',
            '}',
            '$captureSize = New-Object System.Drawing.Size -ArgumentList ([int]$captureBounds.Width), ([int]$captureBounds.Height)',
            '$bitmap = New-Object System.Drawing.Bitmap -ArgumentList ([int]$captureBounds.Width), ([int]$captureBounds.Height)',
            '$graphics = [System.Drawing.Graphics]::FromImage($bitmap)',
            'try {',
            '  $graphics.CopyFromScreen([int]$captureBounds.X, [int]$captureBounds.Y, 0, 0, $captureSize)',
            '  $bitmap.Save($params.path, [System.Drawing.Imaging.ImageFormat]::Png)',
            '} finally {',
            '  $graphics.Dispose()',
            '  $bitmap.Dispose()',
            '}',
            '[PSCustomObject]@{',
            '  observationId = $params.observationId',
            '  path = $params.path',
            '  mimeType = "image/png"',
            '  width = [int]$captureBounds.Width',
            '  height = [int]$captureBounds.Height',
            '  originX = [int]$captureBounds.X',
            '  originY = [int]$captureBounds.Y',
            '  capturedAt = (Get-Date).ToString("o")',
            '  note = $params.note',
            '  scope = $captureScope',
            '  coordinateSystem = "absolute-screen-pixels"',
            '  foregroundWindow = $foregroundWindow',
            '  cursor = $cursor',
            '} | ConvertTo-Json -Compress',
          ].join('\n'),
          { signal: context?.signal, timeoutMs: 20_000 },
        )
        const images = await observationImage(result)
        return {
          ...successOutput(result, start),
          ...(images ? { images } : {}),
        }
      } catch (error) {
        return errorOutput(error, start)
      }
    },
  }
}

export function createComputerClickTool(): ToolDefinitionRuntime {
  return {
    name: 'computer.click',
    description:
      'Move the Windows mouse pointer to absolute screen pixel coordinates and click. Use only after observing or focusing the intended target window.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        x: { type: 'number', description: 'Absolute screen X coordinate.' },
        y: { type: 'number', description: 'Absolute screen Y coordinate.' },
        button: {
          type: 'string',
          enum: ['left', 'right', 'middle'],
          description: 'Mouse button. Defaults to left.',
        },
        doubleClick: { type: 'boolean', description: 'Whether to double-click.' },
      },
      required: ['x', 'y'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      if (!isWindows()) return windowsOnly(start)
      const point = parseComputerPointInput(input, 'required')
      if (!point.ok) {
        return {
          output: point.message,
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          durationMs: Date.now() - start,
        }
      }
      const params = {
        x: point.x,
        y: point.y,
        button: parseButton(input.button),
        doubleClick: input.doubleClick === true,
      }
      try {
        const result = await runPowerShellJson<unknown>(
          [
            `$params = ConvertFrom-Json -InputObject '${psJson(params)}'`,
            'Add-Type -AssemblyName System.Windows.Forms',
            'Add-Type @"',
            'using System;',
            'using System.Runtime.InteropServices;',
            'public static class Win32Mouse {',
            '  [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);',
            '  [DllImport("user32.dll")] public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);',
            '}',
            '"@',
            '$bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen',
            '$maxX = $bounds.X + $bounds.Width - 1',
            '$maxY = $bounds.Y + $bounds.Height - 1',
            'if ([int]$params.x -lt $bounds.X -or [int]$params.x -gt $maxX -or [int]$params.y -lt $bounds.Y -or [int]$params.y -gt $maxY) {',
            '  throw "Coordinates ($($params.x), $($params.y)) are outside virtual screen bounds X=$($bounds.X)..$maxX Y=$($bounds.Y)..$maxY"',
            '}',
            '$down = 0x0002; $up = 0x0004',
            'if ($params.button -eq "right") { $down = 0x0008; $up = 0x0010 }',
            'elseif ($params.button -eq "middle") { $down = 0x0020; $up = 0x0040 }',
            '[Win32Mouse]::SetCursorPos([int]$params.x, [int]$params.y) | Out-Null',
            'Start-Sleep -Milliseconds 80',
            '$clickCount = if ($params.doubleClick) { 2 } else { 1 }',
            'for ($i = 0; $i -lt $clickCount; $i++) {',
            '  [Win32Mouse]::mouse_event([uint32]$down, 0, 0, 0, [UIntPtr]::Zero)',
            '  Start-Sleep -Milliseconds 40',
            '  [Win32Mouse]::mouse_event([uint32]$up, 0, 0, 0, [UIntPtr]::Zero)',
            '  Start-Sleep -Milliseconds 80',
            '}',
            '[PSCustomObject]@{ x = [int]$params.x; y = [int]$params.y; button = $params.button; doubleClick = [bool]$params.doubleClick } | ConvertTo-Json -Compress',
          ].join('\n'),
          { signal: context?.signal },
        )
        return successOutput(result, start)
      } catch (error) {
        return errorOutput(error, start)
      }
    },
  }
}

export function createComputerMoveMouseTool(): ToolDefinitionRuntime {
  return {
    name: 'computer.move_mouse',
    description:
      'Move the Windows mouse pointer to absolute screen pixel coordinates without clicking. Use for hover states, menus, and tooltips after observing the target.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        x: { type: 'number', description: 'Absolute screen X coordinate.' },
        y: { type: 'number', description: 'Absolute screen Y coordinate.' },
        durationMs: {
          type: 'number',
          description: 'Mouse move duration in milliseconds. Defaults to 0 and caps at 5000.',
        },
        steps: {
          type: 'number',
          description:
            'Number of intermediate mouse positions for smooth movement. Defaults to 1 and caps at 100.',
        },
      },
      required: ['x', 'y'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      if (!isWindows()) return windowsOnly(start)
      const point = parseComputerPointInput(input, 'required')
      if (!point.ok) {
        return {
          output: point.message,
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          durationMs: Date.now() - start,
        }
      }
      const params = {
        x: point.x,
        y: point.y,
        durationMs: parseNumber(input.durationMs, 0, { min: 0, max: 5000 }),
        steps: parseNumber(input.steps, 1, { min: 1, max: 100 }),
      }
      try {
        const result = await runPowerShellJson<unknown>(
          [
            `$params = ConvertFrom-Json -InputObject '${psJson(params)}'`,
            'Add-Type -AssemblyName System.Windows.Forms',
            'Add-Type @"',
            'using System;',
            'using System.Runtime.InteropServices;',
            'public static class Win32MoveMouse {',
            '  public struct POINT { public int X; public int Y; }',
            '  [DllImport("user32.dll")] public static extern bool GetCursorPos(out POINT point);',
            '  [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);',
            '}',
            '"@',
            '$bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen',
            '$maxX = $bounds.X + $bounds.Width - 1',
            '$maxY = $bounds.Y + $bounds.Height - 1',
            'if ([int]$params.x -lt $bounds.X -or [int]$params.x -gt $maxX -or [int]$params.y -lt $bounds.Y -or [int]$params.y -gt $maxY) {',
            '  throw "Move coordinates ($($params.x), $($params.y)) are outside virtual screen bounds X=$($bounds.X)..$maxX Y=$($bounds.Y)..$maxY"',
            '}',
            "$startPoint = New-Object 'Win32MoveMouse+POINT'",
            '$hasStart = [Win32MoveMouse]::GetCursorPos([ref]$startPoint)',
            '$startX = if ($hasStart) { [int]$startPoint.X } else { [int]$params.x }',
            '$startY = if ($hasStart) { [int]$startPoint.Y } else { [int]$params.y }',
            '$steps = [int]$params.steps',
            '$sleepMs = if ($steps -gt 0) { [int][Math]::Floor([int]$params.durationMs / $steps) } else { 0 }',
            'for ($i = 1; $i -le $steps; $i++) {',
            '  $progress = [double]$i / [double]$steps',
            '  $x = [int][Math]::Round($startX + (([int]$params.x - $startX) * $progress))',
            '  $y = [int][Math]::Round($startY + (([int]$params.y - $startY) * $progress))',
            '  [Win32MoveMouse]::SetCursorPos($x, $y) | Out-Null',
            '  if ($sleepMs -gt 0) { Start-Sleep -Milliseconds $sleepMs }',
            '}',
            '[PSCustomObject]@{',
            '  x = [int]$params.x',
            '  y = [int]$params.y',
            '  durationMs = [int]$params.durationMs',
            '  steps = [int]$params.steps',
            '} | ConvertTo-Json -Compress',
          ].join('\n'),
          { signal: context?.signal },
        )
        return successOutput(result, start)
      } catch (error) {
        return errorOutput(error, start)
      }
    },
  }
}

export function createComputerDragTool(): ToolDefinitionRuntime {
  return {
    name: 'computer.drag',
    description:
      'Drag the Windows mouse pointer from one absolute screen coordinate to another. Use for sliders, selection, and canvas-style GUI actions after observing the target.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        fromX: {
          type: 'number',
          description: 'Absolute screen X coordinate where the drag starts.',
        },
        fromY: {
          type: 'number',
          description: 'Absolute screen Y coordinate where the drag starts.',
        },
        toX: { type: 'number', description: 'Absolute screen X coordinate where the drag ends.' },
        toY: { type: 'number', description: 'Absolute screen Y coordinate where the drag ends.' },
        button: {
          type: 'string',
          enum: ['left', 'right', 'middle'],
          description: 'Mouse button to hold while dragging. Defaults to left.',
        },
        durationMs: {
          type: 'number',
          description: 'Drag duration in milliseconds. Defaults to 400 and caps at 5000.',
        },
        steps: {
          type: 'number',
          description: 'Number of intermediate mouse positions. Defaults to 12 and caps at 100.',
        },
      },
      required: ['fromX', 'fromY', 'toX', 'toY'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      if (!isWindows()) return windowsOnly(start)
      const drag = parseComputerDragInput(input)
      if (!drag.ok) {
        return {
          output: drag.message,
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          durationMs: Date.now() - start,
        }
      }
      const params = {
        ...drag,
        button: parseButton(input.button),
        durationMs: parseNumber(input.durationMs, 400, { min: 0, max: 5000 }),
        steps: parseNumber(input.steps, 12, { min: 1, max: 100 }),
      }
      try {
        const result = await runPowerShellJson<unknown>(
          [
            `$params = ConvertFrom-Json -InputObject '${psJson(params)}'`,
            'Add-Type -AssemblyName System.Windows.Forms',
            'Add-Type @"',
            'using System;',
            'using System.Runtime.InteropServices;',
            'public static class Win32Drag {',
            '  [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);',
            '  [DllImport("user32.dll")] public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);',
            '}',
            '"@',
            '$bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen',
            '$maxX = $bounds.X + $bounds.Width - 1',
            '$maxY = $bounds.Y + $bounds.Height - 1',
            '$points = @(',
            '  @{ label = "from"; x = [int]$params.fromX; y = [int]$params.fromY },',
            '  @{ label = "to"; x = [int]$params.toX; y = [int]$params.toY }',
            ')',
            'foreach ($point in $points) {',
            '  if ($point.x -lt $bounds.X -or $point.x -gt $maxX -or $point.y -lt $bounds.Y -or $point.y -gt $maxY) {',
            '    throw "Drag $($point.label) coordinates ($($point.x), $($point.y)) are outside virtual screen bounds X=$($bounds.X)..$maxX Y=$($bounds.Y)..$maxY"',
            '  }',
            '}',
            '$down = 0x0002; $up = 0x0004',
            'if ($params.button -eq "right") { $down = 0x0008; $up = 0x0010 }',
            'elseif ($params.button -eq "middle") { $down = 0x0020; $up = 0x0040 }',
            '$steps = [int]$params.steps',
            '$sleepMs = if ($steps -gt 0) { [int][Math]::Floor([int]$params.durationMs / $steps) } else { 0 }',
            '[Win32Drag]::SetCursorPos([int]$params.fromX, [int]$params.fromY) | Out-Null',
            'Start-Sleep -Milliseconds 80',
            '[Win32Drag]::mouse_event([uint32]$down, 0, 0, 0, [UIntPtr]::Zero)',
            'for ($i = 1; $i -le $steps; $i++) {',
            '  $progress = [double]$i / [double]$steps',
            '  $x = [int][Math]::Round([int]$params.fromX + (([int]$params.toX - [int]$params.fromX) * $progress))',
            '  $y = [int][Math]::Round([int]$params.fromY + (([int]$params.toY - [int]$params.fromY) * $progress))',
            '  [Win32Drag]::SetCursorPos($x, $y) | Out-Null',
            '  if ($sleepMs -gt 0) { Start-Sleep -Milliseconds $sleepMs }',
            '}',
            '[Win32Drag]::mouse_event([uint32]$up, 0, 0, 0, [UIntPtr]::Zero)',
            '[PSCustomObject]@{',
            '  fromX = [int]$params.fromX',
            '  fromY = [int]$params.fromY',
            '  toX = [int]$params.toX',
            '  toY = [int]$params.toY',
            '  button = $params.button',
            '  durationMs = [int]$params.durationMs',
            '  steps = [int]$params.steps',
            '} | ConvertTo-Json -Compress',
          ].join('\n'),
          { signal: context?.signal },
        )
        return successOutput(result, start)
      } catch (error) {
        return errorOutput(error, start)
      }
    },
  }
}

export function createComputerTypeTextTool(): ToolDefinitionRuntime {
  return {
    name: 'computer.type_text',
    description:
      'Type text into a Windows application. When hwnd, pid, or title is provided, the tool refocuses that target before input. Defaults to clipboard paste for reliable Unicode input.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        text: { type: 'string', description: 'Text to type or paste.' },
        paste: { type: 'boolean', description: 'Use clipboard paste. Defaults to true.' },
        restoreClipboard: {
          type: 'boolean',
          description: 'Restore the previous clipboard contents after paste. Defaults to true.',
        },
        hwnd: {
          type: 'number',
          description: 'Optional target window handle to refocus before typing.',
        },
        pid: {
          type: 'number',
          description: 'Optional target process id to refocus before typing.',
        },
        title: {
          type: 'string',
          description:
            'Optional case-insensitive target window title substring to refocus before typing.',
        },
      },
      required: ['text'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      if (!isWindows()) return windowsOnly(start)
      const params = {
        text: parseString(input.text),
        paste: input.paste !== false,
        restoreClipboard: input.restoreClipboard !== false,
        hwnd: input.hwnd,
        pid: input.pid,
        title: parseString(input.title).trim(),
      }
      try {
        const result = await runPowerShellJson<unknown>(
          [
            `$params = ConvertFrom-Json -InputObject '${psJson(params)}'`,
            'Add-Type -AssemblyName System.Windows.Forms',
            ...buildComputerFocusPowerShellType(),
            ...buildComputerFocusPowerShellFunction(),
            '$focusResult = [PSCustomObject]@{ focused = $false; setForegroundReturned = $false; foregroundHwnd = 0 }',
            '$targetHwnd = [IntPtr]::Zero',
            'if ($params.hwnd) {',
            '  $targetHwnd = [IntPtr]([Int64]$params.hwnd)',
            '} elseif ($params.pid) {',
            '  $targetProcess = Get-Process -Id ([int]$params.pid) -ErrorAction Stop',
            '  $targetHwnd = [IntPtr]$targetProcess.MainWindowHandle',
            '} elseif ($params.title) {',
            '  $targetProcess = Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -like ("*" + $params.title + "*") } | Select-Object -First 1',
            '  if ($targetProcess) { $targetHwnd = [IntPtr]$targetProcess.MainWindowHandle }',
            '}',
            'if ($targetHwnd -ne [IntPtr]::Zero) {',
            '  $focusResult = Invoke-SepilotForeground -TargetHwnd $targetHwnd',
            '  if (-not $focusResult.focused) { throw "Target window could not be focused before typing" }',
            '}',
            '$clipboardRestored = $false',
            'if ($params.paste) {',
            '  $previousClipboardData = $null',
            '  $hadClipboardData = $false',
            '  if ($params.restoreClipboard) {',
            '    try {',
            '      $previousClipboardData = [System.Windows.Forms.Clipboard]::GetDataObject()',
            '      $hadClipboardData = $null -ne $previousClipboardData',
            '    } catch {',
            '      $previousClipboardData = $null',
            '      $hadClipboardData = $false',
            '    }',
            '  }',
            '  [System.Windows.Forms.Clipboard]::SetText([string]$params.text)',
            '  Start-Sleep -Milliseconds 80',
            '  [System.Windows.Forms.SendKeys]::SendWait("^v")',
            '  Start-Sleep -Milliseconds 120',
            '  if ($params.restoreClipboard) {',
            '    try {',
            '      if ($hadClipboardData) {',
            '        [System.Windows.Forms.Clipboard]::SetDataObject($previousClipboardData, $true)',
            '      } else {',
            '        [System.Windows.Forms.Clipboard]::Clear()',
            '      }',
            '      $clipboardRestored = $true',
            '    } catch {',
            '      $clipboardRestored = $false',
            '    }',
            '  }',
            '} else {',
            '  [System.Windows.Forms.SendKeys]::SendWait([string]$params.text)',
            '}',
            '[PSCustomObject]@{ chars = ([string]$params.text).Length; paste = [bool]$params.paste; restoreClipboard = [bool]$params.restoreClipboard; clipboardRestored = [bool]$clipboardRestored; focused = [bool]$focusResult.focused; focusVerified = [bool]$focusResult.focused; foregroundHwnd = [Int64]$focusResult.foregroundHwnd; setForegroundReturned = [bool]$focusResult.setForegroundReturned } | ConvertTo-Json -Compress',
          ].join('\n'),
          { signal: context?.signal },
        )
        return successOutput(result, start)
      } catch (error) {
        return errorOutput(error, start)
      }
    },
  }
}

export function createComputerHotkeyTool(): ToolDefinitionRuntime {
  return {
    name: 'computer.hotkey',
    description:
      'Send a keyboard shortcut to the currently focused Windows application. Examples: ["CTRL","L"], ["ALT","F4"], ["ENTER"].',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        keys: {
          oneOf: [{ type: 'array', items: { type: 'string' } }, { type: 'string' }],
          description: 'Shortcut keys as an array or CTRL+L style string.',
        },
      },
      required: ['keys'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      if (!isWindows()) return windowsOnly(start)
      const keys = normalizeKeys(input.keys)
      const sequence = toSendKeys(keys)
      if (!sequence) {
        return {
          output: 'keys must contain at least one non-modifier key',
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          durationMs: Date.now() - start,
        }
      }
      const params = { keys, sequence }
      try {
        const result = await runPowerShellJson<unknown>(
          [
            `$params = ConvertFrom-Json -InputObject '${psJson(params)}'`,
            'Add-Type -AssemblyName System.Windows.Forms',
            '[System.Windows.Forms.SendKeys]::SendWait([string]$params.sequence)',
            '[PSCustomObject]@{ keys = $params.keys; sequence = $params.sequence } | ConvertTo-Json -Compress -Depth 4',
          ].join('\n'),
          { signal: context?.signal },
        )
        return successOutput(result, start)
      } catch (error) {
        return errorOutput(error, start)
      }
    },
  }
}

export function createComputerScrollTool(): ToolDefinitionRuntime {
  return {
    name: 'computer.scroll',
    description:
      'Scroll the mouse wheel at optional absolute screen coordinates. Negative deltaY scrolls down, positive scrolls up.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        x: { type: 'number', description: 'Optional absolute screen X coordinate.' },
        y: { type: 'number', description: 'Optional absolute screen Y coordinate.' },
        deltaY: {
          type: 'number',
          description: 'Wheel delta. Negative scrolls down, positive scrolls up. Defaults to -600.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      if (!isWindows()) return windowsOnly(start)
      const point = parseComputerPointInput(input, 'optional')
      if (!point.ok) {
        return {
          output: point.message,
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          durationMs: Date.now() - start,
        }
      }
      const params = {
        hasPoint: point.hasPoint,
        x: point.x,
        y: point.y,
        deltaY: parseNumber(input.deltaY, -600, { min: -5000, max: 5000 }),
      }
      try {
        const result = await runPowerShellJson<unknown>(
          [
            `$params = ConvertFrom-Json -InputObject '${psJson(params)}'`,
            'Add-Type -AssemblyName System.Windows.Forms',
            'Add-Type @"',
            'using System;',
            'using System.Runtime.InteropServices;',
            'public static class Win32Wheel {',
            '  [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);',
            '  [DllImport("user32.dll")] public static extern void mouse_event(uint dwFlags, uint dx, uint dy, int dwData, UIntPtr dwExtraInfo);',
            '}',
            '"@',
            'if ($params.hasPoint) {',
            '  $bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen',
            '  $maxX = $bounds.X + $bounds.Width - 1',
            '  $maxY = $bounds.Y + $bounds.Height - 1',
            '  if ([int]$params.x -lt $bounds.X -or [int]$params.x -gt $maxX -or [int]$params.y -lt $bounds.Y -or [int]$params.y -gt $maxY) {',
            '    throw "Scroll coordinates ($($params.x), $($params.y)) are outside virtual screen bounds X=$($bounds.X)..$maxX Y=$($bounds.Y)..$maxY"',
            '  }',
            '}',
            'if ($params.hasPoint) { [Win32Wheel]::SetCursorPos([int]$params.x, [int]$params.y) | Out-Null; Start-Sleep -Milliseconds 50 }',
            '[Win32Wheel]::mouse_event(0x0800, 0, 0, [int]$params.deltaY, [UIntPtr]::Zero)',
            '$resultX = if ($params.hasPoint) { [int]$params.x } else { $null }',
            '$resultY = if ($params.hasPoint) { [int]$params.y } else { $null }',
            '[PSCustomObject]@{ deltaY = [int]$params.deltaY; x = $resultX; y = $resultY } | ConvertTo-Json -Compress',
          ].join('\n'),
          { signal: context?.signal },
        )
        return successOutput(result, start)
      } catch (error) {
        return errorOutput(error, start)
      }
    },
  }
}

export function createComputerWaitTool(): ToolDefinitionRuntime {
  return {
    name: 'computer.wait',
    description: 'Wait for a short period before observing again after a GUI action.',
    resumeSafety: 'replay-safe',
    inputSchema: {
      type: 'object',
      properties: {
        ms: {
          type: 'number',
          description: 'Milliseconds to wait. Defaults to 1000 and caps at 30000.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const ms = parseNumber(input.ms, 1000, { min: 0, max: 30_000 })
      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(resolve, ms)
        context?.signal?.addEventListener(
          'abort',
          () => {
            clearTimeout(timer)
            reject(new Error('computer.wait aborted'))
          },
          { once: true },
        )
      })
      return successOutput({ waitedMs: ms }, start)
    },
  }
}

/**
 * Walk a Windows window's UI Automation tree and return element names by
 * control type. Use this instead of OCR/vision when the target surface
 * uses standard Win32 controls.
 */
export function createComputerListElementsTool(): ToolDefinitionRuntime {
  return {
    name: 'computer.list_elements',
    description:
      'Enumerate visible UI Automation elements inside a Windows window. More reliable than reading a screenshot for standard Win32 controls such as Explorer, dialogs, and settings panels.',
    resumeSafety: 'replay-safe',
    inputSchema: {
      type: 'object',
      properties: {
        hwnd: { type: 'number', description: 'Window handle from computer.list_windows.' },
        pid: { type: 'number', description: 'Process id; the tool resolves the main window.' },
        title: {
          type: 'string',
          description: 'Case-insensitive title substring; first matching window wins.',
        },
        controlTypes: {
          type: 'array',
          items: { type: 'string' },
          description: 'Control types to keep. Defaults to ["ListItem","Button","TreeItem"].',
        },
        max: {
          type: 'number',
          description: 'Maximum elements to return. Defaults to 200.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      if (!isWindows()) return windowsOnly(start)
      const params = {
        hwnd: input.hwnd,
        pid: input.pid,
        title: parseString(input.title).trim(),
        controlTypes: Array.isArray(input.controlTypes)
          ? input.controlTypes.map((value) => String(value))
          : ['ListItem', 'Button', 'TreeItem'],
        max: parseNumber(input.max, 200, { min: 1, max: 1000 }),
      }
      try {
        const result = await runPowerShellJson<unknown>(
          [
            `$params = ConvertFrom-Json -InputObject '${psJson(params)}'`,
            'Add-Type -AssemblyName UIAutomationClient',
            'Add-Type -AssemblyName UIAutomationTypes',
            '$hwnd = [IntPtr]::Zero',
            'if ($params.hwnd) { $hwnd = [IntPtr]([Int64]$params.hwnd) }',
            'elseif ($params.pid) {',
            '  $proc = Get-Process -Id ([int]$params.pid) -ErrorAction SilentlyContinue',
            '  if ($proc) { $hwnd = [IntPtr]$proc.MainWindowHandle }',
            '} elseif ($params.title) {',
            '  $proc = Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -like ("*" + $params.title + "*") } | Select-Object -First 1',
            '  if ($proc) { $hwnd = [IntPtr]$proc.MainWindowHandle }',
            '}',
            'if ($hwnd -eq [IntPtr]::Zero) { throw "No matching window found" }',
            '$ctMap = @{',
            '  "ListItem" = [System.Windows.Automation.ControlType]::ListItem',
            '  "Button" = [System.Windows.Automation.ControlType]::Button',
            '  "TreeItem" = [System.Windows.Automation.ControlType]::TreeItem',
            '  "Edit" = [System.Windows.Automation.ControlType]::Edit',
            '  "Text" = [System.Windows.Automation.ControlType]::Text',
            '  "MenuItem" = [System.Windows.Automation.ControlType]::MenuItem',
            '  "TabItem" = [System.Windows.Automation.ControlType]::TabItem',
            '  "CheckBox" = [System.Windows.Automation.ControlType]::CheckBox',
            '  "RadioButton" = [System.Windows.Automation.ControlType]::RadioButton',
            '  "Hyperlink" = [System.Windows.Automation.ControlType]::Hyperlink',
            '}',
            '$wanted = @()',
            'foreach ($n in $params.controlTypes) { if ($ctMap.ContainsKey([string]$n)) { $wanted += $ctMap[[string]$n] } }',
            'if ($wanted.Count -eq 0) { throw "No supported controlTypes" }',
            '$root = [System.Windows.Automation.AutomationElement]::FromHandle($hwnd)',
            'if (-not $root) { throw "FromHandle returned null for hwnd=$hwnd" }',
            '$controlTypeProp = [System.Windows.Automation.AutomationElement]::ControlTypeProperty',
            '$conditions = @()',
            'foreach ($ct in $wanted) {',
            '  $conditions += New-Object System.Windows.Automation.PropertyCondition($controlTypeProp, $ct)',
            '}',
            '$cond = if ($conditions.Count -eq 1) { $conditions[0] } else { New-Object System.Windows.Automation.OrCondition($conditions) }',
            '$elements = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)',
            '$out = New-Object System.Collections.ArrayList',
            '$count = 0',
            'foreach ($el in $elements) {',
            '  if ($count -ge [int]$params.max) { break }',
            '  $name = ""',
            '  try { $name = [string]$el.Current.Name } catch { }',
            '  if ([string]::IsNullOrEmpty($name)) { continue }',
            '  $ctName = ""',
            '  try { $ctName = $el.Current.ControlType.LocalizedControlType } catch { }',
            '  $aid = ""',
            '  try { $aid = [string]$el.Current.AutomationId } catch { }',
            '  $rect = $null',
            '  try {',
            '    $r = $el.Current.BoundingRectangle',
            '    if ($r -and -not $r.IsEmpty) {',
            '      $rect = [PSCustomObject]@{ x=[int]$r.X; y=[int]$r.Y; w=[int]$r.Width; h=[int]$r.Height }',
            '    }',
            '  } catch { }',
            '  [void]$out.Add([PSCustomObject]@{ name = $name; controlType = $ctName; automationId = $aid; bounds = $rect })',
            '  $count++',
            '}',
            '[PSCustomObject]@{',
            '  hwnd = $hwnd.ToInt64()',
            '  controlTypes = $params.controlTypes',
            '  count = $out.Count',
            '  truncated = ($elements.Count -gt $out.Count)',
            '  elements = $out',
            '} | ConvertTo-Json -Compress -Depth 6',
          ].join('\n'),
          { signal: context?.signal, timeoutMs: 20_000 },
        )
        return successOutput(result, start)
      } catch (error) {
        return errorOutput(error, start)
      }
    },
  }
}

export function createComputerUseTools(): ToolDefinitionRuntime[] {
  return [
    createComputerListWindowsTool(),
    createComputerLaunchAppTool(),
    createComputerOpenUrlTool(),
    createComputerFocusWindowTool(),
    createComputerObserveTool(),
    createComputerMoveMouseTool(),
    createComputerClickTool(),
    createComputerDragTool(),
    createComputerTypeTextTool(),
    createComputerHotkeyTool(),
    createComputerScrollTool(),
    createComputerWaitTool(),
    createComputerListElementsTool(),
  ]
}
