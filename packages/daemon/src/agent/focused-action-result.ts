import { extractFocusedProcessStart } from './request-shape.js'
import { inputRequestsFocusedSingleBrowserObservation } from './task-contract.js'
import { inputRequestsFocusedSingleProcessObservation } from './action-completion.js'
import { isVisibleUrlOpenRequest } from './desktop-control-intent.js'

interface ObservedActionResult {
  output: string
  status: 'success' | 'error'
  toolName: string
}

const FOCUSED_BROWSER_TOOLS = new Set([
  'browser.navigate',
  'browser.extract',
  'browser.click',
  'browser.evaluate',
])
const FOCUSED_PROCESS_OBSERVATION_TOOLS = new Set([
  'process.follow',
  'process.list',
  'process.read',
  'process.sessions',
  'process.wait',
])

function textField(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function numberField(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function processEndpoint(value: unknown): string | undefined {
  if (!value || typeof value !== 'object') return undefined
  const loopback = value as Record<string, unknown>
  const ports = Array.isArray(loopback.ports)
    ? loopback.ports.filter((port): port is number => (
        typeof port === 'number' && Number.isInteger(port) && port >= 1 && port <= 65_535
      ))
    : []
  return ports[0] ? `http://127.0.0.1:${ports[0]}` : undefined
}

function formatProcessStartResult(output: string, korean: boolean): string {
  let record: Record<string, unknown> | undefined
  try {
    const parsed: unknown = JSON.parse(output)
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      record = parsed as Record<string, unknown>
    }
  } catch {
    // A third-party process tool may return plain text. Its successful output
    // is still grounded evidence, so retain it verbatim below.
  }

  if (!record) {
    return korean
      ? `ANSWER: process.start가 성공했습니다.\n${output.trim()}`
      : `ANSWER: process.start succeeded.\n${output.trim()}`
  }

  const id = textField(record.id)
  const pid = numberField(record.pid)
  const command = textField(record.command)
  const status = textField(record.status)
  const lifetime = textField(record.lifetime)
  const endpoint = processEndpoint(record.loopback)
  const rows = korean
    ? [
        'ANSWER: 프로세스를 시작했습니다.',
        status ? `- 상태: ${status}` : undefined,
        pid !== undefined ? `- PID: ${pid}` : undefined,
        id ? `- process id: ${id}` : undefined,
        command ? `- 명령: ${command}` : undefined,
        lifetime ? `- lifetime: ${lifetime}` : undefined,
        endpoint ? `- endpoint: ${endpoint}` : undefined,
      ]
    : [
        'ANSWER: The process started successfully.',
        status ? `- Status: ${status}` : undefined,
        pid !== undefined ? `- PID: ${pid}` : undefined,
        id ? `- Process id: ${id}` : undefined,
        command ? `- Command: ${command}` : undefined,
        lifetime ? `- Lifetime: ${lifetime}` : undefined,
        endpoint ? `- Endpoint: ${endpoint}` : undefined,
      ]
  return rows.filter(Boolean).join('\n')
}

function formatFocusedActionFailure(
  result: ObservedActionResult,
  korean: boolean,
): string {
  const output = result.output.trim()
  const heading = korean
    ? `INCOMPLETE: ${result.toolName} 실행에 실패했습니다.`
    : `INCOMPLETE: ${result.toolName} failed.`
  return output ? `${heading}\n${output}` : heading
}

function formatVisibleUrlOpenResult(output: string, korean: boolean): string {
  let url: string | undefined
  try {
    const parsed: unknown = JSON.parse(output)
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      url = textField((parsed as Record<string, unknown>).url)
    }
  } catch {
    // Preserve a successful third-party implementation's plain-text receipt.
  }

  if (korean) {
    return url
      ? `ANSWER: 기본 브라우저에서 요청한 페이지를 열었습니다.\n- URL: ${url}`
      : `ANSWER: 기본 브라우저에서 요청한 페이지를 열었습니다.${output.trim() ? `\n${output.trim()}` : ''}`
  }
  return url
    ? `ANSWER: Opened the requested page in the default browser.\n- URL: ${url}`
    : `ANSWER: Opened the requested page in the default browser.${output.trim() ? `\n${output.trim()}` : ''}`
}

/**
 * Finish a structurally bounded, single-action request from its successful
 * current-turn result. There is no reason to ask a provider to restate this
 * evidence, and provider-empty final replies must not turn one successful
 * action into repeated tool/reviewer cycles.
 */
export function deterministicFocusedActionResult(
  input: string,
  result: ObservedActionResult,
): string | undefined {
  const korean = /[가-힣]/u.test(input)

  if (result.toolName === 'computer.open_url' && isVisibleUrlOpenRequest(input)) {
    return result.status === 'success'
      ? formatVisibleUrlOpenResult(result.output, korean)
      : formatFocusedActionFailure(result, korean)
  }

  if (result.toolName === 'process.start' && extractFocusedProcessStart(input)) {
    return result.status === 'success'
      ? formatProcessStartResult(result.output, korean)
      : formatFocusedActionFailure(result, korean)
  }

  if (
    FOCUSED_BROWSER_TOOLS.has(result.toolName)
    && inputRequestsFocusedSingleBrowserObservation(input)
  ) {
    if (result.status === 'error') {
      return formatFocusedActionFailure(result, korean)
    }
    const output = result.output.trim()
    return output ? `ANSWER:\n${output}` : 'ANSWER: The browser observation completed successfully with no text output.'
  }

  if (
    FOCUSED_PROCESS_OBSERVATION_TOOLS.has(result.toolName)
    && inputRequestsFocusedSingleProcessObservation(input)
  ) {
    if (result.status === 'error') {
      return formatFocusedActionFailure(result, korean)
    }
    const output = result.output.trim()
    return output
      ? `ANSWER:\n${output}`
      : 'ANSWER: The process observation completed successfully with no text output.'
  }

  return undefined
}
