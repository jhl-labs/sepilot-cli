import { readFile, stat } from 'node:fs/promises'
import type { ContentPart, Message } from '@sepilotd/core'
import { AgentGraph } from '../engine.js'
import type { Deps } from '../nodes.js'
import type { AgentState } from '../types.js'
import * as N from '../nodes.js'
import { withGraphPrerequisite } from './prerequisite.js'
import { isProviderModelImageInputRejected } from '../../../providers/vision-capability-state.js'

const computerUsePrompt = [
  'You are a computer-use agent for Windows desktop GUI automation.',
  'Operate slowly and visibly: list or focus the target window, observe the screen, take at most one GUI action, wait briefly, then observe again.',
  'For a simple Windows smoke/demo task, you may launch a supported built-in app such as notepad with computer.launch_app, then observe before typing or clicking.',
  'When the user explicitly wants a site, page, link, or map shown in their real desktop browser, use computer.open_url. It opens the default system browser visibly and requires human approval; browser.navigate remains the headless inspection tool.',
  'When typing into a known target, pass hwnd or pid from the latest computer.launch_app, computer.focus_window, computer.list_windows, or computer.observe result so the tool can refocus the intended window before input.',
  'Use only computer.* tools for GUI control unless the user explicitly asks for supporting file, terminal, or web work.',
  'For web search, source comparison, or research tasks, prefer available web/browser/search/extraction tools over slow manual GUI navigation unless the user specifically wants visible desktop interaction.',
  'Never enter passwords, payment details, two-factor codes, or sensitive personal data. Never click destructive, purchase, send, or irreversible controls without explicit user confirmation.',
  'Coordinates are absolute screen pixels from computer.observe metadata. Use computer.move_mouse for hover states and computer.drag for sliders, selections, and canvas-like interactions. If the target is unclear, ask the user instead of guessing.',
  'When the task is complete, summarize what changed and any uncertainty.',
].join(' ')

const COMPUTER_OBSERVATION_MESSAGE_PREFIX = '[Computer observation '
const MAX_SCREENSHOT_BYTES = 20 * 1024 * 1024
const MAX_INJECTED_OBSERVATION_MESSAGES = 3
const MAX_TRACKED_OBSERVATION_IDS = 32
const COMPUTER_GUI_ACTION_TOOLS = new Set([
  'computer.launch_app',
  'computer.open_url',
  'computer.focus_window',
  'computer.move_mouse',
  'computer.click',
  'computer.drag',
  'computer.type_text',
  'computer.hotkey',
  'computer.scroll',
])
const COMPUTER_OBSERVATION_TOOLS = new Set([
  'computer.list_windows',
  'computer.list_elements',
  'computer.observe',
  'computer.wait',
])
const COMPUTER_FOREGROUND_AUTO_OBSERVE_TOOLS = new Set([
  'computer.launch_app',
  'computer.open_url',
  'computer.focus_window',
  'computer.move_mouse',
  'computer.drag',
  'computer.type_text',
  'computer.hotkey',
])

function isComputerTool(name: string): boolean {
  return name.startsWith('computer.')
}

function hasComputerObservationToolBeforeAction(state: AgentState, actionIndex: number): boolean {
  return state.toolCalls
    .slice(0, actionIndex)
    .some((toolCall) => COMPUTER_OBSERVATION_TOOLS.has(toolCall.name))
}

interface ComputerObservation {
  observationId: string
  path: string
  mimeType: string
  width?: number
  height?: number
  originX?: number
  originY?: number
  capturedAt?: string
  coordinateSystem?: string
  scope?: string
  note?: string
  foregroundWindow?: {
    hwnd?: number
    pid?: number
    processName?: string
    title?: string
    bounds?: ComputerRect
  }
  cursor?: {
    x?: number
    y?: number
  }
}

interface ComputerRect {
  x?: number
  y?: number
  width?: number
  height?: number
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function parseString(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

function parseNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function parseForegroundWindow(value: unknown): ComputerObservation['foregroundWindow'] {
  if (!isRecord(value)) return undefined
  const foregroundWindow = {
    hwnd: parseNumber(value.hwnd),
    pid: parseNumber(value.pid),
    processName: parseString(value.processName),
    title: parseString(value.title),
    bounds: parseRect(value.bounds),
  }
  return foregroundWindow.hwnd !== undefined ||
    foregroundWindow.pid !== undefined ||
    foregroundWindow.processName ||
    foregroundWindow.title ||
    foregroundWindow.bounds
    ? foregroundWindow
    : undefined
}

function parseRect(value: unknown): ComputerRect | undefined {
  if (!isRecord(value)) return undefined
  const rect = {
    x: parseNumber(value.x),
    y: parseNumber(value.y),
    width: parseNumber(value.width),
    height: parseNumber(value.height),
  }
  return rect.x !== undefined &&
    rect.y !== undefined &&
    rect.width !== undefined &&
    rect.height !== undefined
    ? rect
    : undefined
}

function parseCursor(value: unknown): ComputerObservation['cursor'] {
  if (!isRecord(value)) return undefined
  const cursor = {
    x: parseNumber(value.x),
    y: parseNumber(value.y),
  }
  return cursor.x !== undefined && cursor.y !== undefined ? cursor : undefined
}

function parseComputerObservation(output: string): ComputerObservation | null {
  try {
    const parsed = JSON.parse(output)
    if (!isRecord(parsed)) return null

    const observationId = parseString(parsed.observationId)
    const path = parseString(parsed.path)
    if (!observationId || !path) return null

    return {
      observationId,
      path,
      mimeType: parseString(parsed.mimeType) ?? 'image/png',
      width: parseNumber(parsed.width),
      height: parseNumber(parsed.height),
      originX: parseNumber(parsed.originX),
      originY: parseNumber(parsed.originY),
      capturedAt: parseString(parsed.capturedAt),
      coordinateSystem: parseString(parsed.coordinateSystem),
      scope: parseString(parsed.scope),
      note: parseString(parsed.note),
      foregroundWindow: parseForegroundWindow(parsed.foregroundWindow),
      cursor: parseCursor(parsed.cursor),
    }
  } catch {
    return null
  }
}

function findLatestUninjectedObservation(state: AgentState): ComputerObservation | null {
  const injected = new Set(state.computerUseObservationIds ?? [])
  for (const result of [...(state.recentToolResults ?? [])].reverse()) {
    if (result.toolName !== 'computer.observe' || result.status !== 'success') {
      continue
    }
    const observation = parseComputerObservation(result.output)
    if (observation && !injected.has(observation.observationId)) {
      return observation
    }
  }
  return null
}

function markObservationHandled(state: AgentState, observationId: string): void {
  state.computerUseObservationIds = [
    ...(state.computerUseObservationIds ?? []).filter((id) => id !== observationId),
    observationId,
  ].slice(-MAX_TRACKED_OBSERVATION_IDS)
}

function isComputerObservationMessage(message: Message): boolean {
  if (message.role !== 'user' || !Array.isArray(message.content)) {
    return false
  }
  const firstText = message.content.find((part) => part.type === 'text')
  return (
    firstText?.type === 'text' && firstText.text.startsWith(COMPUTER_OBSERVATION_MESSAGE_PREFIX)
  )
}

function trimInjectedObservationMessages(messages: Message[]): Message[] {
  const observationIndexes = messages
    .map((message, index) => (isComputerObservationMessage(message) ? index : -1))
    .filter((index) => index >= 0)

  if (observationIndexes.length <= MAX_INJECTED_OBSERVATION_MESSAGES) {
    return messages
  }

  const keep = new Set(observationIndexes.slice(-MAX_INJECTED_OBSERVATION_MESSAGES))
  return messages.filter(
    (message, index) => !isComputerObservationMessage(message) || keep.has(index),
  )
}

function buildObservationText(observation: ComputerObservation): string {
  const originX = observation.originX ?? 0
  const originY = observation.originY ?? 0
  const foreground = observation.foregroundWindow
    ? formatForegroundWindow(observation.foregroundWindow)
    : undefined
  return [
    `${COMPUTER_OBSERVATION_MESSAGE_PREFIX}${observation.observationId}]`,
    observation.width && observation.height
      ? `Screenshot size: ${observation.width}x${observation.height}.`
      : '',
    observation.scope ? `Observation scope: ${observation.scope}.` : '',
    `Coordinate system: ${observation.coordinateSystem ?? 'absolute-screen-pixels'}.`,
    `Screen origin: (${originX}, ${originY}). Use absolute screen pixel coordinates for computer.click.`,
    foreground ? `Foreground window: ${foreground}.` : '',
    observation.foregroundWindow?.bounds
      ? `Foreground bounds: ${formatRect(observation.foregroundWindow.bounds)}.`
      : '',
    observation.cursor?.x !== undefined && observation.cursor.y !== undefined
      ? `Cursor position: (${observation.cursor.x}, ${observation.cursor.y}).`
      : '',
    observation.capturedAt ? `Captured at: ${observation.capturedAt}.` : '',
    observation.note ? `Observation note: ${observation.note}.` : '',
  ]
    .filter(Boolean)
    .join('\n')
}

function formatForegroundWindow(
  foregroundWindow: NonNullable<ComputerObservation['foregroundWindow']>,
): string {
  const title = foregroundWindow.title ?? 'Untitled window'
  const meta = [
    foregroundWindow.processName,
    foregroundWindow.pid !== undefined ? `pid ${foregroundWindow.pid}` : '',
    foregroundWindow.hwnd !== undefined ? `hwnd ${foregroundWindow.hwnd}` : '',
  ].filter(Boolean)
  return meta.length > 0 ? `${title} (${meta.join(', ')})` : title
}

function formatRect(rect: ComputerRect): string {
  return `x=${rect.x}, y=${rect.y}, width=${rect.width}, height=${rect.height}`
}

function appendObservationUnavailableMessage(
  state: AgentState,
  observation: ComputerObservation,
  reason: string,
): void {
  state.messages.push({
    role: 'system',
    content: [
      `[Computer observation unavailable ${observation.observationId}]`,
      reason,
      'Call computer.observe again before choosing visual coordinates.',
    ].join('\n'),
  })
}

export async function injectComputerObservationContext(state: AgentState): Promise<AgentState> {
  const observation = findLatestUninjectedObservation(state)
  if (!observation) return state

  try {
    const file = await stat(observation.path)
    if (!file.isFile()) {
      markObservationHandled(state, observation.observationId)
      appendObservationUnavailableMessage(
        state,
        observation,
        `Observation path is not a file: ${observation.path}`,
      )
      return state
    }
    if (file.size > MAX_SCREENSHOT_BYTES) {
      markObservationHandled(state, observation.observationId)
      appendObservationUnavailableMessage(
        state,
        observation,
        `Screenshot is too large for model context: ${file.size} bytes.`,
      )
      return state
    }

    const data = await readFile(observation.path)
    const content: ContentPart[] = [
      { type: 'text', text: buildObservationText(observation) },
      {
        type: 'image',
        source: {
          type: 'base64',
          mediaType: observation.mimeType,
          data: data.toString('base64'),
        },
      },
    ]
    state.messages.push({ role: 'user', content })
    state.messages = trimInjectedObservationMessages(state.messages)
    markObservationHandled(state, observation.observationId)
  } catch (error) {
    markObservationHandled(state, observation.observationId)
    appendObservationUnavailableMessage(
      state,
      observation,
      error instanceof Error ? error.message : String(error),
    )
  }

  return state
}

function latestGuiActionAwaitingObservation(state: AgentState): string | null {
  let actionIndex = -1
  let actionToolName: string | null = null

  for (const [index, result] of state.toolResults.entries()) {
    if (
      result.status === 'success' &&
      result.toolName &&
      COMPUTER_GUI_ACTION_TOOLS.has(result.toolName)
    ) {
      actionIndex = index
      actionToolName = result.toolName
    }
  }

  if (actionIndex < 0 || !actionToolName) return null

  const observedAfterAction = state.toolResults
    .slice(actionIndex + 1)
    .some((result) => result.toolName === 'computer.observe')
  return observedAfterAction ? null : actionToolName
}

export async function queueComputerUsePostActionObservation(
  state: AgentState,
): Promise<AgentState> {
  const actionToolName = latestGuiActionAwaitingObservation(state)
  if (!actionToolName) return state
  const scope = COMPUTER_FOREGROUND_AUTO_OBSERVE_TOOLS.has(actionToolName)
    ? 'foreground'
    : undefined

  state.toolCalls = [
    {
      id: `computer-auto-observe-${state.iteration}-${state.toolResults.length}`,
      name: 'computer.observe',
      arguments: {
        note: `after ${actionToolName}`,
        ...(scope ? { scope } : {}),
      },
    },
  ]
  return state
}

function hasFreshVisualObservation(state: AgentState): boolean {
  for (const result of [...(state.recentToolResults ?? []), ...state.toolResults].reverse()) {
    if (result.toolName && COMPUTER_GUI_ACTION_TOOLS.has(result.toolName)) return false
    if (result.toolName !== 'computer.observe') continue
    const observation = result.status === 'success' ? parseComputerObservation(result.output) : null
    if (!observation) return false
    return state.messages.some((message) => isComputerObservationMessage(message)
      && Array.isArray(message.content)
      && message.content.some((part) => part.type === 'image')
      && message.content.some((part) => part.type === 'text'
        && part.text.startsWith(`${COMPUTER_OBSERVATION_MESSAGE_PREFIX}${observation.observationId}]`)))
  }
  return false
}

export async function enforceSingleComputerGuiAction(state: AgentState): Promise<AgentState> {
  const firstActionIndex = state.toolCalls.findIndex((toolCall) =>
    COMPUTER_GUI_ACTION_TOOLS.has(toolCall.name),
  )
  if (firstActionIndex < 0) return state

  const mustInspectPriorObservation = hasComputerObservationToolBeforeAction(
    state,
    firstActionIndex,
  )
  const action = state.toolCalls[firstActionIndex]!
  const needsScreen = !['computer.launch_app', 'computer.open_url', 'computer.focus_window'].includes(action.name)
  if (needsScreen && !mustInspectPriorObservation && !hasFreshVisualObservation(state)) {
    state.toolCalls = [...state.toolCalls.slice(0, firstActionIndex), {
      id: `computer-required-observe-${state.iteration}-${state.toolResults.length}`,
      name: 'computer.observe', arguments: { note: 'fresh screenshot required before GUI input' },
    }]
    state.messages.push({ role: 'system', content: 'Deferred GUI input: inspect a successful, current screenshot before choosing the next action.' })
    return state
  }
  const keepCount = mustInspectPriorObservation ? firstActionIndex : firstActionIndex + 1
  const keptToolCalls = state.toolCalls.slice(0, keepCount)
  const deferredToolCalls = state.toolCalls.slice(keepCount)
  if (deferredToolCalls.length === 0) return state

  const deferredComputerTools = deferredToolCalls
    .filter((toolCall) => isComputerTool(toolCall.name))
    .map((toolCall) => toolCall.name)
  if (deferredComputerTools.length === 0) return state

  state.toolCalls = keptToolCalls
  state.messages.push({
    role: 'system',
    content: [
      '[Computer-use safety guard]',
      mustInspectPriorObservation
        ? 'Deferred GUI actions until the model inspects the requested window/screen observation result.'
        : 'Deferred additional computer-use calls after the first GUI action. Observe the screen before choosing the next action.',
      `Deferred tools: ${deferredComputerTools.join(', ')}.`,
    ].join('\n'),
  })
  return state
}

export function buildComputerUseGraph(deps: Deps) {
  const agentDeps = {
    ...deps,
    systemPrompt: [deps.systemPrompt ?? '', computerUsePrompt].filter(Boolean).join('\n\n'),
  }

  const graph = new AgentGraph()
    .setStart('vision_guard')
    .addNode('vision_guard', async (state: AgentState) => state)
    .addEdge('vision_guard', 'context_manager')
    .addNode('context_manager', N.contextManager(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('observation_context', injectComputerObservationContext, {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('agent', N.agent(agentDeps), {
      lifecycleState: 'thinking',
    })
    .addNode('computer_action_guard', enforceSingleComputerGuiAction, {
      lifecycleState: 'thinking',
    })
    .addNode('tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })
    .addNode('post_action_observation', queueComputerUsePostActionObservation, {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('reflection', N.reflection({ critiqueDeps: deps }), {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('iteration_guard', N.iterationGuard(), {
      lifecycleState: 'thinking',
    })
    .addNode('reporter', N.reporter({ enableSkillExtraction: true }), {
      lifecycleState: 'done',
    })
    .addEdge('context_manager', 'observation_context')
    .addEdge('observation_context', 'agent')
    .addConditionalEdge(
      'agent',
      (s: AgentState) => (s.toolCalls.length > 0 ? 'computer_action_guard' : 'reporter'),
      ['computer_action_guard', 'reporter'],
    )
    .addConditionalEdge(
      'computer_action_guard',
      (s: AgentState) => (s.toolCalls.length > 0 ? 'tools' : 'reporter'),
      ['tools', 'reporter'],
    )
    .addEdge('tools', 'post_action_observation')
    .addConditionalEdge(
      'post_action_observation',
      (s: AgentState) => (s.toolCalls.length > 0 ? 'tools' : 'reflection'),
      ['tools', 'reflection'],
    )
    .addEdge('reflection', 'iteration_guard')
    .addConditionalEdge(
      'iteration_guard',
      (s: AgentState) => (s.shouldStop ? 'reporter' : 'observation_context'),
      ['reporter', 'observation_context'],
    )
    .addEdge('reporter', '__end__')
  return withGraphPrerequisite(graph, (state, context, node) => {
    // Text-only finalization/critique may use an auxiliary non-vision model.
    if (['reflection', 'reporter', 'iteration_guard'].includes(node)) return undefined
    const modelId = context?.agentContext.model ?? deps.provider.models[0]?.id ?? ''
    const model = deps.provider.models.find((candidate) => candidate.id === modelId)
    if (model?.capabilities.vision !== true || state.visualAttachmentsDisabled
      || isProviderModelImageInputRejected(deps.provider.id, modelId)) {
      return 'INCOMPLETE: 모델 미지원: Computer Use는 이미지 입력을 지원하는 모델이 필요합니다.'
    }
    return undefined
  })
}
