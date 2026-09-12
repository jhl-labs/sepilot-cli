import type { AgentEvent, AgentExecutionIntent, AgentRunContract, Message, TokenUsage, ToolDefinition } from '@sepilotd/core'
import { z } from 'zod'
import type { AgentState } from './graph/types.js'
import type { ToolRegistry } from '../tools/registry.js'
import { toolExposureGroupForTool } from '../tools/role-filter.js'
import { INSTANT_MEMORY_TOOL_NAMES } from './instant-mode-tool-intent.js'

export const MODE_TRANSFER_TOOL = 'agent.transfer'
export const TOOL_CATALOG_TOOL = 'agent.tools'
export const INSTANT_MAX_ITERATIONS = 4

export interface ExecutionHandoff {
  messages: Message[]
  usage: TokenUsage
  iterations: number
  graphState?: AgentState
}

export interface ModeTransfer {
  mode: string
  reason: string
  tools: string[]
  executionIntent: AgentExecutionIntent
  outcomes: string[]
  constraints: string[]
}

/** New capability needs may expand visibility; explicit restrictions survive. */
export function mergeModeTransferContract(
  current: AgentRunContract | undefined,
  transfer: ModeTransfer,
  input: string,
): AgentRunContract {
  const previous = current?.executionIntent
  const next = transfer.executionIntent
  const allowedTools = previous?.allowedTools === undefined ? next.allowedTools
    : next.allowedTools === undefined ? previous.allowedTools
      : previous.allowedTools.filter((name) => next.allowedTools!.includes(name))
  return {
    ...(current ?? {
      source: 'planner', summary: input,
      acceptanceCriteria: transfer.outcomes.map((text, index) => ({ id: `outcome-${index + 1}`, text })),
      outOfScope: [],
    }),
    constraints: [...new Set([...(current?.constraints ?? []), ...transfer.constraints])],
    executionIntent: {
      ...next,
      capabilities: [...new Set([...(previous?.capabilities ?? []), ...next.capabilities])],
      workspaceMutation: previous?.workspaceMutation === 'forbidden' || next.workspaceMutation === 'forbidden'
        ? 'forbidden' : previous?.workspaceMutation === 'required' ? 'required' : next.workspaceMutation,
      ...(allowedTools !== undefined ? { allowedTools } : {}),
      ...(previous?.toolSequence ? { toolSequence: previous.toolSequence } : {}),
      ...(previous?.retryPolicy ? { retryPolicy: previous.retryPolicy } : {}),
      ...(previous?.authorizedWriteTargets ? { authorizedWriteTargets: previous.authorizedWriteTargets } : {}),
      protectedWriteTargets: [...new Set([...(previous?.protectedWriteTargets ?? []), ...(next.protectedWriteTargets ?? [])])],
    },
  }
}

export interface ModeControl {
  state: NonNullable<import('@sepilotd/core').AgentContext['modeControlState']>
  tools: ToolDefinition[]
  prompt: string
  pending(): boolean
  drainEvents(): AgentEvent[]
  handle(response: Message, snapshot: ExecutionHandoff): 'unhandled' | 'continue' | 'transfer'
}

const transferSchema = z.object({
  mode: z.string().min(1),
  reason: z.string().min(1).max(600),
  tools: z.array(z.string()).max(32).default([]),
  outcomes: z.array(z.string().min(1).max(400)).min(1).max(8),
  constraints: z.array(z.string().min(1).max(400)).max(12).optional(),
  allowedTools: z.array(z.string()).max(64).optional(),
  toolSequence: z.array(z.string()).max(32).optional(),
  retryPolicy: z.literal('forbidden').optional(),
  authorizedWriteTargets: z.array(z.string()).max(32).optional(),
  protectedWriteTargets: z.array(z.string()).max(32).optional(),
  kind: z.enum(['operational-action', 'workspace-change', 'inspection', 'artifact-production', 'conversation']),
  workspaceMutation: z.enum(['forbidden', 'allowed', 'required']),
  capabilities: z.array(z.enum(['process', 'service', 'terminal', 'browser', 'filesystem-read', 'filesystem-write', 'network', 'application-state'])).max(8),
})

/** Internal control messages change execution strategy, never tool authority. */
export function createModeControl(options: {
  modes: Array<{ id: string; description: string; capabilities?: readonly string[] }>
  currentMode: string
  context?: import('@sepilotd/core').AgentContext
  activeToolNames?: readonly string[]
  transferCount?: number
  turnMaxIterations?: number
  tools: ToolRegistry
  canTransfer: boolean
  discoveryContext?: string
  onTransfer(transfer: ModeTransfer, snapshot: ExecutionHandoff): void
}): ModeControl {
  let transferred = false
  const events: AgentEvent[] = []
  const catalog = options.tools.toToolDefinitions(options.context).map((tool) => ({
    name: tool.name,
    description: tool.description.slice(0, 200),
    effect: options.tools.securityDescriptor(tool.name).effect,
    group: toolExposureGroupForTool(tool.name) ?? 'general',
  }))
  const groups = [...new Set(catalog.map((tool) => tool.group))]
  const tools: ToolDefinition[] = options.canTransfer ? [{
    name: MODE_TRANSFER_TOOL,
    description: 'Continue this same task in another execution mode, or activate additional tools in react. Carry current evidence forward. This does not grant permissions. Call alone, without other tool calls.',
    inputSchema: {
      type: 'object', additionalProperties: false,
      properties: {
        mode: { type: 'string', enum: options.modes.map((mode) => mode.id) },
        reason: { type: 'string', description: 'Why the remaining work needs this mode.' },
        tools: { type: 'array', items: { type: 'string' }, maxItems: 32, description: 'Exact additional tool names from agent.tools; omit or pass an empty array when the mode default suffices. This is a visibility preference, not an authorization boundary.' },
        outcomes: { type: 'array', items: { type: 'string' }, minItems: 1, maxItems: 8, description: 'Concrete completion criteria for the whole user goal, retaining completed prerequisites and remaining work.' },
        constraints: { type: 'array', items: { type: 'string' }, description: 'User constraints that must survive the transfer.' },
        allowedTools: { type: 'array', items: { type: 'string' }, description: 'Only when the user explicitly restricts execution to exact tool names. Omit otherwise. Empty forbids all tools.' },
        toolSequence: { type: 'array', items: { type: 'string' }, description: 'Only when the user explicitly requires distinct tools in this order.' },
        retryPolicy: { type: 'string', enum: ['forbidden'], description: 'Only when the user explicitly forbids retries.' },
        authorizedWriteTargets: { type: 'array', items: { type: 'string' }, description: 'Only user-authorized workspace write targets, never inferred implementation paths.' },
        protectedWriteTargets: { type: 'array', items: { type: 'string' }, description: 'User-protected files or directories that must remain unchanged.' },
        kind: { type: 'string', enum: ['operational-action', 'workspace-change', 'inspection', 'artifact-production', 'conversation'] },
        workspaceMutation: { type: 'string', enum: ['forbidden', 'allowed', 'required'], description: 'Workspace files only. Keep forbidden for a requested application-state mutation that does not edit workspace files.' },
        capabilities: { type: 'array', items: { type: 'string', enum: ['process', 'service', 'terminal', 'browser', 'filesystem-read', 'filesystem-write', 'network', 'application-state'] }, description: 'Capabilities required by the entire user goal. application-state covers explicitly requested durable preferences, task registrations and application/remote record mutations (external-write effect); service means runtime service lifecycle, not application records. Include registration/state mutation even when a future worker only reads. Inspection/conversation must not gain application-state. This classification never replaces authorization or approval.' },
      },
      required: ['mode', 'reason', 'outcomes', 'kind', 'workspaceMutation', 'capabilities'],
    },
  }, {
    name: TOOL_CATALOG_TOOL,
    description: 'Discover authorized tool names, descriptions and effects without loading every schema. Search by a short capability term with query, or select a group and page by offset. Discovery does not execute tools. You may batch catalog requests with independent ordinary tool calls; only agent.transfer must be called alone.',
    inputSchema: { type: 'object', properties: { query: { type: 'string', maxLength: 200, description: 'Case-insensitive literal term in an authorized tool name or description; searches all groups when group is omitted.' }, group: { type: 'string', enum: groups }, offset: { type: 'integer', minimum: 0 } }, additionalProperties: false },
  }] : []
  return {
    state: { mode: options.currentMode, transferCount: options.transferCount ?? 0,
      turnMaxIterations: options.turnMaxIterations ?? 50, visibleToolNames: [...(options.activeToolNames ?? [])] },
    tools,
    prompt: [
      `[Execution mode: ${options.currentMode}]`,
      options.currentMode === 'instant'
        ? 'Answer directly when context suffices; use memory tools for focused recall or persistence. As soon as the complete goal (judged semantically, in any language) needs external actions, research, workspace work or a multi-step workflow, call agent.transfer. Never pretend an unavailable tool ran.'
        : 'Stay in this mode while it can finish the goal. When new evidence needs another mode or a missing tool, discover with agent.tools, then call agent.transfer carrying the goal, constraints and successful observations; never repeat completed side effects.',
      'A mode change authorizes nothing: user constraints, explicit tool limits, approvals and workspace boundaries stay in force, and switching never evades a denial or an exhausted budget. tools:[] selects the mode default; name extra tools only after discovery.',
      ...(options.canTransfer ? [
        'Execution modes:',
        ...options.modes.map((mode) => `${mode.id}: ${mode.description.slice(0, 180)}${mode.capabilities ? ` [capabilities: ${mode.capabilities.join(', ') || 'memory only'}]` : ''}`),
        `Authorized tool catalog: ${catalog.length} tools in groups ${groups.map((group) => `${group} (${catalog.filter((tool) => tool.group === group).length})`).join(', ')}. agent.tools takes a short capability query, or a group with an offset.`,
      ] : ['Execution transfer is unavailable for this bounded turn; answer honestly from the permitted evidence.']),
      ...(options.discoveryContext ? [options.discoveryContext] : []),
    ].join('\n'),
    pending: () => transferred,
    drainEvents: () => events.splice(0),
    handle(response, snapshot) {
      let calls = response.toolCalls ?? []
      if (!calls.some((call) => tools.some((tool) => tool.name === call.name))) return 'unhandled'
      const transferBatch = calls.some((call) => call.name === MODE_TRANSFER_TOOL)
      const normalCalls = transferBatch ? [] : calls.filter((call) => call.name !== TOOL_CATALOG_TOOL)
      if (normalCalls.length > 0) {
        calls = calls.filter((call) => call.name === TOOL_CATALOG_TOOL)
        snapshot.messages.push({ ...structuredClone(response), toolCalls: structuredClone(calls) })
        response.toolCalls = normalCalls
      } else snapshot.messages.push(structuredClone(response))
      let transfer: ModeTransfer | undefined
      for (const call of calls) {
        let output: unknown = { error: 'agent.transfer must be sent alone. No sibling calls were executed.' }
        if (!transferBatch && call.name === TOOL_CATALOG_TOOL) {
          const parsed = z.object({ query: z.string().max(200).optional(), group: z.string().optional(), offset: z.number().int().min(0).optional() }).strict().safeParse(call.arguments)
          if (!parsed.success || (parsed.data.group && !groups.includes(parsed.data.group as typeof groups[number]))) {
            output = { error: 'Supply an available group and a nonnegative integer offset.' }
          } else {
            const offset = parsed.data.offset ?? 0
            const query = parsed.data.query?.trim().toLocaleLowerCase()
            const page = catalog.filter((tool) => (!parsed.data.group || tool.group === parsed.data.group)
              && (!query || `${tool.name} ${tool.description}`.toLocaleLowerCase().includes(query)))
            output = { tools: page.slice(offset, offset + 16), nextOffset: offset + 16 < page.length ? offset + 16 : null,
              activation: 'These are discovery results, not callable schemas. To use a tool that is not already active, call agent.transfer alone with mode:react, tools:[exact names], and the required semantic capabilities. Preserve the original user goal and constraints.' }
          }
        } else if (calls.length === 1 && call.name === MODE_TRANSFER_TOOL) {
          const parsed = transferSchema.safeParse(call.arguments)
          const target = parsed.success ? options.modes.find((mode) => mode.id === parsed.data.mode) : undefined
          if (!parsed.success) output = { error: parsed.error.issues.map((issue) => `${issue.path.join('.')}: ${issue.message}`).join('; ') }
          else if (!target) output = { error: 'Requested mode is unavailable.' }
          else if ([...parsed.data.tools, ...(parsed.data.allowedTools ?? []), ...(parsed.data.toolSequence ?? [])].some((name) => !catalog.some((tool) => tool.name === name))) output = { error: 'Requested tools are outside the authorized catalog.' }
          else if (target.capabilities && parsed.data.capabilities.some((capability) => !target.capabilities!.includes(capability))) output = { error: 'The requested mode cannot satisfy all required capabilities.' }
          else if (target.id !== 'react' && target.id !== 'instant' && parsed.data.tools.length > 0) output = { error: 'Additional tools are activated in react. Choose the graph default with tools:[] or select react with exact tools.' }
          else if (target.id === 'instant' && parsed.data.tools.some((name) => !(INSTANT_MEMORY_TOOL_NAMES as readonly string[]).includes(name))) output = { error: 'Instant exposes focused memory tools only; select a tool-capable mode.' }
          else if (target.id === options.currentMode
            && parsed.data.tools.every((name) => options.activeToolNames?.includes(name))
            && parsed.data.capabilities.every((capability) => options.context?.runContract?.executionIntent?.capabilities.includes(capability))) output = { error: 'Already in that mode with these capabilities; continue here. Existing user constraints and approvals remain authoritative.' }
          else if (target.id === options.currentMode && parsed.data.tools.length === 0 && parsed.data.capabilities.length === 0) output = { error: 'Already in that mode; request an additional authorized capability or continue here.' }
          else {
            transfer = {
              mode: target.id, reason: parsed.data.reason, tools: parsed.data.tools,
              outcomes: parsed.data.outcomes, constraints: parsed.data.constraints ?? [],
              executionIntent: {
                kind: parsed.data.kind, workspaceMutation: parsed.data.workspaceMutation,
                capabilities: parsed.data.capabilities, capabilityPolicy: 'closed',
                ...(parsed.data.allowedTools !== undefined ? { allowedTools: parsed.data.allowedTools } : {}),
                ...(parsed.data.toolSequence ? { toolSequence: parsed.data.toolSequence } : {}),
                ...(parsed.data.retryPolicy ? { retryPolicy: parsed.data.retryPolicy } : {}),
                ...(parsed.data.authorizedWriteTargets ? { authorizedWriteTargets: parsed.data.authorizedWriteTargets } : {}),
                ...(parsed.data.protectedWriteTargets ? { protectedWriteTargets: parsed.data.protectedWriteTargets } : {}),
              },
            }
            output = { transferred: true, mode: target.id, reason: parsed.data.reason }
          }
        }
        const status = output && typeof output === 'object' && 'error' in output ? 'error' : 'success'
        const content = JSON.stringify(output)
        events.push({ type: 'tool_call', toolCall: structuredClone(call) },
          { type: 'tool_result', toolCallId: call.id, status, output: content })
        snapshot.messages.push({ role: 'tool', name: call.name, toolCallId: call.id, content,
          metadata: { toolResultStatus: status } })
      }
      if (!transfer) return normalCalls.length > 0 ? 'unhandled' : 'continue'
      transferred = true
      options.onTransfer(transfer, snapshot)
      return 'transfer'
    },
  }
}
