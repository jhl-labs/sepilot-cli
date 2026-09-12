// 글쓰기 모드 (canvas) 전용 LLM tools. fs.* 또는 apply_patch와 달리 daemon-owned
// in-memory doc session을 직접 다룬다 → desktop renderer가 WS event로 즉시 본다.
//
// activeSessionResolver:
//   채팅 시점에 어떤 session이 활성인지 알려준다. 일반적으로 desktop이
//   chat-stream payload에 `writingDocSessionId`를 첨부 → daemon이 그 값을
//   AgentContext에 옮겨놓고, 이 resolver가 그 값을 꺼낸다. resolver가 null을
//   반환하면 모든 doc.* tool은 "활성 글쓰기 doc 없음" 에러를 낸다.

import type { DocOutlineEntry } from '@sepilotd/core'
import type { ToolDefinitionRuntime, ToolResult } from '../../tools/registry.js'
import { strictWorkspacePathViolation } from '../../security/policy-engine.js'
import type { DocRegistry } from './session.js'
import { AmbiguousSectionError, parseOutline } from './parse.js'

export type ActiveDocResolver = (context: {
  sessionId: string
  writingDocId?: string
}) => string | null

interface DocToolDeps {
  registry: DocRegistry
  resolve: ActiveDocResolver
}

function activeId(
  deps: DocToolDeps,
  context: { sessionId: string; writingDocId?: string } | undefined,
): string | null {
  if (!context) return null
  return deps.resolve(context)
}

function noActiveDocResult(): ToolResult {
  return {
    output:
      '활성 글쓰기 문서가 없습니다. desktop의 글쓰기 모드에서 파일을 열어야 doc.* 도구를 사용할 수 있습니다.',
    status: 'error',
    durationMs: 0,
  }
}

function activeDocWorkspaceBoundaryResult(
  registry: DocRegistry,
  id: string,
  context: { workspaceRoot?: string } | undefined,
  start: number,
): ToolResult | null {
  const path = registry.get(id)?.path
  if (!path || !context?.workspaceRoot) return null
  const violation = strictWorkspacePathViolation(path, context.workspaceRoot)
  if (!violation) return null
  return {
    output: violation,
    status: 'error',
    code: 'WORKSPACE_BOUNDARY_PERMANENT',
    durationMs: Date.now() - start,
  }
}

function summarizeUpdate(version: number, changes: number, sumNewLen: number): string {
  return `doc updated (v${version}, ${changes} change${changes === 1 ? '' : 's'}, ${sumNewLen} chars written). desktop editor가 즉시 반영합니다.`
}

function outlineEntryLine(entry: DocOutlineEntry): string {
  return `[${entry.index}] ${'#'.repeat(entry.level)} ${entry.title} (chars ${entry.start}..${entry.end})`
}

function ambiguousSectionResult(error: AmbiguousSectionError, start: number): ToolResult {
  return {
    output:
      `ambiguous section ${JSON.stringify(error.selector)}. outline index로 다시 지정하세요:\n`
      + error.candidates.map(outlineEntryLine).join('\n'),
    status: 'error',
    durationMs: Date.now() - start,
  }
}

function findSectionForTool(
  registry: DocRegistry,
  id: string,
  selector: string | number,
  start: number,
): { entry: DocOutlineEntry | null; result?: ToolResult } {
  try {
    return { entry: registry.findSection(id, selector) }
  } catch (error) {
    if (error instanceof AmbiguousSectionError) {
      return { entry: null, result: ambiguousSectionResult(error, start) }
    }
    throw error
  }
}

function readExpectedVersion(value: unknown): number | undefined | null {
  if (value == null) return undefined
  const version = Number(value)
  if (!Number.isInteger(version) || version < 1) return null
  return version
}

function invalidExpectedVersionResult(start: number): ToolResult {
  return {
    output: 'invalid expectedVersion: must be a positive integer doc.version',
    status: 'error',
    durationMs: Date.now() - start,
  }
}

function docVersionMismatchResult(
  expectedVersion: number,
  currentVersion: number,
  start: number,
): ToolResult {
  return {
    output: `doc.version mismatch: expected ${expectedVersion}, current ${currentVersion}`,
    status: 'error',
    durationMs: Date.now() - start,
  }
}

function recoverableDocMutationError(error: unknown, start: number): ToolResult | null {
  if (error instanceof Error && error.message.startsWith('doc.version mismatch:')) {
    return {
      output: error.message,
      status: 'error',
      durationMs: Date.now() - start,
    }
  }
  return null
}

export function createDocTools(deps: DocToolDeps): ToolDefinitionRuntime[] {
  const { registry } = deps

  const docGet: ToolDefinitionRuntime = {
    name: 'doc.get',
    description:
      '활성 글쓰기 문서의 내용을 반환. `section` 인자(string=heading 텍스트, number=outline index)를 주면 해당 섹션만, 없으면 전체. 큰 문서면 먼저 doc.outline으로 outline을 본 뒤 필요한 섹션만 doc.get 하는 것을 권장.',
    scheduling: { mode: 'parallel-safe', resource: 'doc' },
    inputSchema: {
      type: 'object',
      properties: {
        section: { type: ['string', 'number'], description: 'heading 텍스트 또는 outline index (0-based)' },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const id = activeId(deps, context && {
        sessionId: context.sessionId,
        writingDocId: context.writingDocId,
      })
      if (!id) return noActiveDocResult()
      const workspaceBoundary = activeDocWorkspaceBoundaryResult(registry, id, context, start)
      if (workspaceBoundary) return workspaceBoundary
      const session = registry.get(id)
      if (!session) return noActiveDocResult()
      if (input.section == null) {
        return {
          output: session.content,
          status: 'success',
          durationMs: Date.now() - start,
        }
      }
      const sel = input.section as string | number
      const found = findSectionForTool(registry, id, sel, start)
      if (found.result) return found.result
      const entry = found.entry
      if (!entry) {
        return {
          output: `section not found: ${JSON.stringify(sel)}. doc.outline으로 사용 가능 섹션을 확인하세요.`,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      return {
        output: session.content.slice(entry.start, entry.end),
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }

  const docOutline: ToolDefinitionRuntime = {
    name: 'doc.outline',
    description: '활성 글쓰기 문서의 heading outline을 반환 (level/title/index/start/end). 큰 문서를 부분 편집하기 전 호출.',
    scheduling: { mode: 'parallel-safe', resource: 'doc' },
    inputSchema: { type: 'object', properties: {} },
    async execute(_input, context): Promise<ToolResult> {
      const start = Date.now()
      const id = activeId(deps, context && {
        sessionId: context.sessionId,
        writingDocId: context.writingDocId,
      })
      if (!id) return noActiveDocResult()
      const workspaceBoundary = activeDocWorkspaceBoundaryResult(registry, id, context, start)
      if (workspaceBoundary) return workspaceBoundary
      const outline = registry.outline(id)
      const text =
        outline.length === 0
          ? '(no headings)'
          : outline
              .map((e) => `[${e.index}] ${'#'.repeat(e.level)} ${e.title} (chars ${e.start}..${e.end})`)
              .join('\n')
      return { output: text, status: 'success', durationMs: Date.now() - start }
    },
  }

  const docReplaceSection: ToolDefinitionRuntime = {
    name: 'doc.replace_section',
    description: '특정 heading 섹션을 통째 교체. section은 heading 텍스트(완전일치 우선, case-insensitive fallback) 또는 outline index. text에는 새 heading 라인부터 포함시킬 것 (예: "## 새 제목\\n새 본문...").',
    scheduling: { mode: 'sequential', resource: 'doc' },
    inputSchema: {
      type: 'object',
      properties: {
        section: { type: ['string', 'number'] },
        text: { type: 'string' },
      },
      required: ['section', 'text'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const id = activeId(deps, context && {
        sessionId: context.sessionId,
        writingDocId: context.writingDocId,
      })
      if (!id) return noActiveDocResult()
      const workspaceBoundary = activeDocWorkspaceBoundaryResult(registry, id, context, start)
      if (workspaceBoundary) return workspaceBoundary
      const found = findSectionForTool(
        registry,
        id,
        input.section as string | number,
        start,
      )
      if (found.result) return found.result
      const entry = found.entry
      if (!entry) {
        return {
          output: `section not found: ${JSON.stringify(input.section)}. doc.outline 먼저.`,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      const text = String(input.text ?? '')
      const { version } = registry.apply(
        id,
        [{ start: entry.start, end: entry.end, newText: text }],
        'llm',
        { toolName: 'doc.replace_section', label: `섹션 "${entry.title}" 교체` },
      )
      return {
        output: summarizeUpdate(version, 1, text.length),
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }

  const docReplaceRange: ToolDefinitionRuntime = {
    name: 'doc.replace_range',
    description: 'char offset 기반 정확한 range 교체. doc.outline의 start/end 또는 doc.get으로 확인한 위치를 사용. 사용자 selection을 chat에 보낸 컨텍스트의 [start, end]도 그대로 사용 가능.',
    scheduling: { mode: 'sequential', resource: 'doc' },
    inputSchema: {
      type: 'object',
      properties: {
        start: { type: 'number' },
        end: { type: 'number' },
        text: { type: 'string' },
        expectedVersion: {
          type: 'number',
          description: 'doc.get/doc.outline에서 본 현재 doc.version. mismatch면 stale offset 적용을 거부.',
        },
      },
      required: ['start', 'end', 'text'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const id = activeId(deps, context && {
        sessionId: context.sessionId,
        writingDocId: context.writingDocId,
      })
      if (!id) return noActiveDocResult()
      const workspaceBoundary = activeDocWorkspaceBoundaryResult(registry, id, context, start)
      if (workspaceBoundary) return workspaceBoundary
      const s = Number(input.start ?? -1)
      const e = Number(input.end ?? -1)
      const text = String(input.text ?? '')
      const expectedVersion = readExpectedVersion(input.expectedVersion)
      if (expectedVersion === null) return invalidExpectedVersionResult(start)
      if (!Number.isFinite(s) || !Number.isFinite(e) || s < 0 || e < s) {
        return {
          output: 'invalid range: start/end must be non-negative integers with start<=end',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      const session = registry.get(id)!
      if (e > session.content.length) {
        return {
          output: `end ${e} exceeds doc length ${session.content.length}. doc.outline로 최신 offset 확인 후 다시.`,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      let version: number
      try {
        const result = registry.apply(
          id,
          [{ start: s, end: e, newText: text }],
          'llm',
          {
            toolName: 'doc.replace_range',
            label: `range ${s}..${e} 교체`,
            expectedVersion,
          },
        )
        version = result.version
      } catch (error) {
        const result = recoverableDocMutationError(error, start)
        if (result) return result
        throw error
      }
      return {
        output: summarizeUpdate(version, 1, text.length),
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }

  const docInsertAfterSection: ToolDefinitionRuntime = {
    name: 'doc.insert_after_section',
    description: '지정 섹션 *바로 뒤*에 새 내용을 삽입. text는 새 줄 보장(앞에 \\n 추가됨).',
    scheduling: { mode: 'sequential', resource: 'doc' },
    inputSchema: {
      type: 'object',
      properties: {
        section: { type: ['string', 'number'] },
        text: { type: 'string' },
      },
      required: ['section', 'text'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const id = activeId(deps, context && {
        sessionId: context.sessionId,
        writingDocId: context.writingDocId,
      })
      if (!id) return noActiveDocResult()
      const workspaceBoundary = activeDocWorkspaceBoundaryResult(registry, id, context, start)
      if (workspaceBoundary) return workspaceBoundary
      const found = findSectionForTool(
        registry,
        id,
        input.section as string | number,
        start,
      )
      if (found.result) return found.result
      const entry = found.entry
      if (!entry) {
        return {
          output: `section not found: ${JSON.stringify(input.section)}`,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      const text = String(input.text ?? '')
      const prefix = text.startsWith('\n') ? '' : '\n'
      const inserted = prefix + text + (text.endsWith('\n') ? '' : '\n')
      const { version } = registry.apply(
        id,
        [{ start: entry.end, end: entry.end, newText: inserted }],
        'llm',
        { toolName: 'doc.insert_after_section', label: `섹션 "${entry.title}" 뒤에 추가` },
      )
      return {
        output: summarizeUpdate(version, 1, inserted.length),
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }

  const docAppend: ToolDefinitionRuntime = {
    name: 'doc.append',
    description: '문서 맨 끝에 추가. 앞에 자동으로 빈 줄 보장.',
    scheduling: { mode: 'sequential', resource: 'doc' },
    inputSchema: {
      type: 'object',
      properties: { text: { type: 'string' } },
      required: ['text'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const id = activeId(deps, context && {
        sessionId: context.sessionId,
        writingDocId: context.writingDocId,
      })
      if (!id) return noActiveDocResult()
      const workspaceBoundary = activeDocWorkspaceBoundaryResult(registry, id, context, start)
      if (workspaceBoundary) return workspaceBoundary
      const session = registry.get(id)!
      const text = String(input.text ?? '')
      const prefix = session.content.length > 0 && !session.content.endsWith('\n') ? '\n\n' : '\n'
      const inserted = prefix + text
      const { version } = registry.apply(
        id,
        [{ start: session.content.length, end: session.content.length, newText: inserted }],
        'llm',
        { toolName: 'doc.append', label: '문서 끝에 추가' },
      )
      return {
        output: summarizeUpdate(version, 1, inserted.length),
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }

  const docRewrite: ToolDefinitionRuntime = {
    name: 'doc.rewrite',
    description: '문서 전체를 새 본문으로 교체. 큰 퇴고나 새로운 outline으로 통째 다시 쓸 때만. 부분 편집이면 doc.replace_section 권장.',
    scheduling: { mode: 'sequential', resource: 'doc' },
    inputSchema: {
      type: 'object',
      properties: { text: { type: 'string' } },
      required: ['text'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const id = activeId(deps, context && {
        sessionId: context.sessionId,
        writingDocId: context.writingDocId,
      })
      if (!id) return noActiveDocResult()
      const workspaceBoundary = activeDocWorkspaceBoundaryResult(registry, id, context, start)
      if (workspaceBoundary) return workspaceBoundary
      const session = registry.get(id)!
      const text = String(input.text ?? '')
      const { version } = registry.apply(
        id,
        [{ start: 0, end: session.content.length, newText: text }],
        'llm',
        { toolName: 'doc.rewrite', label: '전체 rewrite' },
      )
      return {
        output: summarizeUpdate(version, 1, text.length),
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }

  const docDiffPreview: ToolDefinitionRuntime = {
    name: 'doc.diff_preview',
    description: '큰 변경을 즉시 적용하지 않고 미리보기로 만들어 사용자에게 수락/취소 받기. inline diff가 우측 editor에 보여진다. changes는 ascending 또는 descending 무관 (registry가 정렬). summary는 한 줄짜리 설명.',
    scheduling: { mode: 'sequential', resource: 'doc' },
    inputSchema: {
      type: 'object',
      properties: {
        summary: { type: 'string' },
        expectedVersion: {
          type: 'number',
          description: 'doc.get/doc.outline에서 본 현재 doc.version. mismatch면 stale preview 생성을 거부.',
        },
        changes: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              start: { type: 'number' },
              end: { type: 'number' },
              newText: { type: 'string' },
            },
            required: ['start', 'end', 'newText'],
          },
        },
      },
      required: ['summary', 'changes'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const id = activeId(deps, context && {
        sessionId: context.sessionId,
        writingDocId: context.writingDocId,
      })
      if (!id) return noActiveDocResult()
      const workspaceBoundary = activeDocWorkspaceBoundaryResult(registry, id, context, start)
      if (workspaceBoundary) return workspaceBoundary
      const expectedVersion = readExpectedVersion(input.expectedVersion)
      if (expectedVersion === null) return invalidExpectedVersionResult(start)
      const session = registry.get(id)
      if (!session) return noActiveDocResult()
      if (expectedVersion !== undefined && expectedVersion !== session.version) {
        return docVersionMismatchResult(expectedVersion, session.version, start)
      }
      const summary = String(input.summary ?? 'pending change')
      const raw = Array.isArray(input.changes) ? input.changes : []
      const changes = raw
        .map((c: unknown) => {
          if (typeof c !== 'object' || c === null) return null
          const o = c as { start?: unknown; end?: unknown; newText?: unknown }
          const s = Number(o.start ?? NaN)
          const e = Number(o.end ?? NaN)
          if (!Number.isFinite(s) || !Number.isFinite(e) || s < 0 || e < s) return null
          return { start: s, end: e, newText: String(o.newText ?? '') }
        })
        .filter((c): c is { start: number; end: number; newText: string } => c !== null)
      if (changes.length === 0) {
        return {
          output: 'no valid changes; preview not created',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      const preview = registry.createPreview(id, changes, summary)
      return {
        output:
          `diff preview 생성됨 (id=${preview.id}, ${changes.length} change${changes.length === 1 ? '' : 's'}). 사용자가 수락하기 전까지 doc은 변경되지 않습니다. 사용자에게 수락/취소를 안내하세요.`,
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }

  // outline cache invalidation은 매번 parseOutline 호출이라 불필요 (작은 doc 가정).
  // 큰 doc은 향후 line-index cache 도입 검토.
  void parseOutline

  return [
    docGet,
    docOutline,
    docReplaceSection,
    docReplaceRange,
    docInsertAfterSection,
    docAppend,
    docRewrite,
    docDiffPreview,
  ]
}
