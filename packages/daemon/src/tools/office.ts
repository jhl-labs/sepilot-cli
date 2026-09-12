import { spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { mkdir, readFile, realpath, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { extname, isAbsolute, join, relative } from 'node:path'
import { resolveToolPath } from './path-utils.js'
import type { ToolDefinitionRuntime, ToolExecutionContext, ToolResult } from './registry.js'

type OfficeApp = 'word' | 'excel' | 'powerpoint'
type TextReplaceOccurrence = 'first' | 'all'
type OfficeEditOperation = 'replace_text' | 'set_range_values' | 'replace_selection'

export interface OfficeDocumentRef {
  app: OfficeApp
  name: string
  path?: string | null
  saved?: boolean
  readOnly?: boolean
  active?: boolean
}

export interface OfficeReadSnapshot {
  app: OfficeApp
  name: string
  path?: string | null
  saved?: boolean
  kind: 'word' | 'excel' | 'powerpoint'
  text?: string
  truncated?: boolean
  activeSheet?: string
  usedRange?: string
  rows?: string[][]
  slides?: Array<{ index: number; text: string }>
}

export interface OfficeApplyResult {
  app: OfficeApp
  name: string
  path?: string | null
  operation: OfficeEditOperation
  changed: number
  saved: boolean
  target?: Record<string, unknown>
}

export interface OfficeSelectionSnapshot {
  app: OfficeApp
  name: string
  path?: string | null
  saved?: boolean
  kind: 'word' | 'excel' | 'powerpoint'
  text?: string
  activeSheet?: string
  range?: string
  rows?: string[][]
  slideIndex?: number | null
  shapeCount?: number
  selectionType?: string
}

export interface OfficeBridge {
  listOpenDocuments(): Promise<{ documents: OfficeDocumentRef[] }>
  readActive(input: {
    app: OfficeApp
    maxChars: number
    maxRows: number
    maxColumns: number
  }): Promise<OfficeReadSnapshot>
  readSelection(input: {
    app: OfficeApp
    maxChars: number
    maxRows: number
    maxColumns: number
  }): Promise<OfficeSelectionSnapshot>
  applyEdit(input: {
    app: OfficeApp
    operation: OfficeEditOperation
    findText?: string
    replacement?: string
    occurrence?: TextReplaceOccurrence
    sheet?: string
    range?: string
    values?: unknown
    save?: boolean
    expectedPath?: string
  }): Promise<OfficeApplyResult>
}

export interface PowerPointPresentationSnapshot {
  app: 'powerpoint'
  kind: 'powerpoint'
  name: string
  path: string
  readOnly: boolean
  slideCount: number
  slideIndex: number
}

export interface PowerPointShapeMetadata {
  index: number
  name: string
  shapeType: number
  left: number
  top: number
  width: number
  height: number
  hasText: boolean
  text?: string
  alternativeText?: string
  hasChart?: boolean
  hasTable?: boolean
}

export interface PowerPointSlideSnapshot extends PowerPointPresentationSnapshot {
  activeSlideIndex: number
  title?: string
  text: string
  notes: string
  shapes: PowerPointShapeMetadata[]
  truncated: boolean
}

export interface PowerPointSlideCaptureSnapshot extends PowerPointPresentationSnapshot {
  activeSlideIndex: number
  width: number
  height: number
}

export type PowerPointSlideNavigation =
  | { direction: 'next' | 'previous' }
  | { direction: 'index'; index: number }

/**
 * Read-only PowerPoint review capability. It is separate from OfficeBridge so
 * consumers that only need the established document/edit contract do not have
 * to implement presentation window state or slide inspection.
 */
export interface PowerPointReviewBridge {
  openReadOnly(input: { path: string }): Promise<PowerPointPresentationSnapshot>
  navigateSlide(
    input: PowerPointSlideNavigation & { expectedPath: string },
  ): Promise<PowerPointPresentationSnapshot>
  readSlide(input: {
    expectedPath: string
    index?: number
    maxChars: number
  }): Promise<PowerPointSlideSnapshot>
}

/** Optional visual renderer kept separate from the structured review bridge. */
export interface PowerPointSlideCaptureBridge {
  captureSlide(input: {
    expectedPath: string
    outputPath: string
    index?: number
    width: number
  }): Promise<PowerPointSlideCaptureSnapshot>
}

export interface OfficeToolsDeps {
  bridge?: OfficeBridge
  presentationBridge?: PowerPointReviewBridge
  presentationCaptureBridge?: PowerPointSlideCaptureBridge
}

function isOfficeApp(value: unknown): value is OfficeApp {
  return value === 'word' || value === 'excel' || value === 'powerpoint'
}

function normalizeApp(input: Record<string, unknown>): OfficeApp | null {
  return isOfficeApp(input.app) ? input.app : null
}

function boolInput(input: Record<string, unknown>, name: string): boolean {
  return input[name] === true
}

function intInput(
  input: Record<string, unknown>,
  name: string,
  fallback: number,
  min: number,
  max: number,
): number {
  const raw = input[name]
  return typeof raw === 'number' && Number.isFinite(raw)
    ? Math.max(min, Math.min(max, Math.trunc(raw)))
    : fallback
}

function workspaceContains(path: string, cwd: string): boolean {
  const rel = relative(resolveToolPath(cwd), resolveToolPath(path))
  return rel === '' || (!!rel && !rel.startsWith('..') && !isAbsolute(rel))
}

type WorkspacePresentationResolution =
  | { ok: true; path: string; workspaceRoot: string }
  | { ok: false; error: ToolResult }

function presentationPathError(
  code: string,
  output: string,
): WorkspacePresentationResolution {
  return {
    ok: false,
    error: { status: 'error', code, output, durationMs: 0 },
  }
}

function pathIsContained(path: string, root: string): boolean {
  const rel = relative(root, path)
  return rel === '' || (!!rel && !rel.startsWith('..') && !isAbsolute(rel))
}

async function resolveWorkspacePresentationPath(
  rawPath: unknown,
  context: ToolExecutionContext | undefined,
): Promise<WorkspacePresentationResolution> {
  if (typeof rawPath !== 'string' || !rawPath.trim()) {
    return presentationPathError(
      'INVALID_INPUT_PERMANENT',
      'path must be a non-empty workspace-relative or absolute .pptx path.',
    )
  }

  const workspaceInput = context?.workspaceRoot?.trim() || context?.cwd?.trim()
  if (!workspaceInput) {
    return presentationPathError(
      'OFFICE_WORKSPACE_REQUIRED_PERMANENT',
      'PowerPoint review requires a selected workspace so the daemon can enforce the file boundary.',
    )
  }

  try {
    const lexicalWorkspaceRoot = resolveToolPath(workspaceInput)
    const workspaceRoot = await realpath(lexicalWorkspaceRoot)
    const candidate = resolveToolPath(rawPath.trim(), context?.cwd ?? lexicalWorkspaceRoot)

    // Check before and after realpath. The second check rejects a symlink inside
    // the workspace that resolves to a file outside it.
    if (!pathIsContained(candidate, lexicalWorkspaceRoot)) {
      return presentationPathError(
        'OFFICE_OUTSIDE_WORKSPACE_USER',
        `Presentation is outside the selected workspace: ${candidate}`,
      )
    }

    if (extname(candidate).toLowerCase() !== '.pptx') {
      return presentationPathError(
        'OFFICE_UNSUPPORTED_PRESENTATION_PERMANENT',
        'PowerPoint review accepts .pptx files only. Macro-enabled or legacy presentation formats are not opened.',
      )
    }

    const path = await realpath(candidate)
    if (!pathIsContained(path, workspaceRoot)) {
      return presentationPathError(
        'OFFICE_OUTSIDE_WORKSPACE_USER',
        `Presentation resolves outside the selected workspace: ${candidate}`,
      )
    }
    if (!(await stat(path)).isFile()) {
      return presentationPathError(
        'OFFICE_PRESENTATION_NOT_FOUND_PERMANENT',
        `Presentation is not a file: ${candidate}`,
      )
    }
    return { ok: true, path, workspaceRoot }
  } catch (error) {
    const code = error && typeof error === 'object' && 'code' in error
      ? String(error.code)
      : ''
    if (code === 'ENOENT' || code === 'ENOTDIR') {
      return presentationPathError(
        'OFFICE_PRESENTATION_NOT_FOUND_PERMANENT',
        `Presentation was not found: ${rawPath.trim()}`,
      )
    }
    return presentationPathError(
      'OFFICE_PRESENTATION_PATH_INVALID_PERMANENT',
      error instanceof Error ? error.message : String(error),
    )
  }
}

function isPowerPointReviewBridge(value: unknown): value is PowerPointReviewBridge {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Partial<PowerPointReviewBridge>
  return typeof candidate.openReadOnly === 'function'
    && typeof candidate.navigateSlide === 'function'
    && typeof candidate.readSlide === 'function'
}

function isPowerPointSlideCaptureBridge(value: unknown): value is PowerPointSlideCaptureBridge {
  if (!value || typeof value !== 'object') return false
  return typeof (value as Partial<PowerPointSlideCaptureBridge>).captureSlide === 'function'
}

function presentationBridgeError(error: unknown, start: number): ToolResult {
  const message = error instanceof Error ? error.message : String(error)
  const tagged = message.match(/\[(OFFICE_[A-Z0-9_]+)\]/)?.[1]
  const objectCode = error && typeof error === 'object' && 'code' in error
    ? String(error.code)
    : undefined
  return {
    status: 'error',
    code: tagged ?? (objectCode?.startsWith('OFFICE_') ? objectCode : 'OFFICE_BRIDGE_UNAVAILABLE_PERMANENT'),
    output: message,
    durationMs: Date.now() - start,
  }
}

async function validatePresentationSnapshot(
  snapshot: PowerPointPresentationSnapshot,
  context: ToolExecutionContext | undefined,
  expectedPath: string,
): Promise<ToolResult | null> {
  if (snapshot.readOnly !== true) {
    return {
      status: 'error',
      code: 'OFFICE_PRESENTATION_WRITABLE_USER',
      output:
        'PowerPoint review requires a read-only presentation. Close the writable copy and reopen it through office.open_presentation.',
      durationMs: 0,
    }
  }
  if (!Number.isInteger(snapshot.slideCount) || snapshot.slideCount < 1) {
    return {
      status: 'error',
      code: 'OFFICE_PRESENTATION_INVALID_PERMANENT',
      output: 'PowerPoint returned an invalid slide count.',
      durationMs: 0,
    }
  }
  if (
    !Number.isInteger(snapshot.slideIndex)
    || snapshot.slideIndex < 1
    || snapshot.slideIndex > snapshot.slideCount
  ) {
    return {
      status: 'error',
      code: 'OFFICE_SLIDE_RANGE_PERMANENT',
      output: `PowerPoint returned slide ${snapshot.slideIndex}; expected a slide in 1..${snapshot.slideCount}.`,
      durationMs: 0,
    }
  }

  const activePath = await resolveWorkspacePresentationPath(snapshot.path, context)
  if (!activePath.ok) return activePath.error
  if (relative(expectedPath, activePath.path) !== '') {
    return {
      status: 'error',
      code: 'OFFICE_ACTIVE_PRESENTATION_CHANGED_USER',
      output:
        `The active PowerPoint presentation changed during review. Expected ${expectedPath}, found ${activePath.path}.`,
      durationMs: 0,
    }
  }
  return null
}

function enforceDocumentScope(
  snapshot: { path?: string | null; name: string },
  input: Record<string, unknown>,
  context: ToolExecutionContext | undefined,
): ToolResult | null {
  const cwd = context?.cwd
  const allowOutsideWorkspace = boolInput(input, 'allowOutsideWorkspace')
  const allowUnsaved = boolInput(input, 'allowUnsaved')
  const path = typeof snapshot.path === 'string' && snapshot.path.trim()
    ? snapshot.path.trim()
    : ''

  if (!path) {
    if (allowUnsaved) return null
    return {
      status: 'error',
      code: 'OFFICE_UNSAVED_DOCUMENT_USER',
      output:
        `Active Office document "${snapshot.name}" is unsaved. ` +
        'Save it inside the workspace first, or pass allowUnsaved=true after explicit user approval.',
      durationMs: 0,
    }
  }

  if (!cwd && !allowOutsideWorkspace) {
    return {
      status: 'error',
      code: 'OFFICE_WORKSPACE_REQUIRED_PERMANENT',
      output:
        'Office live document access needs a session cwd so the daemon can enforce the workspace boundary. ' +
        'Pass cwd, or pass allowOutsideWorkspace=true after explicit user approval.',
      durationMs: 0,
    }
  }

  if (cwd && !workspaceContains(path, cwd) && !allowOutsideWorkspace) {
    return {
      status: 'error',
      code: 'OFFICE_OUTSIDE_WORKSPACE_USER',
      output:
        `Active Office document is outside the workspace: ${path}. ` +
        `Workspace: ${resolveToolPath(cwd)}. ` +
        'Move/save the document inside the workspace, or pass allowOutsideWorkspace=true after explicit user approval.',
      durationMs: 0,
    }
  }

  return null
}

function withDuration(result: ToolResult, start: number): ToolResult {
  return { ...result, durationMs: Date.now() - start }
}

function countOccurrences(text: string, needle: string): number {
  if (!needle) return 0
  let count = 0
  let offset = 0
  while (true) {
    const index = text.indexOf(needle, offset)
    if (index < 0) return count
    count += 1
    offset = index + needle.length
  }
}

function textFromSnapshot(snapshot: OfficeReadSnapshot): string {
  if (typeof snapshot.text === 'string') return snapshot.text
  if (snapshot.slides?.length) {
    return snapshot.slides.map((slide) => slide.text).join('\n\n')
  }
  if (snapshot.rows?.length) {
    return snapshot.rows.map((row) => row.join('\t')).join('\n')
  }
  return ''
}

function operationFromInput(input: Record<string, unknown>): OfficeEditOperation | null {
  return input.operation === 'replace_text'
    || input.operation === 'set_range_values'
    || input.operation === 'replace_selection'
    ? input.operation
    : null
}

function occurrenceFromInput(input: Record<string, unknown>): TextReplaceOccurrence {
  return input.occurrence === 'all' ? 'all' : 'first'
}

function validateEditInput(
  app: OfficeApp,
  operation: OfficeEditOperation,
  input: Record<string, unknown>,
): ToolResult | null {
  if (operation === 'replace_text') {
    if (app === 'excel') {
      return {
        status: 'error',
        code: 'INVALID_INPUT_PERMANENT',
        output: 'replace_text supports Word and PowerPoint. Use set_range_values for Excel.',
        durationMs: 0,
      }
    }
    if (typeof input.findText !== 'string' || input.findText.length === 0) {
      return {
        status: 'error',
        code: 'INVALID_INPUT_PERMANENT',
        output: 'findText is required for replace_text.',
        durationMs: 0,
      }
    }
    if (typeof input.replacement !== 'string') {
      return {
        status: 'error',
        code: 'INVALID_INPUT_PERMANENT',
        output: 'replacement is required for replace_text.',
        durationMs: 0,
      }
    }
  }

  if (operation === 'set_range_values') {
    if (app !== 'excel') {
      return {
        status: 'error',
        code: 'INVALID_INPUT_PERMANENT',
        output: 'set_range_values supports Excel only.',
        durationMs: 0,
      }
    }
    if (typeof input.range !== 'string' || input.range.trim() === '') {
      return {
        status: 'error',
        code: 'INVALID_INPUT_PERMANENT',
        output: 'range is required for set_range_values, for example "B2:D4".',
        durationMs: 0,
      }
    }
    if (!Object.hasOwn(input, 'values')) {
      return {
        status: 'error',
        code: 'INVALID_INPUT_PERMANENT',
        output: 'values is required for set_range_values.',
        durationMs: 0,
      }
    }
  }

  if (operation === 'replace_selection') {
    if (app === 'excel') {
      if (!Object.hasOwn(input, 'values')) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'values is required for Excel replace_selection.',
          durationMs: 0,
        }
      }
    } else if (typeof input.replacement !== 'string') {
      return {
        status: 'error',
        code: 'INVALID_INPUT_PERMANENT',
        output: 'replacement is required for Word/PowerPoint replace_selection.',
        durationMs: 0,
      }
    }
  }

  return null
}

async function readEditScopeSnapshot(
  bridge: OfficeBridge,
  app: OfficeApp,
  operation: OfficeEditOperation,
  input: Record<string, unknown>,
): Promise<OfficeReadSnapshot | OfficeSelectionSnapshot> {
  if (operation === 'replace_selection') {
    return bridge.readSelection({
      app,
      maxChars: intInput(input, 'maxChars', 30_000, 500, 200_000),
      maxRows: intInput(input, 'maxRows', 120, 1, 500),
      maxColumns: intInput(input, 'maxColumns', 40, 1, 80),
    })
  }
  return bridge.readActive({
    app,
    maxChars: intInput(input, 'maxChars', 30_000, 500, 200_000),
    maxRows: intInput(input, 'maxRows', 120, 1, 500),
    maxColumns: intInput(input, 'maxColumns', 40, 1, 80),
  })
}

function selectionPreview(
  app: OfficeApp,
  snapshot: OfficeSelectionSnapshot,
  input: Record<string, unknown>,
): Record<string, unknown> {
  if (app === 'excel') {
    return {
      app,
      document: { name: snapshot.name, path: snapshot.path ?? null },
      operation: 'replace_selection',
      activeSheet: snapshot.activeSheet,
      range: snapshot.range,
      currentRows: snapshot.rows,
      values: input.values,
    }
  }
  return {
    app,
    document: { name: snapshot.name, path: snapshot.path ?? null },
    operation: 'replace_selection',
    selectionType: snapshot.selectionType,
    slideIndex: snapshot.slideIndex ?? null,
    currentText: snapshot.text ?? '',
    replacement: input.replacement,
  }
}

async function applyConfirmedEdit(
  bridge: OfficeBridge,
  input: Record<string, unknown>,
  context: ToolExecutionContext | undefined,
  start: number,
): Promise<ToolResult> {
  if (input.confirm !== true) {
    return {
      status: 'error',
      code: 'OFFICE_CONFIRMATION_REQUIRED_USER',
      output: 'Office live edits require confirm=true after the user approves an office.preview_edit or office.read_selection result.',
      durationMs: Date.now() - start,
    }
  }

  const app = normalizeApp(input)
  const operation = operationFromInput(input)
  if (!app || !operation) {
    return {
      status: 'error',
      code: 'INVALID_INPUT_PERMANENT',
      output: 'app and operation are required.',
      durationMs: Date.now() - start,
    }
  }
  const invalid = validateEditInput(app, operation, input)
  if (invalid) return withDuration(invalid, start)

  try {
    const snapshot = await readEditScopeSnapshot(bridge, app, operation, input)
    const scopeError = enforceDocumentScope(snapshot, input, context)
    if (scopeError) return withDuration(scopeError, start)
    const expectedPath = snapshot.path?.trim() || undefined
    const result = await bridge.applyEdit({
      app,
      operation,
      findText: typeof input.findText === 'string' ? input.findText : undefined,
      replacement: typeof input.replacement === 'string' ? input.replacement : undefined,
      occurrence: occurrenceFromInput(input),
      sheet: typeof input.sheet === 'string' ? input.sheet : undefined,
      range: typeof input.range === 'string' ? input.range : undefined,
      values: input.values,
      save: input.save === true,
      expectedPath,
    })
    if (expectedPath) {
      await context?.workspaceMutation?.recordWrite(expectedPath)
    }
    return {
      status: 'success',
      output: JSON.stringify(result, null, 2),
      durationMs: Date.now() - start,
    }
  } catch (error) {
    return {
      status: 'error',
      code: 'OFFICE_EDIT_FAILED_PERMANENT',
      output: error instanceof Error ? error.message : String(error),
      durationMs: Date.now() - start,
    }
  }
}

function createBridge(): OfficeBridge & PowerPointReviewBridge & PowerPointSlideCaptureBridge {
  return createPowerShellOfficeBridge()
}

export function createOfficeTools(deps: OfficeToolsDeps = {}): ToolDefinitionRuntime[] {
  const bridge = deps.bridge ?? createBridge()
  const presentationBridge = deps.presentationBridge
    ?? (isPowerPointReviewBridge(bridge) ? bridge : null)
  const presentationCaptureBridge = deps.presentationCaptureBridge
    ?? (isPowerPointSlideCaptureBridge(bridge) ? bridge : null)
  const lastPresentationReadBySession = new Map<string, {
    path: string
    slideIndex: number
  }>()

  const clearPresentationReadCursor = (context: ToolExecutionContext | undefined): void => {
    if (context?.sessionId) lastPresentationReadBySession.delete(context.sessionId)
  }

  return [
    {
      name: 'office.list_open_documents',
      description:
        'List open Microsoft Word, Excel, and PowerPoint documents from the local desktop Office apps. ' +
        'Returns app, name, path, saved/read-only state, and active marker. Does not modify files or app state.',
      resumeSafety: 'replay-safe',
      scheduling: { mode: 'sequential', resource: 'office.com' },
      inputSchema: {
        type: 'object',
        properties: {},
      },
      async execute(): Promise<ToolResult> {
        const start = Date.now()
        try {
          const result = await bridge.listOpenDocuments()
          return {
            status: 'success',
            output: JSON.stringify(result, null, 2),
            durationMs: Date.now() - start,
          }
        } catch (error) {
          return {
            status: 'error',
            code: 'OFFICE_BRIDGE_UNAVAILABLE_PERMANENT',
            output: error instanceof Error ? error.message : String(error),
            durationMs: Date.now() - start,
          }
        }
      },
    },
    {
      name: 'office.open_presentation',
      description:
        'Open a .pptx file from the selected workspace in desktop Microsoft PowerPoint with ReadOnly=true. ' +
        'Returns the canonical path, slide count, and current slide. Rejects files outside the workspace, symlink escapes, writable copies, and non-.pptx formats.',
      resumeSafety: 'replay-risky',
      scheduling: { mode: 'sequential', resource: 'office.com' },
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Workspace-relative or absolute path to a .pptx file inside the selected workspace.' },
        },
        required: ['path'],
      },
      async execute(input, context): Promise<ToolResult> {
        const start = Date.now()
        const resolved = await resolveWorkspacePresentationPath(input.path, context)
        if (!resolved.ok) return withDuration(resolved.error, start)
        if (!presentationBridge) {
          return {
            status: 'error',
            code: 'OFFICE_BRIDGE_UNAVAILABLE_PERMANENT',
            output: 'The configured Office bridge does not support PowerPoint slide review.',
            durationMs: Date.now() - start,
          }
        }

        try {
          const snapshot = await presentationBridge.openReadOnly({ path: resolved.path })
          const invalid = await validatePresentationSnapshot(snapshot, context, resolved.path)
          if (invalid) return withDuration(invalid, start)
          clearPresentationReadCursor(context)
          await context?.workspaceMutation?.recordRead(resolved.path)
          return {
            status: 'success',
            output: JSON.stringify(snapshot, null, 2),
            durationMs: Date.now() - start,
          }
        } catch (error) {
          return presentationBridgeError(error, start)
        }
      },
    },
    {
      name: 'office.navigate_slide',
      description:
        'Move the active read-only workspace PowerPoint deck by exactly one slide (next/previous) or to one exact 1-based slide index. ' +
        'Requires the expected presentation path and rejects boundary overflow instead of wrapping.',
      resumeSafety: 'replay-risky',
      scheduling: { mode: 'sequential', resource: 'office.com' },
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'The same workspace presentation path passed to office.open_presentation.' },
          direction: { type: 'string', enum: ['next', 'previous', 'index'] },
          index: { type: 'number', description: 'Required only for direction=index. Uses a 1-based slide index.' },
        },
        required: ['path', 'direction'],
      },
      async execute(input, context): Promise<ToolResult> {
        const start = Date.now()
        const resolved = await resolveWorkspacePresentationPath(input.path, context)
        if (!resolved.ok) return withDuration(resolved.error, start)
        const direction = input.direction
        if (direction !== 'next' && direction !== 'previous' && direction !== 'index') {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: 'direction must be one of: next, previous, index.',
            durationMs: Date.now() - start,
          }
        }
        const index = input.index
        if (direction === 'index' && (!Number.isInteger(index) || Number(index) < 1)) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: 'direction=index requires a positive integer index.',
            durationMs: Date.now() - start,
          }
        }
        if (!presentationBridge) {
          return {
            status: 'error',
            code: 'OFFICE_BRIDGE_UNAVAILABLE_PERMANENT',
            output: 'The configured Office bridge does not support PowerPoint slide review.',
            durationMs: Date.now() - start,
          }
        }

        try {
          const navigation: PowerPointSlideNavigation & { expectedPath: string } = direction === 'index'
            ? { direction, index: Number(index), expectedPath: resolved.path }
            : { direction, expectedPath: resolved.path }
          const snapshot = await presentationBridge.navigateSlide(navigation)
          const invalid = await validatePresentationSnapshot(snapshot, context, resolved.path)
          if (invalid) return withDuration(invalid, start)
          clearPresentationReadCursor(context)
          await context?.workspaceMutation?.recordRead(resolved.path)
          return {
            status: 'success',
            output: JSON.stringify(snapshot, null, 2),
            durationMs: Date.now() - start,
          }
        } catch (error) {
          return presentationBridgeError(error, start)
        }
      },
    },
    {
      name: 'office.read_slide',
      description:
        'Read one specified or current slide from the active read-only workspace PowerPoint deck. ' +
        'An explicit index also makes that slide the visible review cursor. Returns title, visible text, speaker notes, shape metadata, current slide, and deck slide count without reading the whole presentation.',
      resumeSafety: 'replay-safe',
      scheduling: { mode: 'sequential', resource: 'office.com' },
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'The same workspace presentation path passed to office.open_presentation.' },
          index: { type: 'number', description: 'Optional 1-based slide index. Omit to read the slide currently visible in PowerPoint.' },
          maxChars: { type: 'number', description: 'Maximum characters for slide text and notes separately. Defaults to 20000.' },
        },
        required: ['path'],
      },
      async execute(input, context): Promise<ToolResult> {
        const start = Date.now()
        const resolved = await resolveWorkspacePresentationPath(input.path, context)
        if (!resolved.ok) return withDuration(resolved.error, start)
        if (input.index !== undefined && (!Number.isInteger(input.index) || Number(input.index) < 1)) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: 'index must be a positive integer when provided.',
            durationMs: Date.now() - start,
          }
        }
        if (!presentationBridge) {
          return {
            status: 'error',
            code: 'OFFICE_BRIDGE_UNAVAILABLE_PERMANENT',
            output: 'The configured Office bridge does not support PowerPoint slide review.',
            durationMs: Date.now() - start,
          }
        }

        try {
          const snapshot = await presentationBridge.readSlide({
            expectedPath: resolved.path,
            index: input.index === undefined ? undefined : Number(input.index),
            maxChars: intInput(input, 'maxChars', 20_000, 500, 100_000),
          })
          const invalid = await validatePresentationSnapshot(snapshot, context, resolved.path)
          if (invalid) return withDuration(invalid, start)
          if (
            !Number.isInteger(snapshot.activeSlideIndex)
            || snapshot.activeSlideIndex < 1
            || snapshot.activeSlideIndex > snapshot.slideCount
          ) {
            return {
              status: 'error',
              code: 'OFFICE_SLIDE_RANGE_PERMANENT',
              output: `PowerPoint returned active slide ${snapshot.activeSlideIndex}; expected 1..${snapshot.slideCount}.`,
              durationMs: Date.now() - start,
            }
          }
          if (
            input.index !== undefined
            && snapshot.activeSlideIndex !== Number(input.index)
          ) {
            return {
              status: 'error',
              code: 'OFFICE_SLIDE_STATE_PERMANENT',
              output:
                `PowerPoint read slide ${input.index}, but the visible review cursor remained on slide ${snapshot.activeSlideIndex}.`,
              durationMs: Date.now() - start,
            }
          }
          if (context?.sessionId) {
            lastPresentationReadBySession.set(context.sessionId, {
              path: resolved.path,
              slideIndex: snapshot.activeSlideIndex,
            })
          }
          await context?.workspaceMutation?.recordRead(resolved.path)
          return {
            status: 'success',
            output: JSON.stringify(snapshot, null, 2),
            durationMs: Date.now() - start,
          }
        } catch (error) {
          return presentationBridgeError(error, start)
        }
      },
    },
    {
      name: 'office.capture_slide',
      description:
        'Render exactly one specified or current slide from the active read-only workspace PowerPoint deck as a PNG image for visual review. ' +
        'Returns slide metadata and attaches only that rendered slide to the next model turn; it never captures another desktop window.',
      resumeSafety: 'replay-safe',
      resumeSafetyForInput(input) {
        return input.index === undefined ? 'replay-risky' : 'replay-safe'
      },
      scheduling: { mode: 'sequential', resource: 'office.com' },
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'The same workspace presentation path passed to office.open_presentation.' },
          index: { type: 'number', description: 'Optional 1-based slide index. Omit to capture the slide currently visible in PowerPoint.' },
          width: { type: 'number', description: 'PNG width in pixels. Defaults to 1600 and is clamped to 640..2560.' },
        },
        required: ['path'],
      },
      async execute(input, context): Promise<ToolResult> {
        const start = Date.now()
        const resolved = await resolveWorkspacePresentationPath(input.path, context)
        if (!resolved.ok) return withDuration(resolved.error, start)
        if (input.index !== undefined && (!Number.isInteger(input.index) || Number(input.index) < 1)) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: 'index must be a positive integer when provided.',
            durationMs: Date.now() - start,
          }
        }
        if (!presentationCaptureBridge) {
          return {
            status: 'error',
            code: 'OFFICE_BRIDGE_UNAVAILABLE_PERMANENT',
            output: 'The configured Office bridge does not support PowerPoint slide capture.',
            durationMs: Date.now() - start,
          }
        }

        const renderDirectory = join(tmpdir(), 'sepilotd-presentation-review')
        const outputPath = join(renderDirectory, `${randomUUID()}.png`)
        try {
          await mkdir(renderDirectory, { recursive: true })
          const snapshot = await presentationCaptureBridge.captureSlide({
            expectedPath: resolved.path,
            outputPath,
            index: input.index === undefined ? undefined : Number(input.index),
            width: intInput(input, 'width', 1_600, 640, 2_560),
          })
          const invalid = await validatePresentationSnapshot(snapshot, context, resolved.path)
          if (invalid) return withDuration(invalid, start)
          if (
            !Number.isInteger(snapshot.activeSlideIndex)
            || snapshot.activeSlideIndex < 1
            || snapshot.activeSlideIndex > snapshot.slideCount
            || !Number.isInteger(snapshot.width)
            || snapshot.width < 1
            || !Number.isInteger(snapshot.height)
            || snapshot.height < 1
          ) {
            return {
              status: 'error',
              code: 'OFFICE_PRESENTATION_INVALID_PERMANENT',
              output: 'PowerPoint returned invalid slide capture metadata.',
              durationMs: Date.now() - start,
            }
          }
          if (
            input.index !== undefined
            && snapshot.activeSlideIndex !== Number(input.index)
          ) {
            return {
              status: 'error',
              code: 'OFFICE_SLIDE_STATE_PERMANENT',
              output:
                `PowerPoint captured slide ${snapshot.activeSlideIndex}; expected visible slide ${input.index}.`,
              durationMs: Date.now() - start,
            }
          }
          const lastRead = context?.sessionId
            ? lastPresentationReadBySession.get(context.sessionId)
            : undefined
          if (context?.sessionId && input.index === undefined && !lastRead) {
            return {
              status: 'error',
              code: 'OFFICE_SLIDE_STATE_PERMANENT',
              output:
                'The current slide has not been verified in this daemon session. Read the current slide again before capture.',
              durationMs: Date.now() - start,
            }
          }
          if (
            lastRead
            && (lastRead.path !== resolved.path || lastRead.slideIndex !== snapshot.activeSlideIndex)
          ) {
            return {
              status: 'error',
              code: 'OFFICE_SLIDE_STATE_PERMANENT',
              output:
                `PowerPoint captured slide ${snapshot.activeSlideIndex}, but the last verified read for this session was slide ${lastRead.slideIndex}. Read the current slide again before capture.`,
              durationMs: Date.now() - start,
            }
          }

          const image = await readFile(outputPath)
          if (image.length === 0 || image.length > 16 * 1024 * 1024) {
            return {
              status: 'error',
              code: 'OFFICE_SLIDE_CAPTURE_LIMIT_PERMANENT',
              output: `Rendered slide PNG size ${image.length} bytes is outside the supported 1..16777216 byte range.`,
              durationMs: Date.now() - start,
            }
          }
          await context?.workspaceMutation?.recordRead(resolved.path)
          return {
            status: 'success',
            output: JSON.stringify(snapshot, null, 2),
            images: [{
              mediaType: 'image/png',
              data: image.toString('base64'),
              displayToClient: true,
            }],
            durationMs: Date.now() - start,
          }
        } catch (error) {
          return presentationBridgeError(error, start)
        } finally {
          clearPresentationReadCursor(context)
          await rm(outputPath, { force: true }).catch(() => undefined)
        }
      },
    },
    {
      name: 'office.read_active',
      description:
        'Read the active Word document text, Excel used range, or PowerPoint slide text from a running Office app. ' +
        'Requires the active document to be saved inside the session workspace unless allowOutsideWorkspace or allowUnsaved is explicitly true.',
      resumeSafety: 'replay-safe',
      scheduling: { mode: 'sequential', resource: 'office.com' },
      inputSchema: {
        type: 'object',
        properties: {
          app: { type: 'string', enum: ['word', 'excel', 'powerpoint'] },
          maxChars: { type: 'number', description: 'Maximum text characters. Defaults to 20000.' },
          maxRows: { type: 'number', description: 'Excel row cap. Defaults to 80.' },
          maxColumns: { type: 'number', description: 'Excel column cap. Defaults to 24.' },
          allowOutsideWorkspace: { type: 'boolean', description: 'Allow reading a saved active document outside cwd after explicit approval.' },
          allowUnsaved: { type: 'boolean', description: 'Allow reading an unsaved active document after explicit approval.' },
        },
        required: ['app'],
      },
      async execute(input, context): Promise<ToolResult> {
        const start = Date.now()
        const app = normalizeApp(input)
        if (!app) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: 'app must be one of: word, excel, powerpoint.',
            durationMs: Date.now() - start,
          }
        }
        try {
          const snapshot = await bridge.readActive({
            app,
            maxChars: intInput(input, 'maxChars', 20_000, 500, 200_000),
            maxRows: intInput(input, 'maxRows', 80, 1, 500),
            maxColumns: intInput(input, 'maxColumns', 24, 1, 80),
          })
          const scopeError = enforceDocumentScope(snapshot, input, context)
          if (scopeError) return withDuration(scopeError, start)
          if (snapshot.path) {
            await context?.workspaceMutation?.recordRead(snapshot.path)
          }
          return {
            status: 'success',
            output: JSON.stringify(snapshot, null, 2),
            durationMs: Date.now() - start,
          }
        } catch (error) {
          return {
            status: 'error',
            code: 'OFFICE_BRIDGE_UNAVAILABLE_PERMANENT',
            output: error instanceof Error ? error.message : String(error),
            durationMs: Date.now() - start,
          }
        }
      },
    },
    {
      name: 'office.preview_edit',
      description:
        'Preview a live edit against the active Office document without modifying it. ' +
        'Use before office.apply_edit or office.replace_selection and show the preview to the user for confirmation.',
      resumeSafety: 'replay-safe',
      scheduling: { mode: 'sequential', resource: 'office.com' },
      inputSchema: {
        type: 'object',
        properties: {
          app: { type: 'string', enum: ['word', 'excel', 'powerpoint'] },
          operation: { type: 'string', enum: ['replace_text', 'set_range_values', 'replace_selection'] },
          findText: { type: 'string' },
          replacement: { type: 'string' },
          occurrence: { type: 'string', enum: ['first', 'all'] },
          sheet: { type: 'string' },
          range: { type: 'string' },
          values: {},
          allowOutsideWorkspace: { type: 'boolean' },
          allowUnsaved: { type: 'boolean' },
        },
        required: ['app', 'operation'],
      },
      async execute(input, context): Promise<ToolResult> {
        const start = Date.now()
        const app = normalizeApp(input)
        const operation = operationFromInput(input)
        if (!app || !operation) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: 'app and operation are required.',
            durationMs: Date.now() - start,
          }
        }
        const invalid = validateEditInput(app, operation, input)
        if (invalid) return withDuration(invalid, start)

        try {
          const snapshot = await readEditScopeSnapshot(bridge, app, operation, input)
          const scopeError = enforceDocumentScope(snapshot, input, context)
          if (scopeError) return withDuration(scopeError, start)
          if (snapshot.path) {
            await context?.workspaceMutation?.recordRead(snapshot.path)
          }

          let preview: Record<string, unknown>
          if (operation === 'replace_selection') {
            preview = selectionPreview(app, snapshot as OfficeSelectionSnapshot, input)
          } else if (operation === 'replace_text') {
            preview = {
              app,
              document: { name: snapshot.name, path: snapshot.path ?? null },
              operation,
              occurrence: occurrenceFromInput(input),
              findText: input.findText,
              replacement: input.replacement,
              matches: countOccurrences(textFromSnapshot(snapshot), String(input.findText)),
              willReplace:
                occurrenceFromInput(input) === 'all'
                  ? countOccurrences(textFromSnapshot(snapshot), String(input.findText))
                  : Math.min(1, countOccurrences(textFromSnapshot(snapshot), String(input.findText))),
            }
          } else {
            preview = {
              app,
              document: { name: snapshot.name, path: snapshot.path ?? null },
              operation,
              sheet:
                typeof input.sheet === 'string'
                  ? input.sheet
                  : (snapshot as OfficeReadSnapshot).activeSheet,
              range: input.range,
              values: input.values,
              currentUsedRange: (snapshot as OfficeReadSnapshot).usedRange,
            }
          }

          const nextTool = operation === 'replace_selection'
            ? 'office.replace_selection'
            : 'office.apply_edit'
          return {
            status: 'success',
            output: JSON.stringify({
              ...preview,
              nextStep:
                `After the user approves this preview, call ${nextTool} with the same arguments plus confirm=true.`,
            }, null, 2),
            durationMs: Date.now() - start,
          }
        } catch (error) {
          return {
            status: 'error',
            code: 'OFFICE_BRIDGE_UNAVAILABLE_PERMANENT',
            output: error instanceof Error ? error.message : String(error),
            durationMs: Date.now() - start,
          }
        }
      },
    },
    {
      name: 'office.apply_edit',
      description:
        'Apply a confirmed live edit to the active Word, Excel, or PowerPoint document. ' +
        'Requires confirm=true, is policy-supervised by default, verifies the active document path before editing, and does not save unless save=true.',
      resumeSafety: 'replay-risky',
      scheduling: { mode: 'sequential', resource: 'office.com' },
      inputSchema: {
        type: 'object',
        properties: {
          app: { type: 'string', enum: ['word', 'excel', 'powerpoint'] },
          operation: { type: 'string', enum: ['replace_text', 'set_range_values', 'replace_selection'] },
          confirm: { type: 'boolean', description: 'Must be true after user approval.' },
          findText: { type: 'string' },
          replacement: { type: 'string' },
          occurrence: { type: 'string', enum: ['first', 'all'] },
          sheet: { type: 'string' },
          range: { type: 'string' },
          values: {},
          save: { type: 'boolean', description: 'Save the Office document after editing. Defaults to false.' },
          allowOutsideWorkspace: { type: 'boolean' },
          allowUnsaved: { type: 'boolean' },
        },
        required: ['app', 'operation', 'confirm'],
      },
      async execute(input, context): Promise<ToolResult> {
        const start = Date.now()
        return applyConfirmedEdit(bridge, input, context, start)
      },
    },
    {
      name: 'office.read_selection',
      description:
        'Read the current selection in the active Word document, Excel workbook, or PowerPoint deck. ' +
        'Use with computer.observe when the user points to visible content or selects a region in the Office UI.',
      resumeSafety: 'replay-safe',
      scheduling: { mode: 'sequential', resource: 'office.com' },
      inputSchema: {
        type: 'object',
        properties: {
          app: { type: 'string', enum: ['word', 'excel', 'powerpoint'] },
          maxChars: { type: 'number', description: 'Maximum text characters. Defaults to 20000.' },
          maxRows: { type: 'number', description: 'Excel selected row cap. Defaults to 80.' },
          maxColumns: { type: 'number', description: 'Excel selected column cap. Defaults to 24.' },
          allowOutsideWorkspace: { type: 'boolean' },
          allowUnsaved: { type: 'boolean' },
        },
        required: ['app'],
      },
      async execute(input, context): Promise<ToolResult> {
        const start = Date.now()
        const app = normalizeApp(input)
        if (!app) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: 'app must be one of: word, excel, powerpoint.',
            durationMs: Date.now() - start,
          }
        }
        try {
          const snapshot = await bridge.readSelection({
            app,
            maxChars: intInput(input, 'maxChars', 20_000, 500, 200_000),
            maxRows: intInput(input, 'maxRows', 80, 1, 500),
            maxColumns: intInput(input, 'maxColumns', 24, 1, 80),
          })
          const scopeError = enforceDocumentScope(snapshot, input, context)
          if (scopeError) return withDuration(scopeError, start)
          if (snapshot.path) {
            await context?.workspaceMutation?.recordRead(snapshot.path)
          }
          return {
            status: 'success',
            output: JSON.stringify(snapshot, null, 2),
            durationMs: Date.now() - start,
          }
        } catch (error) {
          return {
            status: 'error',
            code: 'OFFICE_BRIDGE_UNAVAILABLE_PERMANENT',
            output: error instanceof Error ? error.message : String(error),
            durationMs: Date.now() - start,
          }
        }
      },
    },
    {
      name: 'office.replace_selection',
      description:
        'Replace the current Office selection after user confirmation. Word/PowerPoint use replacement text; Excel uses values for the selected range. ' +
        'Call office.read_selection and show the proposed replacement first, then call this with confirm=true.',
      resumeSafety: 'replay-risky',
      scheduling: { mode: 'sequential', resource: 'office.com' },
      inputSchema: {
        type: 'object',
        properties: {
          app: { type: 'string', enum: ['word', 'excel', 'powerpoint'] },
          confirm: { type: 'boolean', description: 'Must be true after user approval.' },
          replacement: { type: 'string', description: 'Replacement text for Word or PowerPoint.' },
          values: { description: 'Replacement value or 2D array for the selected Excel range.' },
          save: { type: 'boolean', description: 'Save the Office document after editing. Defaults to false.' },
          allowOutsideWorkspace: { type: 'boolean' },
          allowUnsaved: { type: 'boolean' },
        },
        required: ['app', 'confirm'],
      },
      async execute(input, context): Promise<ToolResult> {
        const start = Date.now()
        return applyConfirmedEdit(
          bridge,
          { ...input, operation: 'replace_selection' },
          context,
          start,
        )
      },
    },
  ]
}

function createPowerShellOfficeBridge(): OfficeBridge & PowerPointReviewBridge & PowerPointSlideCaptureBridge {
  return {
    async listOpenDocuments() {
      return runOfficePowerShell('list', {})
    },
    async readActive(input) {
      return runOfficePowerShell('read', input)
    },
    async readSelection(input) {
      return runOfficePowerShell('selection', input)
    },
    async applyEdit(input) {
      return runOfficePowerShell('apply', input)
    },
    async openReadOnly(input) {
      return runOfficePowerShell('presentation-open', input)
    },
    async navigateSlide(input) {
      return runOfficePowerShell('presentation-navigate', input)
    },
    async readSlide(input) {
      return runOfficePowerShell('presentation-read-slide', input)
    },
    async captureSlide(input) {
      return runOfficePowerShell('presentation-capture-slide', input)
    },
  }
}

async function runOfficePowerShell<T>(operation: string, payload: Record<string, unknown>): Promise<T> {
  if (process.platform !== 'win32') {
    throw new Error('Office live bridge is only available on Windows with desktop Microsoft Office.')
  }

  const stdout = await runPowerShellJson(OFFICE_BRIDGE_SCRIPT, payload, operation)
  try {
    return JSON.parse(stdout) as T
  } catch (error) {
    throw new Error(
      `Office bridge returned invalid JSON: ${error instanceof Error ? error.message : String(error)}\n${stdout.slice(0, 2_000)}`,
    )
  }
}

const OFFICE_BRIDGE_TIMEOUT_MS = 45_000
const OFFICE_BRIDGE_STDOUT_LIMIT = 8 * 1024 * 1024
const OFFICE_BRIDGE_STDERR_LIMIT = 1024 * 1024

function runPowerShellJson(
  script: string,
  payload: Record<string, unknown>,
  operation: string,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn('powershell.exe', [
      '-NoProfile',
      '-NonInteractive',
      '-ExecutionPolicy',
      'Bypass',
      '-Command',
      script,
    ], {
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
      env: {
        ...process.env,
        SEPILOTD_OFFICE_OPERATION: operation,
      },
    })

    let stdout = ''
    let stderr = ''
    let settled = false
    const finish = (callback: () => void) => {
      if (settled) return
      settled = true
      clearTimeout(timeout)
      callback()
    }
    const terminateWith = (error: Error) => {
      if (settled) return
      child.kill()
      finish(() => reject(error))
    }
    const timeout = setTimeout(() => {
      terminateWith(
        new Error(
          `[OFFICE_BRIDGE_TIMEOUT_TRANSIENT] Office operation ${operation} exceeded ${OFFICE_BRIDGE_TIMEOUT_MS}ms.`,
        ),
      )
    }, OFFICE_BRIDGE_TIMEOUT_MS)
    timeout.unref?.()
    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')
    child.stdout.on('data', (chunk) => {
      stdout += String(chunk)
      if (Buffer.byteLength(stdout, 'utf8') > OFFICE_BRIDGE_STDOUT_LIMIT) {
        terminateWith(
          new Error(
            `[OFFICE_BRIDGE_OUTPUT_LIMIT_PERMANENT] Office operation ${operation} exceeded the ${OFFICE_BRIDGE_STDOUT_LIMIT}-byte stdout limit.`,
          ),
        )
      }
    })
    child.stderr.on('data', (chunk) => {
      stderr += String(chunk)
      if (Buffer.byteLength(stderr, 'utf8') > OFFICE_BRIDGE_STDERR_LIMIT) {
        terminateWith(
          new Error(
            `[OFFICE_BRIDGE_OUTPUT_LIMIT_PERMANENT] Office operation ${operation} exceeded the ${OFFICE_BRIDGE_STDERR_LIMIT}-byte stderr limit.`,
          ),
        )
      }
    })
    child.on('error', (error) => finish(() => reject(error)))
    child.on('close', (code) => {
      if (settled) return
      if (code === 0) {
        finish(() => resolve(stdout.trim()))
      } else {
        finish(() => reject(new Error(stderr.trim() || `powershell.exe exited with code ${code}`)))
      }
    })
    child.stdin.end(JSON.stringify(payload))
  })
}

const OFFICE_BRIDGE_SCRIPT = String.raw`
$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding $false
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$raw = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($raw)) {
  $payload = [pscustomobject]@{}
} else {
  $payload = $raw | ConvertFrom-Json
}

function Out-Json($obj) {
  $obj | ConvertTo-Json -Depth 32 -Compress
}

function Get-ComObject($progId) {
  try {
    return [System.Runtime.InteropServices.Marshal]::GetActiveObject($progId)
  } catch {
    return $null
  }
}

function Get-DocPath($doc) {
  try {
    $full = [string]$doc.FullName
    if ($full -and $full.Trim().Length -gt 0) { return $full }
  } catch {}
  return $null
}

function Get-PropBool($obj, $name) {
  try { return [bool]$obj.$name } catch { return $false }
}

function Count-Occurrences($text, $needle) {
  if ([string]::IsNullOrEmpty($needle)) { return 0 }
  $count = 0
  $offset = 0
  while ($true) {
    $idx = $text.IndexOf($needle, $offset, [System.StringComparison]::Ordinal)
    if ($idx -lt 0) { return $count }
    $count += 1
    $offset = $idx + $needle.Length
  }
}

function Truncate-Text($text, $maxChars) {
  if ($null -eq $text) { return @('', $false) }
  $s = [string]$text
  if ($s.Length -le $maxChars) { return @($s, $false) }
  return @($s.Substring(0, $maxChars), $true)
}

function Convert-RangeRows($range, $maxRows, $maxColumns) {
  $rowCount = [Math]::Min([int]$range.Rows.Count, $maxRows)
  $columnCount = [Math]::Min([int]$range.Columns.Count, $maxColumns)
  $rows = @()
  for ($r = 1; $r -le $rowCount; $r++) {
    $row = @()
    for ($c = 1; $c -le $columnCount; $c++) {
      $row += [string]$range.Cells.Item($r, $c).Text
    }
    $rows += ,$row
  }
  return @{
    rows = $rows
    truncated = (($range.Rows.Count -gt $rowCount) -or ($range.Columns.Count -gt $columnCount))
  }
}

function Get-ActiveWord {
  $app = Get-ComObject 'Word.Application'
  if ($null -eq $app -or $app.Documents.Count -lt 1) { throw 'No active Word document.' }
  return $app.ActiveDocument
}

function Get-ActiveExcel {
  $app = Get-ComObject 'Excel.Application'
  if ($null -eq $app -or $null -eq $app.ActiveWorkbook) { throw 'No active Excel workbook.' }
  return @{ app = $app; workbook = $app.ActiveWorkbook; sheet = $app.ActiveSheet }
}

function Get-ActivePowerPoint {
  $app = Get-ComObject 'PowerPoint.Application'
  if ($null -eq $app -or $null -eq $app.ActivePresentation) { throw 'No active PowerPoint presentation.' }
  return $app.ActivePresentation
}

function Get-OrCreatePowerPoint {
  $app = Get-ComObject 'PowerPoint.Application'
  if ($null -eq $app) {
    $app = New-Object -ComObject PowerPoint.Application
  }
  try { $app.Visible = -1 } catch {}
  # Disable presentation macros for automation-opened files. .pptx cannot carry
  # VBA, and the Node boundary rejects all other extensions as defense in depth.
  try { $app.AutomationSecurity = 3 } catch {}
  return $app
}

function Assert-ReadOnlyPresentation($pres) {
  if (-not (Get-PropBool $pres 'ReadOnly')) {
    throw '[OFFICE_PRESENTATION_WRITABLE_USER] The active presentation is writable. Close it and reopen it through office.open_presentation.'
  }
}

function Activate-Presentation($pres) {
  try {
    if ($pres.Windows.Count -gt 0) {
      $pres.Windows.Item(1).Activate()
    }
  } catch {}
}

function Get-CurrentSlideIndex($app, $pres) {
  if ($null -eq $app -or $null -eq $app.ActivePresentation) {
    throw '[OFFICE_PRESENTATION_NOT_ACTIVE_USER] No active PowerPoint presentation.'
  }
  $activePath = Get-DocPath $app.ActivePresentation
  Assert-ExpectedPath $activePath (Get-DocPath $pres)
  if ($null -eq $app.ActiveWindow -or $null -eq $app.ActiveWindow.View) {
    throw '[OFFICE_PRESENTATION_WINDOW_UNAVAILABLE_USER] PowerPoint has no active presentation window.'
  }
  try {
    $index = [int]$app.ActiveWindow.View.Slide.SlideIndex
  } catch {
    throw '[OFFICE_PRESENTATION_WINDOW_UNAVAILABLE_USER] PowerPoint could not determine the visible slide.'
  }
  if ($index -lt 1 -or $index -gt [int]$pres.Slides.Count) {
    throw "[OFFICE_SLIDE_RANGE_PERMANENT] PowerPoint returned invalid active slide $index."
  }
  return $index
}

function Assert-SlideIndex($index, $count) {
  if ($index -lt 1 -or $index -gt $count) {
    throw "[OFFICE_SLIDE_RANGE_PERMANENT] Slide $index is outside the valid range 1..$count."
  }
}

function Presentation-Summary($app, $pres, $slideIndex) {
  return [pscustomobject]@{
    app = 'powerpoint'
    kind = 'powerpoint'
    name = [string]$pres.Name
    path = Get-DocPath $pres
    readOnly = Get-PropBool $pres 'ReadOnly'
    slideCount = [int]$pres.Slides.Count
    slideIndex = [int]$slideIndex
  }
}

function Open-PresentationReadOnly($payload) {
  $path = [string]$payload.path
  if ([string]::IsNullOrWhiteSpace($path)) { throw 'path is required.' }
  $app = Get-OrCreatePowerPoint
  $pres = $null
  $openedNow = $false

  foreach ($candidate in @($app.Presentations)) {
    $candidatePath = Get-DocPath $candidate
    if ([string]::Equals($candidatePath, $path, [System.StringComparison]::OrdinalIgnoreCase)) {
      $pres = $candidate
      break
    }
  }

  if ($null -ne $pres) {
    Assert-ReadOnlyPresentation $pres
  } else {
    # MsoTriState: ReadOnly=true(-1), Untitled=false(0), WithWindow=true(-1).
    $pres = $app.Presentations.Open($path, -1, 0, -1)
    $openedNow = $true
    Assert-ReadOnlyPresentation $pres
  }

  Activate-Presentation $pres
  if ($pres.Slides.Count -lt 1) {
    throw '[OFFICE_PRESENTATION_INVALID_PERMANENT] The presentation has no slides.'
  }
  # Re-opening an already active review deck is idempotent: preserve the
  # reviewer's current page instead of silently jumping back to slide 1.
  if ($openedNow) { try { $app.ActiveWindow.View.GotoSlide(1) } catch {} }
  $current = Get-CurrentSlideIndex $app $pres
  return Presentation-Summary $app $pres $current
}

function Navigate-PresentationSlide($payload) {
  $app = Get-ComObject 'PowerPoint.Application'
  if ($null -eq $app -or $null -eq $app.ActivePresentation) { throw 'No active PowerPoint presentation.' }
  $pres = $app.ActivePresentation
  Assert-ExpectedPath (Get-DocPath $pres) $payload.expectedPath
  Assert-ReadOnlyPresentation $pres
  Activate-Presentation $pres

  $count = [int]$pres.Slides.Count
  $current = Get-CurrentSlideIndex $app $pres
  $direction = [string]$payload.direction
  if ($direction -eq 'next') {
    $target = $current + 1
  } elseif ($direction -eq 'previous') {
    $target = $current - 1
  } elseif ($direction -eq 'index') {
    $target = [int]$payload.index
  } else {
    throw "Unsupported slide navigation direction: $direction"
  }
  Assert-SlideIndex $target $count
  $app.ActiveWindow.View.GotoSlide($target)
  return Presentation-Summary $app $pres $target
}

function Read-PresentationSlide($payload) {
  $app = Get-ComObject 'PowerPoint.Application'
  if ($null -eq $app -or $null -eq $app.ActivePresentation) { throw 'No active PowerPoint presentation.' }
  $pres = $app.ActivePresentation
  Assert-ExpectedPath (Get-DocPath $pres) $payload.expectedPath
  Assert-ReadOnlyPresentation $pres
  Activate-Presentation $pres

  $count = [int]$pres.Slides.Count
  $activeSlideIndex = Get-CurrentSlideIndex $app $pres
  $target = $activeSlideIndex
  if ($null -ne $payload.index) { $target = [int]$payload.index }
  Assert-SlideIndex $target $count
  # An explicit read target becomes the visible review cursor. This keeps a
  # following capture without an index bound to the exact slide just read and
  # avoids maintaining Office-specific target state in the generic agent loop.
  if ($target -ne $activeSlideIndex) {
    $app.ActiveWindow.View.GotoSlide($target)
    $activeSlideIndex = $target
  }
  $maxChars = if ($payload.maxChars) { [int]$payload.maxChars } else { 20000 }
  $slide = $pres.Slides.Item($target)

  $title = $null
  try {
    $titleShape = $slide.Shapes.Title
    if ($titleShape.HasTextFrame -and $titleShape.TextFrame.HasText) {
      $title = [string]$titleShape.TextFrame.TextRange.Text
    }
  } catch {}

  $texts = @()
  $shapeMetadata = @()
  $shapeTextTruncated = $false
  $shapeLimit = 200
  $shapeCount = 0
  foreach ($shape in @($slide.Shapes)) {
    if ($shapeCount -ge $shapeLimit) {
      $shapeTextTruncated = $true
      break
    }
    $shapeCount += 1
    $hasText = $false
    $shapeText = $null
    try {
      if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
        $hasText = $true
        $shapeText = [string]$shape.TextFrame.TextRange.Text
      }
    } catch {}
    $shapePair = Truncate-Text $shapeText ([Math]::Min($maxChars, 4000))
    if ($shapePair[1]) { $shapeTextTruncated = $true }
    if ($hasText) { $texts += $shapePair[0] }
    $alternativeText = $null
    try { $alternativeText = [string]$shape.AlternativeText } catch {}
    $alternativePair = Truncate-Text $alternativeText 2000
    if ($alternativePair[1]) { $shapeTextTruncated = $true }
    $hasChart = $false
    $hasTable = $false
    try { $hasChart = [bool]$shape.HasChart } catch {}
    try { $hasTable = [bool]$shape.HasTable } catch {}
    $shapeMetadata += [pscustomobject]@{
      index = [int]$shape.ZOrderPosition
      name = [string]$shape.Name
      shapeType = [int]$shape.Type
      left = [Math]::Round([double]$shape.Left, 2)
      top = [Math]::Round([double]$shape.Top, 2)
      width = [Math]::Round([double]$shape.Width, 2)
      height = [Math]::Round([double]$shape.Height, 2)
      hasText = $hasText
      text = if ($hasText) { $shapePair[0] } else { $null }
      alternativeText = $alternativePair[0]
      hasChart = $hasChart
      hasTable = $hasTable
    }
  }

  $noteTexts = @()
  $noteShapeCount = 0
  try {
    foreach ($noteShape in @($slide.NotesPage.Shapes)) {
      if ($noteShapeCount -ge 100) {
        $shapeTextTruncated = $true
        break
      }
      $noteShapeCount += 1
      $isBody = $false
      try { $isBody = ([int]$noteShape.PlaceholderFormat.Type -eq 2) } catch {}
      if (-not $isBody) { continue }
      try {
        if ($noteShape.HasTextFrame -and $noteShape.TextFrame.HasText) {
          $noteShapePair = Truncate-Text ([string]$noteShape.TextFrame.TextRange.Text) ([Math]::Min($maxChars, 4000))
          $noteTexts += $noteShapePair[0]
          if ($noteShapePair[1]) { $shapeTextTruncated = $true }
        }
      } catch {}
    }
  } catch {}

  $textPair = Truncate-Text ($texts -join [Environment]::NewLine) $maxChars
  $notesPair = Truncate-Text ($noteTexts -join [Environment]::NewLine) $maxChars
  return [pscustomobject]@{
    app = 'powerpoint'
    kind = 'powerpoint'
    name = [string]$pres.Name
    path = Get-DocPath $pres
    readOnly = Get-PropBool $pres 'ReadOnly'
    slideCount = $count
    slideIndex = $target
    activeSlideIndex = $activeSlideIndex
    title = $title
    text = $textPair[0]
    notes = $notesPair[0]
    shapes = $shapeMetadata
    truncated = ([bool]$textPair[1] -or [bool]$notesPair[1] -or $shapeTextTruncated)
  }
}

function Capture-PresentationSlide($payload) {
  $app = Get-ComObject 'PowerPoint.Application'
  if ($null -eq $app -or $null -eq $app.ActivePresentation) { throw 'No active PowerPoint presentation.' }
  $pres = $app.ActivePresentation
  Assert-ExpectedPath (Get-DocPath $pres) $payload.expectedPath
  Assert-ReadOnlyPresentation $pres

  $count = [int]$pres.Slides.Count
  $activeSlideIndex = Get-CurrentSlideIndex $app $pres
  $target = $activeSlideIndex
  if ($null -ne $payload.index) { $target = [int]$payload.index }
  Assert-SlideIndex $target $count

  $outputPath = [string]$payload.outputPath
  if ([string]::IsNullOrWhiteSpace($outputPath)) { throw 'outputPath is required.' }
  $width = [int]$payload.width
  if ($width -lt 640 -or $width -gt 2560) {
    throw '[OFFICE_SLIDE_CAPTURE_LIMIT_PERMANENT] Capture width must be within 640..2560 pixels.'
  }
  $slideWidth = [double]$pres.PageSetup.SlideWidth
  $slideHeight = [double]$pres.PageSetup.SlideHeight
  if ($slideWidth -le 0 -or $slideHeight -le 0) {
    throw '[OFFICE_PRESENTATION_INVALID_PERMANENT] PowerPoint returned invalid slide dimensions.'
  }
  $height = [Math]::Max(1, [Math]::Min(4096, [Math]::Round($width * $slideHeight / $slideWidth)))
  $slide = $pres.Slides.Item($target)
  $slide.Export($outputPath, 'PNG', $width, $height)
  if (-not [System.IO.File]::Exists($outputPath)) {
    throw '[OFFICE_SLIDE_CAPTURE_FAILED_PERMANENT] PowerPoint did not create the rendered slide image.'
  }

  return [pscustomobject]@{
    app = 'powerpoint'
    kind = 'powerpoint'
    name = [string]$pres.Name
    path = Get-DocPath $pres
    readOnly = Get-PropBool $pres 'ReadOnly'
    slideCount = $count
    slideIndex = $target
    activeSlideIndex = $activeSlideIndex
    width = $width
    height = [int]$height
  }
}

function List-Documents {
  $items = @()

  $word = Get-ComObject 'Word.Application'
  if ($null -ne $word) {
    $activePath = $null
    try { $activePath = Get-DocPath $word.ActiveDocument } catch {}
    foreach ($doc in @($word.Documents)) {
      $path = Get-DocPath $doc
      $items += [pscustomobject]@{
        app = 'word'
        name = [string]$doc.Name
        path = $path
        saved = Get-PropBool $doc 'Saved'
        readOnly = Get-PropBool $doc 'ReadOnly'
        active = ($path -eq $activePath -and $path -ne $null)
      }
    }
  }

  $excel = Get-ComObject 'Excel.Application'
  if ($null -ne $excel) {
    $activePath = $null
    try { $activePath = Get-DocPath $excel.ActiveWorkbook } catch {}
    foreach ($book in @($excel.Workbooks)) {
      $path = Get-DocPath $book
      $items += [pscustomobject]@{
        app = 'excel'
        name = [string]$book.Name
        path = $path
        saved = Get-PropBool $book 'Saved'
        readOnly = Get-PropBool $book 'ReadOnly'
        active = ($path -eq $activePath -and $path -ne $null)
      }
    }
  }

  $ppt = Get-ComObject 'PowerPoint.Application'
  if ($null -ne $ppt) {
    $activePath = $null
    try { $activePath = Get-DocPath $ppt.ActivePresentation } catch {}
    foreach ($pres in @($ppt.Presentations)) {
      $path = Get-DocPath $pres
      $items += [pscustomobject]@{
        app = 'powerpoint'
        name = [string]$pres.Name
        path = $path
        saved = Get-PropBool $pres 'Saved'
        readOnly = Get-PropBool $pres 'ReadOnly'
        active = ($path -eq $activePath -and $path -ne $null)
      }
    }
  }

  return @{ documents = $items }
}

function Read-Active($payload) {
  $app = [string]$payload.app
  $maxChars = if ($payload.maxChars) { [int]$payload.maxChars } else { 20000 }
  $maxRows = if ($payload.maxRows) { [int]$payload.maxRows } else { 80 }
  $maxColumns = if ($payload.maxColumns) { [int]$payload.maxColumns } else { 24 }

  if ($app -eq 'word') {
    $doc = Get-ActiveWord
    $pair = Truncate-Text $doc.Content.Text $maxChars
    return @{
      app = 'word'
      kind = 'word'
      name = [string]$doc.Name
      path = Get-DocPath $doc
      saved = Get-PropBool $doc 'Saved'
      text = $pair[0]
      truncated = $pair[1]
    }
  }

  if ($app -eq 'excel') {
    $active = Get-ActiveExcel
    $book = $active.workbook
    $sheet = $active.sheet
    $used = $sheet.UsedRange
    $rangeData = Convert-RangeRows $used $maxRows $maxColumns
    return @{
      app = 'excel'
      kind = 'excel'
      name = [string]$book.Name
      path = Get-DocPath $book
      saved = Get-PropBool $book 'Saved'
      activeSheet = [string]$sheet.Name
      usedRange = [string]$used.Address($false, $false)
      rows = $rangeData.rows
      truncated = $rangeData.truncated
    }
  }

  if ($app -eq 'powerpoint') {
    $pres = Get-ActivePowerPoint
    $slides = @()
    foreach ($slide in @($pres.Slides)) {
      $texts = @()
      foreach ($shape in @($slide.Shapes)) {
        try {
          if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
            $texts += [string]$shape.TextFrame.TextRange.Text
          }
        } catch {}
      }
      $slides += [pscustomobject]@{ index = [int]$slide.SlideIndex; text = ($texts -join [Environment]::NewLine) }
    }
    return @{
      app = 'powerpoint'
      kind = 'powerpoint'
      name = [string]$pres.Name
      path = Get-DocPath $pres
      saved = Get-PropBool $pres 'Saved'
      slides = $slides
    }
  }

  throw "Unsupported Office app: $app"
}

function Read-Selection($payload) {
  $app = [string]$payload.app
  $maxChars = if ($payload.maxChars) { [int]$payload.maxChars } else { 20000 }
  $maxRows = if ($payload.maxRows) { [int]$payload.maxRows } else { 80 }
  $maxColumns = if ($payload.maxColumns) { [int]$payload.maxColumns } else { 24 }

  if ($app -eq 'word') {
    $word = Get-ComObject 'Word.Application'
    if ($null -eq $word -or $word.Documents.Count -lt 1) { throw 'No active Word document.' }
    $doc = $word.ActiveDocument
    $selection = $word.Selection
    $pair = Truncate-Text $selection.Text $maxChars
    return @{
      app = 'word'
      kind = 'word'
      name = [string]$doc.Name
      path = Get-DocPath $doc
      saved = Get-PropBool $doc 'Saved'
      text = $pair[0]
      selectionType = [string]$selection.Type
      range = "$($selection.Start)..$($selection.End)"
    }
  }

  if ($app -eq 'excel') {
    $active = Get-ActiveExcel
    $book = $active.workbook
    $range = $active.app.Selection
    $rangeData = Convert-RangeRows $range $maxRows $maxColumns
    return @{
      app = 'excel'
      kind = 'excel'
      name = [string]$book.Name
      path = Get-DocPath $book
      saved = Get-PropBool $book 'Saved'
      activeSheet = [string]$range.Worksheet.Name
      range = [string]$range.Address($false, $false)
      rows = $rangeData.rows
      selectionType = 'range'
    }
  }

  if ($app -eq 'powerpoint') {
    $ppt = Get-ComObject 'PowerPoint.Application'
    if ($null -eq $ppt -or $null -eq $ppt.ActivePresentation) { throw 'No active PowerPoint presentation.' }
    $pres = $ppt.ActivePresentation
    $selection = $ppt.ActiveWindow.Selection
    $texts = @()
    $shapeCount = 0
    $slideIndex = $null
    try { $slideIndex = [int]$ppt.ActiveWindow.View.Slide.SlideIndex } catch {}
    try {
      if ($selection.TextRange) {
        $texts += [string]$selection.TextRange.Text
      }
    } catch {}
    try {
      foreach ($shape in @($selection.ShapeRange)) {
        $shapeCount += 1
        try {
          if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
            $texts += [string]$shape.TextFrame.TextRange.Text
          }
        } catch {}
      }
    } catch {}
    $pair = Truncate-Text ($texts -join [Environment]::NewLine) $maxChars
    return @{
      app = 'powerpoint'
      kind = 'powerpoint'
      name = [string]$pres.Name
      path = Get-DocPath $pres
      saved = Get-PropBool $pres 'Saved'
      text = $pair[0]
      slideIndex = $slideIndex
      shapeCount = $shapeCount
      selectionType = [string]$selection.Type
    }
  }

  throw "Unsupported Office app: $app"
}

function Assert-ExpectedPath($actual, $expected) {
  if ([string]::IsNullOrWhiteSpace($expected)) { return }
  if (-not [string]::Equals([string]$actual, [string]$expected, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "[OFFICE_ACTIVE_DOCUMENT_CHANGED_USER] Active Office document changed. Expected '$expected', found '$actual'."
  }
}

function Apply-Edit($payload) {
  $app = [string]$payload.app
  $operation = [string]$payload.operation
  $save = ($payload.save -eq $true)

  if ($operation -eq 'replace_text') {
    $findText = [string]$payload.findText
    $replacement = [string]$payload.replacement
    if ([string]::IsNullOrEmpty($findText)) { throw 'findText is required.' }
    $replaceAll = ([string]$payload.occurrence) -eq 'all'

    if ($app -eq 'word') {
      $doc = Get-ActiveWord
      Assert-ExpectedPath (Get-DocPath $doc) $payload.expectedPath
      $before = [string]$doc.Content.Text
      $matches = Count-Occurrences $before $findText
      if ($matches -lt 1) {
        return @{ app = 'word'; name = [string]$doc.Name; path = Get-DocPath $doc; operation = $operation; changed = 0; saved = $false }
      }
      $range = $doc.Content
      $find = $range.Find
      $find.ClearFormatting() | Out-Null
      $find.Replacement.ClearFormatting() | Out-Null
      $replaceCode = if ($replaceAll) { 2 } else { 1 }
      $ok = $find.Execute($findText, $false, $false, $false, $false, $false, $true, 1, $false, $replacement, $replaceCode)
      if ($save) { $doc.Save() | Out-Null }
      return @{
        app = 'word'
        name = [string]$doc.Name
        path = Get-DocPath $doc
        operation = $operation
        changed = if ($replaceAll) { $matches } elseif ($ok) { 1 } else { 0 }
        saved = $save
      }
    }

    if ($app -eq 'powerpoint') {
      $pres = Get-ActivePowerPoint
      Assert-ExpectedPath (Get-DocPath $pres) $payload.expectedPath
      $changed = 0
      foreach ($slide in @($pres.Slides)) {
        foreach ($shape in @($slide.Shapes)) {
          if (-not $shape.HasTextFrame) { continue }
          if (-not $shape.TextFrame.HasText) { continue }
          $old = [string]$shape.TextFrame.TextRange.Text
          $matches = Count-Occurrences $old $findText
          if ($matches -lt 1) { continue }
          if ($replaceAll) {
            $shape.TextFrame.TextRange.Text = $old.Replace($findText, $replacement)
            $changed += $matches
          } else {
            $idx = $old.IndexOf($findText, [System.StringComparison]::Ordinal)
            if ($idx -ge 0) {
              $shape.TextFrame.TextRange.Text = $old.Substring(0, $idx) + $replacement + $old.Substring($idx + $findText.Length)
              $changed += 1
              break
            }
          }
        }
        if ((-not $replaceAll) -and $changed -gt 0) { break }
      }
      if ($save) { $pres.Save() | Out-Null }
      return @{
        app = 'powerpoint'
        name = [string]$pres.Name
        path = Get-DocPath $pres
        operation = $operation
        changed = $changed
        saved = $save
      }
    }
  }

  if ($operation -eq 'replace_selection') {
    if ($app -eq 'word') {
      $word = Get-ComObject 'Word.Application'
      if ($null -eq $word -or $word.Documents.Count -lt 1) { throw 'No active Word document.' }
      $doc = $word.ActiveDocument
      Assert-ExpectedPath (Get-DocPath $doc) $payload.expectedPath
      $selection = $word.Selection
      $selection.Text = [string]$payload.replacement
      if ($save) { $doc.Save() | Out-Null }
      return @{
        app = 'word'
        name = [string]$doc.Name
        path = Get-DocPath $doc
        operation = $operation
        changed = 1
        saved = $save
      }
    }

    if ($app -eq 'powerpoint') {
      $ppt = Get-ComObject 'PowerPoint.Application'
      if ($null -eq $ppt -or $null -eq $ppt.ActivePresentation) { throw 'No active PowerPoint presentation.' }
      $pres = $ppt.ActivePresentation
      Assert-ExpectedPath (Get-DocPath $pres) $payload.expectedPath
      $selection = $ppt.ActiveWindow.Selection
      $changed = 0
      try {
        if ($selection.TextRange) {
          $selection.TextRange.Text = [string]$payload.replacement
          $changed = 1
        }
      } catch {}
      if ($changed -eq 0) {
        try {
          foreach ($shape in @($selection.ShapeRange)) {
            if ($shape.HasTextFrame) {
              $shape.TextFrame.TextRange.Text = [string]$payload.replacement
              $changed += 1
            }
          }
        } catch {}
      }
      if ($changed -eq 0) { throw 'No editable PowerPoint text selection.' }
      if ($save) { $pres.Save() | Out-Null }
      return @{
        app = 'powerpoint'
        name = [string]$pres.Name
        path = Get-DocPath $pres
        operation = $operation
        changed = $changed
        saved = $save
      }
    }

    if ($app -eq 'excel') {
      $active = Get-ActiveExcel
      $book = $active.workbook
      Assert-ExpectedPath (Get-DocPath $book) $payload.expectedPath
      $range = $active.app.Selection
      $values = $payload.values
      $changed = 0
      if ($values -is [System.Array]) {
        for ($r = 0; $r -lt $values.Count; $r++) {
          $row = $values[$r]
          if ($row -isnot [System.Array]) { $row = @($row) }
          for ($c = 0; $c -lt $row.Count; $c++) {
            $range.Cells.Item($r + 1, $c + 1).Value2 = $row[$c]
            $changed += 1
          }
        }
      } else {
        $range.Value2 = $values
        $changed = 1
      }
      if ($save) { $book.Save() | Out-Null }
      return @{
        app = 'excel'
        name = [string]$book.Name
        path = Get-DocPath $book
        operation = $operation
        changed = $changed
        saved = $save
        target = @{ sheet = [string]$range.Worksheet.Name; range = [string]$range.Address($false, $false) }
      }
    }
  }

  if ($operation -eq 'set_range_values' -and $app -eq 'excel') {
    $active = Get-ActiveExcel
    $book = $active.workbook
    Assert-ExpectedPath (Get-DocPath $book) $payload.expectedPath
    $sheet = if ([string]::IsNullOrWhiteSpace([string]$payload.sheet)) { $active.sheet } else { $book.Worksheets.Item([string]$payload.sheet) }
    $range = $sheet.Range([string]$payload.range)
    $values = $payload.values
    $changed = 0
    if ($values -is [System.Array]) {
      for ($r = 0; $r -lt $values.Count; $r++) {
        $row = $values[$r]
        if ($row -isnot [System.Array]) { $row = @($row) }
        for ($c = 0; $c -lt $row.Count; $c++) {
          $range.Cells.Item($r + 1, $c + 1).Value2 = $row[$c]
          $changed += 1
        }
      }
    } else {
      $range.Value2 = $values
      $changed = 1
    }
    if ($save) { $book.Save() | Out-Null }
    return @{
      app = 'excel'
      name = [string]$book.Name
      path = Get-DocPath $book
      operation = $operation
      changed = $changed
      saved = $save
      target = @{ sheet = [string]$sheet.Name; range = [string]$range.Address($false, $false) }
    }
  }

  throw "Unsupported Office edit: app=$app operation=$operation"
}

switch ($env:SEPILOTD_OFFICE_OPERATION) {
  'list' { Out-Json (List-Documents); break }
  'read' { Out-Json (Read-Active $payload); break }
  'selection' { Out-Json (Read-Selection $payload); break }
  'apply' { Out-Json (Apply-Edit $payload); break }
  'presentation-open' { Out-Json (Open-PresentationReadOnly $payload); break }
  'presentation-navigate' { Out-Json (Navigate-PresentationSlide $payload); break }
  'presentation-read-slide' { Out-Json (Read-PresentationSlide $payload); break }
  'presentation-capture-slide' { Out-Json (Capture-PresentationSlide $payload); break }
  default { throw "Unsupported Office operation: $env:SEPILOTD_OFFICE_OPERATION" }
}
`
