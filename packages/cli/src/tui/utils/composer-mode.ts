export type ComposerInputMode = 'chat' | 'shell'

export interface ComposerInputPresentation {
  mode: ComposerInputMode
  editorValue: string
  editorCursorOffset: number
}

/**
 * Keep the leading bang as the transport-level shell shortcut while presenting
 * the remainder as a real shell editor. This lets submission/history continue
 * to use the existing `!command` contract without making users edit a hidden
 * implementation prefix after entering shell mode.
 */
export function deriveComposerInputPresentation(
  value: string,
  cursorOffset = value.length,
): ComposerInputPresentation {
  if (!value.startsWith('!')) {
    return {
      mode: 'chat',
      editorValue: value,
      editorCursorOffset: Math.max(0, Math.min(value.length, cursorOffset)),
    }
  }

  const editorValue = value.slice(1)
  return {
    mode: 'shell',
    editorValue,
    editorCursorOffset: Math.max(0, Math.min(editorValue.length, cursorOffset - 1)),
  }
}
