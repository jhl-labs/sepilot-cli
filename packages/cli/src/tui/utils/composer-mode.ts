export type ComposerInputMode = 'chat' | 'shell'

export interface ComposerInputPresentation {
  mode: ComposerInputMode
  editorValue: string
  editorCursorOffset: number
}

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
