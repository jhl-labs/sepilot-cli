import { useInput } from 'ink'

interface KeybindingHandlers {
  onPageUp?: () => void
  onPageDown?: () => void
  onCtrlP?: () => void
  onCtrlN?: () => void
  onCtrlShiftLeft?: () => void
  onCtrlShiftRight?: () => void
  onCtrlC?: () => void
  onCtrlL?: () => void
  onCtrlO?: () => void
  onCtrlS?: () => void
  onCtrlT?: () => void
  onCtrlF?: () => void
  onCtrlG?: () => void
  onCtrlA?: () => void
  onCtrlY?: () => void
  onCtrlB?: () => void
  onCtrlR?: () => void
  onShiftTab?: () => void
  onEscape?: () => void
  onApprove?: () => void
  onApproveSession?: () => void
  onApproveAlways?: () => void
  onDeny?: () => void
  // Ctrl+O terminal transcript mode handlers. When `copyViewActive` is true,
  // ONLY close handling fires; normal-mode handlers are skipped so opening
  // transcript copy mode doesn't accidentally trigger pickers, autonomy
  // cycling, etc.
  // Kept on the same `useInput` so the raw-mode + stdin-listener registration
  // stays single-source (previously a separate `useInput` toggled via
  // `isActive` raced with this one and silently dropped keys on some
  // terminals).
  copyViewActive?: boolean
  onCopyClose?: () => void
}

export function useKeybindings(handlers: KeybindingHandlers, enabled = true) {
  useInput((input, key) => {
    // Terminal transcript mode owns the keymap exclusively while it's up.
    if (handlers.copyViewActive) {
      if (key.escape || (input === 'o' && key.ctrl)) {
        handlers.onCopyClose?.()
        return
      }
      return
    }

    if (key.pageUp) handlers.onPageUp?.()
    if (key.pageDown) handlers.onPageDown?.()
    if (input === 'p' && key.ctrl) handlers.onCtrlP?.()
    if (input === 'n' && key.ctrl) handlers.onCtrlN?.()
    if (key.leftArrow && key.ctrl && key.shift) handlers.onCtrlShiftLeft?.()
    if (key.rightArrow && key.ctrl && key.shift) handlers.onCtrlShiftRight?.()
    if ((input === 'c' && key.ctrl) || input === '\u0003') handlers.onCtrlC?.()
    if (input === 'l' && key.ctrl) handlers.onCtrlL?.()
    if (input === 'o' && key.ctrl) handlers.onCtrlO?.()
    if (input === 's' && key.ctrl) handlers.onCtrlS?.()
    if (input === 't' && key.ctrl) handlers.onCtrlT?.()
    if (input === 'f' && key.ctrl) handlers.onCtrlF?.()
    if (input === 'g' && key.ctrl) handlers.onCtrlG?.()
    if (input === 'a' && key.ctrl) handlers.onCtrlA?.()
    if (input === 'y' && key.ctrl) handlers.onCtrlY?.()
    if (input === 'b' && key.ctrl) handlers.onCtrlB?.()
    if (input === 'r' && key.ctrl) handlers.onCtrlR?.()
    if (key.tab && key.shift) handlers.onShiftTab?.()
    if (key.escape) handlers.onEscape?.()
    if (!key.ctrl && input.toLowerCase() === 'y') handlers.onApprove?.()
    if (!key.ctrl && input.toLowerCase() === 's') handlers.onApproveSession?.()
    if (!key.ctrl && input.toLowerCase() === 'a') handlers.onApproveAlways?.()
    if (!key.ctrl && input.toLowerCase() === 'n') handlers.onDeny?.()
  }, { isActive: enabled })
}
