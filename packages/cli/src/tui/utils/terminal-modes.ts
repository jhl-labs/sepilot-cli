// Alternate screen plus mouse reporting modes used by the TUI.
//
// 1002 captures button-drag reports in terminals that require it before they
// translate touchpad/touch scroll gestures into application events. 1007 asks
// xterm-compatible terminals to route wheel gestures to the alternate screen
// application instead of host scrollback. 2004 wraps paste payloads so the
// composer can summarize them without losing the original text. Unsupported
// modes are ignored.
export const TUI_ENTER_SEQUENCE = [
  '\x1b[?1049h',
  '\x1b[H',
  '\x1b[?1000h',
  '\x1b[?1002h',
  '\x1b[?1006h',
  '\x1b[?1007h',
  '\x1b[?2004h',
].join('')

export const TUI_EXIT_SEQUENCE = [
  '\x1b[?2004l',
  '\x1b[?1007l',
  '\x1b[?1006l',
  '\x1b[?1002l',
  '\x1b[?1000l',
  '\x1b[?1049l',
].join('')
