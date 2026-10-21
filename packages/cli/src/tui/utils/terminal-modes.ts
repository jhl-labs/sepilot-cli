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
