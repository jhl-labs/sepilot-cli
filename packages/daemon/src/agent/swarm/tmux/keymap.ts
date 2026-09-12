const KEY_ALIASES: Readonly<Record<string, string>> = {
  'c-c': 'C-c',
  'ctrl-c': 'C-c',
  'c-d': 'C-d',
  'ctrl-d': 'C-d',
  escape: 'Escape',
  esc: 'Escape',
  up: 'Up',
  down: 'Down',
  left: 'Left',
  right: 'Right',
  tab: 'Tab',
  btab: 'BTab',
  'shift-tab': 'BTab',
  bspace: 'BSpace',
  backspace: 'BSpace',
  home: 'Home',
  end: 'End',
  pageup: 'PageUp',
  pagedown: 'PageDown',
  enter: 'Enter',
  space: 'Space',
}

export function resolveTmuxKeyName(name: string): string | null {
  return KEY_ALIASES[name.trim().toLowerCase()] ?? null
}
