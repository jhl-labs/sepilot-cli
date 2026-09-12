// The inline shell writes to host scrollback. It deliberately leaves alternate
// screen and mouse reporting disabled so native select, copy, scroll and find
// continue to work. Bracketed paste remains enabled for safe paste handling.
export const INLINE_ENTER_SEQUENCE = '\u001b[?2004h'
export const INLINE_EXIT_SEQUENCE = '\u001b[?2004l'
