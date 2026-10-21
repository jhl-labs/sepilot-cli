export type TerminalImageProtocol = 'kitty' | 'iterm2' | 'sixel' | 'none'

export type TerminalImageMime = 'image/jpeg' | 'image/png'

function isWezTerm(env: NodeJS.ProcessEnv): boolean {
  return (env.TERM_PROGRAM ?? '').toLowerCase() === 'wezterm'
}

export function detectTerminalImageProtocol(env: NodeJS.ProcessEnv): TerminalImageProtocol {
  if (env.SEPILOT_TUI_IMAGES === '0') return 'none'
  const term = (env.TERM ?? '').toLowerCase()
  const program = (env.TERM_PROGRAM ?? '').toLowerCase()
  if (program === 'wezterm') return 'kitty'
  if (program === 'iterm.app') return 'iterm2'
  if (env.KITTY_WINDOW_ID || term === 'xterm-kitty' || term === 'xterm-ghostty' || program === 'ghostty') {
    return 'kitty'
  }
  if (term.includes('sixel')) return 'sixel'
  return 'none'
}

export function terminalAcceptsImageMime(env: NodeJS.ProcessEnv, mime: TerminalImageMime): boolean {
  const protocol = detectTerminalImageProtocol(env)
  if (protocol === 'iterm2') return true
  if (protocol === 'kitty') return mime === 'image/png' || isWezTerm(env)
  return false
}

const KITTY_CHUNK = 4096

export function encodeTerminalImage(
  protocol: Exclude<TerminalImageProtocol, 'none'>,
  imageBase64: string,
  size: { width: number; height: number },
): string {
  if (protocol === 'iterm2') {
    return `\u001b]1337;File=inline=1;width=auto;preserveAspectRatio=1:${imageBase64}\u0007`
  }
  if (protocol === 'kitty') {
    if (!Number.isInteger(size.width) || !Number.isInteger(size.height) || size.width <= 0 || size.height <= 0) {
      return ''
    }
    const chunks: string[] = []
    for (let i = 0; i < imageBase64.length; i += KITTY_CHUNK) chunks.push(imageBase64.slice(i, i + KITTY_CHUNK))
    return chunks
      .map((chunk, index) => {
        const more = index < chunks.length - 1 ? 1 : 0
        const control = index === 0 ? `a=T,f=100,s=${size.width},v=${size.height},m=${more}` : `m=${more}`
        return `\u001b_G${control};${chunk}\u001b\\`
      })
      .join('')
  }
  return ''
}
