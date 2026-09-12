import { highlight } from 'cli-highlight'

export function highlightCode(code: string, language?: string): string {
  try {
    return highlight(code, { language: language || 'plaintext', ignoreIllegals: true })
  } catch {
    return code
  }
}
