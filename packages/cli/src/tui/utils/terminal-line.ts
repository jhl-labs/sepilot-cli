export const ERASE_TO_END_OF_LINE = '\x1b[K'

export function clearLineTail(text: string): string {
  return `${text}${ERASE_TO_END_OF_LINE}`
}
