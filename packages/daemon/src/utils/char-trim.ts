export function trimTrailingChars(value: string, chars: string): string {
  let end = value.length
  while (end > 0 && chars.includes(value[end - 1]!)) end -= 1
  return end === value.length ? value : value.slice(0, end)
}

export function trimLeadingChars(value: string, chars: string): string {
  let start = 0
  while (start < value.length && chars.includes(value[start]!)) start += 1
  return start === 0 ? value : value.slice(start)
}

export function trimChars(value: string, leading: string, trailing: string = leading): string {
  return trimTrailingChars(trimLeadingChars(value, leading), trailing)
}
