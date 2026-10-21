export function closeTruncatedJson(input: string): string {
  const stack: string[] = []
  let inString = false
  let escape = false
  for (let i = 0; i < input.length; i += 1) {
    const c = input[i]
    if (escape) {
      escape = false
      continue
    }
    if (inString) {
      if (c === '\\') escape = true
      else if (c === '"') inString = false
      continue
    }
    if (c === '"') inString = true
    else if (c === '{') stack.push('}')
    else if (c === '[') stack.push(']')
    else if (c === '}' || c === ']') stack.pop()
  }
  let result = input
  if (inString) result += '"'
  result = result.replace(/,\s*$/, '')
  for (let i = stack.length - 1; i >= 0; i -= 1) {
    result += stack[i]
  }
  return result
}
