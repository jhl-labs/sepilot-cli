export function isReturnKey(input: string, key: { return?: boolean; name?: unknown; sequence?: unknown }): boolean {
  const keyName = typeof key.name === 'string' ? key.name.toLowerCase() : ''
  return Boolean(key.return)
    || input === '\r'
    || input === '\n'
    || key.sequence === '\r'
    || key.sequence === '\n'
    || keyName === 'return'
    || keyName === 'enter'
}
