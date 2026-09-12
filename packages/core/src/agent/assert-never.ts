export function assertNever(value: never, context = 'unhandled variant'): never {
  throw new Error(`${context}: ${JSON.stringify(value)}`)
}
