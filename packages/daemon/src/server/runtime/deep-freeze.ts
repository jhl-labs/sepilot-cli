/**
 * Recursively freeze an object so any mutation throws in strict mode.
 * Skips non-enumerable properties and prototype chains.
 */
export function deepFreeze<T>(obj: T): Readonly<T> {
  if (obj === null || typeof obj !== 'object') return obj

  Object.freeze(obj)
  for (const val of Object.values(obj)) {
    if (val !== null && typeof val === 'object' && !Object.isFrozen(val)) {
      deepFreeze(val)
    }
  }
  return obj
}