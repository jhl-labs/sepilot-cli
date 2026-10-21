const KOREAN_TERSE_MAP_RESULT =
  /^(?:[\p{L}\p{N}._-]+\s+){0,5}(?:지도|맵)\s*(?:검색\s*)?(?:결과|화면)[.!?？]?$/iu
const ENGLISH_TERSE_MAP_RESULT =
  /^(?:[\p{L}\p{N}._-]+\s+){0,5}maps?\s+(?:search\s+)?(?:results?|view)[.!?]?$/iu

export function isTerseVisibleWebSurfaceRequest(input: string): boolean {
  const normalized = input.trim().replace(/\s+/g, ' ')
  if (!normalized || normalized.length > 120) return false
  return KOREAN_TERSE_MAP_RESULT.test(normalized) || ENGLISH_TERSE_MAP_RESULT.test(normalized)
}
