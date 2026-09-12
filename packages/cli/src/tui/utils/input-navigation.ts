export function selectPreviousListIndex(currentIndex: number): number {
  return Math.max(0, currentIndex - 1)
}

export function selectNextListIndex(
  currentIndex: number,
  itemCount: number,
): number {
  if (itemCount <= 0) {
    return 0
  }

  return Math.min(itemCount - 1, currentIndex + 1)
}
