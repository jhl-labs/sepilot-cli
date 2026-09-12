interface CalculateListPanelHeightOptions {
  itemCount: number
  reservedRows: number
  maxVisibleItems: number
  minimumHeight: number
  emptyStateRows?: number
  extraRows?: number
}

export interface VisibleWindow {
  start: number
  end: number
  clampedIndex: number
}

export function fitInlinePanelHeight(
  desiredHeight: number,
  availableRows: number,
  minimumHeight: number,
  extraRows = 0,
): number {
  const availableHeight = Math.max(0, availableRows - extraRows)

  if (desiredHeight < minimumHeight || availableHeight < minimumHeight) {
    return 0
  }

  return Math.min(desiredHeight, availableHeight)
}

export function calculateRenderedPanelRows(
  height: number,
  extraRows = 0,
): number {
  if (height <= 0) {
    return 0
  }

  return height + extraRows
}

export function calculateListViewportCapacity(
  availableRows: number,
  reservedRows: number,
  preferredMaxVisibleItems: number,
): number {
  if (availableRows <= reservedRows) {
    return 0
  }

  return Math.max(
    0,
    Math.min(preferredMaxVisibleItems, availableRows - reservedRows),
  )
}

export function calculateListPanelHeight({
  itemCount,
  reservedRows,
  maxVisibleItems,
  minimumHeight,
  emptyStateRows = 1,
  extraRows = 0,
}: CalculateListPanelHeightOptions): number {
  const visibleRows = itemCount > 0
    ? Math.min(itemCount, Math.max(1, maxVisibleItems))
    : emptyStateRows

  return Math.max(
    minimumHeight,
    reservedRows + extraRows + visibleRows,
  )
}

export function calculateVisibleWindow(
  itemCount: number,
  selectedIndex: number,
  maxVisibleItems: number,
): VisibleWindow {
  if (itemCount <= 0 || maxVisibleItems <= 0) {
    return {
      start: 0,
      end: 0,
      clampedIndex: 0,
    }
  }

  const clampedIndex = Math.max(0, Math.min(selectedIndex, itemCount - 1))
  if (itemCount <= maxVisibleItems) {
    return {
      start: 0,
      end: itemCount,
      clampedIndex,
    }
  }

  const start = Math.max(
    0,
    Math.min(
      clampedIndex - Math.floor(maxVisibleItems / 2),
      itemCount - maxVisibleItems,
    ),
  )

  return {
    start,
    end: start + maxVisibleItems,
    clampedIndex,
  }
}
