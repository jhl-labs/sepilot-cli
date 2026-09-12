const MAX_ENTRIES = 20

let recent: string[] = []

export function recordAttachment(path: string): void {
  recent = [path, ...recent.filter((entry) => entry !== path)].slice(0, MAX_ENTRIES)
}

export function getRecentAttachments(): readonly string[] {
  return recent
}

export function __resetAttachmentHistoryForTest(): void {
  recent = []
}
