const FENCE_LINE_PATTERN = /^```[a-zA-Z0-9_+\-.]*$/gm
const FENCE_CLOSE = '```'

export interface MessageChunkOptions {
  limit: number
  minSplitRatio?: number
}

function collectFenceLines(chunk: string): string[] {
  FENCE_LINE_PATTERN.lastIndex = 0
  const matches: string[] = []
  let match: RegExpExecArray | null
  while ((match = FENCE_LINE_PATTERN.exec(chunk)) !== null) {
    matches.push(match[0])
  }
  return matches
}

export function splitChannelMessage(
  text: string,
  options: MessageChunkOptions,
): string[] {
  const limit = Math.max(1, options.limit)
  if (text.length <= limit) {
    return [text]
  }

  const chunks: string[] = []
  let remaining = text
  let pendingFenceOpener = ''

  while (remaining.length > limit) {
    const minUsefulSplit = Math.floor(limit * (options.minSplitRatio ?? 0.6))
    let splitAt = remaining.lastIndexOf('\n', limit)
    if (splitAt < minUsefulSplit) {
      splitAt = remaining.lastIndexOf(' ', limit)
    }
    if (splitAt < minUsefulSplit) {
      splitAt = limit
    }

    let chunk = remaining.slice(0, splitAt).trimEnd()
    if (pendingFenceOpener) {
      chunk = pendingFenceOpener + '\n' + chunk
    }

    const fences = collectFenceLines(chunk)
    if (fences.length % 2 === 1) {
      pendingFenceOpener = fences[fences.length - 1] ?? FENCE_CLOSE
      chunk = chunk + '\n' + FENCE_CLOSE
    } else {
      pendingFenceOpener = ''
    }

    chunks.push(chunk || remaining.slice(0, limit))
    remaining = remaining.slice(splitAt).trimStart()
  }

  if (remaining) {
    chunks.push(pendingFenceOpener
      ? pendingFenceOpener + '\n' + remaining
      : remaining)
  }

  return chunks
}
