import { readdir } from 'node:fs/promises'
import { join } from 'node:path'
import type { FilePickerItem } from '../components/FilePicker.js'

export type FilePickerFetchResult =
  | { ok: true; items: FilePickerItem[]; cancelled?: false }
  | { ok: true; cancelled: true }
  | { ok: false; error: string }

export async function loadFilePickerItems(opts: {
  dir: string
  cancelled: () => boolean
}): Promise<FilePickerFetchResult> {
  try {
    const entries = await readdir(opts.dir, { withFileTypes: true })
    if (opts.cancelled()) return { ok: true, cancelled: true }
    const items: FilePickerItem[] = entries
      .sort((left, right) => {
        if (left.isDirectory() && !right.isDirectory()) return -1
        if (!left.isDirectory() && right.isDirectory()) return 1
        return left.name.localeCompare(right.name)
      })
      .map((entry) => ({
        absolutePath: join(opts.dir, entry.name),
        label: `${entry.name}${entry.isDirectory() ? '/' : ''}`,
        isDirectory: entry.isDirectory(),
      }))
    return { ok: true, items }
  } catch (error) {
    if (opts.cancelled()) return { ok: true, cancelled: true }
    return {
      ok: false,
      error: error instanceof Error ? error.message : 'Failed to load files.',
    }
  }
}
