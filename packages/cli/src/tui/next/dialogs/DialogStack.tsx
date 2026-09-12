import type { ReactElement } from 'react'

export interface DialogEntry {
  id: string
  node: ReactElement
}

export interface DialogStackProps {
  entries: DialogEntry[]
}

/** Render only the top entry so hidden dialogs cannot consume keyboard input. */
export function DialogStack({ entries }: DialogStackProps): ReactElement | null {
  return entries.at(-1)?.node ?? null
}
