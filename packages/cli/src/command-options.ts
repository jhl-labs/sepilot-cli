import type { Command } from 'commander'

export function mergeActionOptionsWithGlobals(actionCommand: Command): void {
  Object.assign(actionCommand.opts(), actionCommand.optsWithGlobals())
}
