import { loadCliState, saveCliState } from '../tui/cli-state.js'

const MAX_MODEL_MRU = 10

/**
 * Records a provider/model target (e.g. `openai/gpt-5.4`) as most-recently-used
 * in `~/.sepilotd/cli-state.json`. All file IO failures are swallowed so the
 * model picker never crashes on a broken/unwritable state file.
 */
export async function recordModelUse(target: string): Promise<void> {
  const trimmed = target.trim()
  if (!trimmed) return
  try {
    const state = await loadCliState()
    const nextMru = [trimmed, ...(state.modelMru ?? []).filter((entry) => entry !== trimmed)].slice(
      0,
      MAX_MODEL_MRU,
    )
    await saveCliState({ ...state, modelMru: nextMru })
  } catch {
    // Swallow: MRU tracking must never crash the picker.
  }
}

/**
 * Reads the most-recently-used provider/model targets, most-recent-first.
 * Returns an empty array on any file IO failure.
 */
export async function readModelMru(): Promise<string[]> {
  try {
    const state = await loadCliState()
    return state.modelMru ?? []
  } catch {
    return []
  }
}
