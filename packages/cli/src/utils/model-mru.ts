import { loadCliState, saveCliState } from '../tui/cli-state.js'

const MAX_MODEL_MRU = 10

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
  }
}

export async function readModelMru(): Promise<string[]> {
  try {
    const state = await loadCliState()
    return state.modelMru ?? []
  } catch {
    return []
  }
}
