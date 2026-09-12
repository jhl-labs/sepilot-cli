import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import {
  isThemeId,
  type ThemeId,
} from './theme.js'

interface CliState {
  lastSessionId?: string
  recentSessionIds?: string[]
  projectBindings?: Record<string, string>
  themeId?: ThemeId
  modelMru?: string[]
  /** End-of-turn recap toggle. Undefined means enabled (default on). */
  recapEnabled?: boolean
}

const cliStatePath = join(homedir(), '.sepilotd', 'cli-state.json')
const MAX_RECENT_SESSION_IDS = 5
const MAX_PROJECT_BINDINGS = 20
const MAX_MODEL_MRU = 10

function normalizeCliState(state: CliState): CliState {
  const recentSessionIds = Array.from(new Set(state.recentSessionIds ?? []))
    .filter(Boolean)
    .slice(0, MAX_RECENT_SESSION_IDS)
  const projectBindings = Object.fromEntries(
    Object.entries(state.projectBindings ?? {})
      .filter(([rootDir, projectId]) => Boolean(rootDir) && Boolean(projectId))
      .slice(-MAX_PROJECT_BINDINGS)
      .map(([rootDir, projectId]) => [resolve(rootDir), projectId]),
  )
  const modelMru = Array.from(new Set((state.modelMru ?? []).filter(Boolean)))
    .slice(0, MAX_MODEL_MRU)

  return {
    lastSessionId: state.lastSessionId,
    recentSessionIds,
    projectBindings,
    themeId: isThemeId(state.themeId ?? '') ? state.themeId : undefined,
    modelMru,
    recapEnabled: typeof state.recapEnabled === 'boolean' ? state.recapEnabled : undefined,
  }
}

export async function loadCliState(): Promise<CliState> {
  try {
    const raw = await readFile(cliStatePath, 'utf-8')
    return normalizeCliState(JSON.parse(raw) as CliState)
  } catch {
    return normalizeCliState({})
  }
}

export async function saveCliState(state: CliState): Promise<void> {
  const nextState = normalizeCliState(state)
  await mkdir(dirname(cliStatePath), { recursive: true })
  await writeFile(cliStatePath, JSON.stringify(nextState, null, 2), 'utf-8')
}

export async function recordSessionAccess(sessionId: string): Promise<CliState> {
  const currentState = await loadCliState()
  const nextState = normalizeCliState({
    ...currentState,
    lastSessionId: sessionId,
    recentSessionIds: [
      sessionId,
      ...(currentState.recentSessionIds ?? []).filter((id) => id !== sessionId),
    ],
  })
  await saveCliState(nextState)
  return nextState
}

export async function removeSessionAccess(sessionId: string): Promise<CliState> {
  const currentState = await loadCliState()
  const recentSessionIds = (currentState.recentSessionIds ?? [])
    .filter((id) => id !== sessionId)
  const nextState = normalizeCliState({
    ...currentState,
    lastSessionId: currentState.lastSessionId === sessionId
      ? recentSessionIds[0]
      : currentState.lastSessionId,
    recentSessionIds,
  })
  await saveCliState(nextState)
  return nextState
}

export async function recordProjectBinding(
  rootDir: string,
  projectId: string,
): Promise<CliState> {
  const currentState = await loadCliState()
  const normalizedRootDir = resolve(rootDir)
  const nextState = normalizeCliState({
    ...currentState,
    projectBindings: {
      ...Object.fromEntries(
        Object.entries(currentState.projectBindings ?? {})
          .filter(([existingRootDir]) => existingRootDir !== normalizedRootDir),
      ),
      [normalizedRootDir]: projectId,
    },
  })
  await saveCliState(nextState)
  return nextState
}

export async function clearProjectBinding(rootDir: string): Promise<CliState> {
  const currentState = await loadCliState()
  const normalizedRootDir = resolve(rootDir)
  const nextState = normalizeCliState({
    ...currentState,
    projectBindings: Object.fromEntries(
      Object.entries(currentState.projectBindings ?? {})
        .filter(([existingRootDir]) => existingRootDir !== normalizedRootDir),
    ),
  })
  await saveCliState(nextState)
  return nextState
}

export async function recordRecapPreference(recapEnabled: boolean): Promise<CliState> {
  const currentState = await loadCliState()
  const nextState = normalizeCliState({
    ...currentState,
    recapEnabled,
  })
  await saveCliState(nextState)
  return nextState
}

export async function recordThemePreference(themeId: ThemeId): Promise<CliState> {
  const currentState = await loadCliState()
  const nextState = normalizeCliState({
    ...currentState,
    themeId,
  })
  await saveCliState(nextState)
  return nextState
}
