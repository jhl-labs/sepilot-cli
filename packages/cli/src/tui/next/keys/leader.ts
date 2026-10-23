import type { CliLocale } from '../../../utils/locale.js'
import { fitStatusSegments, STATUS_SEPARATOR } from '../runtime/status-layout.js'

export interface LeaderState {
  armed: boolean
  armedAt: number | null
}

export const IDLE_LEADER_STATE: LeaderState = { armed: false, armedAt: null }
export const LEADER_TIMEOUT_MS = 500

export type LeaderResult =
  | { state: LeaderState; action: 'none' }
  | { state: LeaderState; action: 'arm' }
  | { state: LeaderState; action: 'disarm' }
  | { state: LeaderState; action: 'run'; commandId: string }
  | { state: LeaderState; action: 'literal'; text: string }

export interface LeaderInput {
  key: string
  ctrl: boolean
  now: number
}

export interface LeaderOptions {
  leaderKey: string
  bindings: Map<string, string>
}

function leaderLetter(leaderKey: string): string | undefined {
  const match = /^ctrl\+([a-z])$/i.exec(leaderKey)
  return match?.[1]
}

function isLeaderPress(input: LeaderInput, leaderKey: string): boolean {
  const letter = leaderLetter(leaderKey)
  return letter !== undefined && input.ctrl && input.key.toLowerCase() === letter.toLowerCase()
}

function controlCharacter(leaderKey: string): string {
  const letter = leaderLetter(leaderKey)
  if (!letter) return ''
  return String.fromCharCode(letter.toLowerCase().charCodeAt(0) - 96)
}

export function reduceLeaderKey(
  state: LeaderState,
  input: LeaderInput,
  options: LeaderOptions,
): LeaderResult {
  const leaderPressed = isLeaderPress(input, options.leaderKey)

  if (!state.armed) {
    return leaderPressed
      ? { state: { armed: true, armedAt: input.now }, action: 'arm' }
      : { state, action: 'none' }
  }

  if (leaderPressed) {
    return {
      state: IDLE_LEADER_STATE,
      action: 'literal',
      text: controlCharacter(options.leaderKey),
    }
  }

  const armedAt = state.armedAt ?? input.now
  if (input.now - armedAt > LEADER_TIMEOUT_MS) {
    return { state: IDLE_LEADER_STATE, action: 'disarm' }
  }

  const commandId = options.bindings.get(input.key)
  return commandId
    ? { state: IDLE_LEADER_STATE, action: 'run', commandId }
    : { state: IDLE_LEADER_STATE, action: 'disarm' }
}

export function leaderHintItems(
  bindings: Map<string, string>,
  titles: Map<string, string>,
  width?: number,
  locale: CliLocale = 'en',
): string[] {
  const parts: Array<{ id: string; text: string; priority: number }> = []
  const seen = new Set<string>()
  const shortTitles: Record<string, [string, string]> = {
    'shell.help': ['help', '도움말'],
    'model.pick': ['model', '모델'],
    'bot.pick': ['bot', '봇'],
    'settings.open': ['settings', '설정'],
    'session.new': ['new', '새 세션'],
    'session.resume': ['resume', '이어서'],
  }

  for (const [key, commandId] of bindings) {
    if (seen.has(commandId)) continue
    const title = width == null ? titles.get(commandId)
      : shortTitles[commandId]?.[locale === 'ko' ? 1 : 0] ?? titles.get(commandId)
    if (title) {
      seen.add(commandId)
      parts.push({ id: commandId, text: `${key} ${title}`, priority: commandId === 'shell.help' ? 100 : 50 })
    }
  }

  return (width == null ? parts : fitStatusSegments(parts, width)).map(({ text }) => text)
}

export function leaderHint(
  bindings: Map<string, string>,
  titles: Map<string, string>,
  width?: number,
  locale: CliLocale = 'en',
): string {
  return leaderHintItems(bindings, titles, width, locale).join(STATUS_SEPARATOR)
}
