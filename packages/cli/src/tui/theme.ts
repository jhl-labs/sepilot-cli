const themePresets = {
  dark: {
    primary: '#5B9BD5',
    secondary: '#6C757D',
    success: '#28A745',
    error: '#DC3545',
    warning: '#FFC107',
    pending: '#FFB000',
    info: '#17A2B8',
    muted: '#6C757D',
    text: '#FFFFFF',
    dimText: '#888888',
    border: '#444444',
    toolBg: '#1E1E1E',
    headerBg: '#2D2D2D',
    inputBorder: '#555555',
  },
  light: {
    primary: '#D97706',
    secondary: '#9A7B4F',
    success: '#3E8E5B',
    error: '#D05C52',
    warning: '#F2C14E',
    pending: '#D38B1F',
    info: '#2D8C9F',
    muted: '#B19064',
    text: '#FFF7ED',
    dimText: '#D6C2A9',
    border: '#8E6C47',
    toolBg: '#33261C',
    headerBg: '#423224',
    inputBorder: '#9B7A57',
  },
  nord: {
    primary: '#88C0D0',
    secondary: '#4C566A',
    success: '#A3BE8C',
    error: '#BF616A',
    warning: '#EBCB8B',
    pending: '#D08770',
    info: '#81A1C1',
    muted: '#616E88',
    text: '#ECEFF4',
    dimText: '#7B88A1',
    border: '#3B4252',
    toolBg: '#2E3440',
    headerBg: '#3B4252',
    inputBorder: '#4C566A',
  },
  gruvbox: {
    primary: '#FE8019',
    secondary: '#928374',
    success: '#B8BB26',
    error: '#FB4934',
    warning: '#FABD2F',
    pending: '#FE8019',
    info: '#83A598',
    muted: '#928374',
    text: '#EBDBB2',
    dimText: '#A89984',
    border: '#504945',
    toolBg: '#282828',
    headerBg: '#3C3836',
    inputBorder: '#665C54',
  },
  catppuccin: {
    primary: '#89B4FA',
    secondary: '#6C7086',
    success: '#A6E3A1',
    error: '#F38BA8',
    warning: '#F9E2AF',
    pending: '#FAB387',
    info: '#94E2D5',
    muted: '#7F849C',
    text: '#CDD6F4',
    dimText: '#9399B2',
    border: '#45475A',
    toolBg: '#1E1E2E',
    headerBg: '#313244',
    inputBorder: '#585B70',
  },
  tokyonight: {
    primary: '#7AA2F7',
    secondary: '#565F89',
    success: '#9ECE6A',
    error: '#F7768E',
    warning: '#E0AF68',
    pending: '#FF9E64',
    info: '#7DCFFF',
    muted: '#737AA2',
    text: '#C0CAF5',
    dimText: '#737AA2',
    border: '#292E42',
    toolBg: '#1A1B26',
    headerBg: '#24283B',
    inputBorder: '#3B4261',
  },
} as const

export type ThemeId = keyof typeof themePresets
export type ThemePalette = typeof themePresets.dark

export const themeOptions = [
  {
    id: 'dark',
    label: 'Dark',
    description: 'Blue-gray default palette for long coding sessions.',
  },
  {
    id: 'light',
    label: 'Light',
    description: 'Warm sand accent palette with brighter contrast.',
  },
  {
    id: 'nord',
    label: 'Nord',
    description: 'Cool arctic blues with soft frost accents.',
  },
  {
    id: 'gruvbox',
    label: 'Gruvbox',
    description: 'Retro warm earth tones on a dark base.',
  },
  {
    id: 'catppuccin',
    label: 'Catppuccin',
    description: 'Pastel Mocha palette, gentle on the eyes.',
  },
  {
    id: 'tokyonight',
    label: 'Tokyo Night',
    description: 'Muted neon blues over deep midnight.',
  },
] as const satisfies ReadonlyArray<{
  id: ThemeId
  label: string
  description: string
}>

const themeIds = themeOptions.map((option) => option.id)

let activeThemeId: ThemeId = 'dark'

export const colors: ThemePalette = {
  ...themePresets[activeThemeId],
}

export function isThemeId(value: string): value is ThemeId {
  return themeIds.includes(value as ThemeId)
}

export function getThemeOption(themeId: ThemeId) {
  return themeOptions.find((option) => option.id === themeId) ?? themeOptions[0]
}

export function getActiveThemeId(): ThemeId {
  return activeThemeId
}

export function setActiveTheme(themeId: ThemeId): ThemeId {
  activeThemeId = themeId
  Object.assign(colors, themePresets[themeId])
  return activeThemeId
}

export function resolveThemeId(
  query: string,
  currentThemeId = activeThemeId,
): ThemeId | null {
  const normalized = query.trim().toLowerCase()

  if (!normalized || ['toggle', 'next'].includes(normalized)) {
    const currentIndex = themeIds.indexOf(currentThemeId)
    return themeIds[(currentIndex + 1) % themeIds.length]
  }
  if (normalized === 'prev') {
    const currentIndex = themeIds.indexOf(currentThemeId)
    return themeIds[(currentIndex - 1 + themeIds.length) % themeIds.length]
  }
  if (['default', 'system'].includes(normalized)) {
    return 'dark'
  }

  const directMatch = themeOptions.find((option) => option.id === normalized)
  if (directMatch) {
    return directMatch.id
  }

  const prefixMatch = themeOptions.find((option) => (
    option.id.startsWith(normalized) || option.label.toLowerCase().startsWith(normalized)
  ))
  return prefixMatch?.id ?? null
}

export const symbols = {
  user: '\u25CF',       // ●
  assistant: '\u25CF',  // ●
  success: '\u2713',    // ✓
  error: '\u2717',      // ✗
  pending: '\u23F8',    // ⏸
  thinking: '\u25D0',   // ◐
  collapsed: '\u25B6',  // ▶
  expanded: '\u25BC',   // ▼
  separator: '\u2502',  // │
} as const
