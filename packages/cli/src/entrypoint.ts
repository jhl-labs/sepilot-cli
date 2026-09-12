import { readFileSync } from 'node:fs'

const ROOT_TUI_VALUE_OPTIONS = ['--url', '--model', '--provider', '--session'] as const
const ROOT_TUI_BOOLEAN_OPTIONS = new Set(['--resume', '--json', '--interactive'])
const ROOT_PROMPT_VALUE_OPTIONS = ['--prompt', '-p'] as const
const ROOT_PROMPT_CONTEXT_VALUE_OPTIONS = [
  '--url',
  '--model',
  '--provider',
  '--session',
  '--max-tokens',
  '--output-format',
] as const
const ROOT_EXIT_OPTIONS = new Set(['--help', '-h', '--version', '-V'])

export function loadCliVersion(packageJsonUrl: URL = new URL('../package.json', import.meta.url)): string {
  // A build-time override (set by `@sepilotd/bundle`'s `bun build --compile`,
  // which can't read `package.json` from inside Bun's virtual filesystem) wins.
  const fromEnv = process.env.SEPILOT_VERSION?.trim()
  if (fromEnv) return fromEnv
  try {
    const raw = readFileSync(packageJsonUrl, 'utf8')
    const parsed = JSON.parse(raw) as { version?: string }
    return parsed.version?.trim() || '0.0.0'
  } catch {
    return '0.0.0'
  }
}

export function readOptionValue(argv: string[], flag: string): string | undefined {
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index]
    if (token === flag) return argv[index + 1]
    if (token.startsWith(`${flag}=`)) return token.slice(flag.length + 1)
  }

  return undefined
}

export function shouldLaunchTui(argv: string[]): boolean {
  if (argv.some((token) => ROOT_EXIT_OPTIONS.has(token))) return false

  let skipNextValue = false

  for (const token of argv) {
    if (skipNextValue) {
      skipNextValue = false
      continue
    }

    if (ROOT_TUI_VALUE_OPTIONS.includes(token as (typeof ROOT_TUI_VALUE_OPTIONS)[number])) {
      skipNextValue = true
      continue
    }

    if (ROOT_TUI_VALUE_OPTIONS.some((flag) => token.startsWith(`${flag}=`))) {
      continue
    }

    if (ROOT_TUI_BOOLEAN_OPTIONS.has(token)) {
      continue
    }

    return false
  }

  return true
}

export function shouldRunRootPrompt(argv: string[]): boolean {
  if (argv.some((token) => ROOT_EXIT_OPTIONS.has(token))) return false

  let skipNextValue = false
  let hasPromptOption = false

  for (const token of argv) {
    if (skipNextValue) {
      skipNextValue = false
      continue
    }

    if (ROOT_PROMPT_VALUE_OPTIONS.includes(token as (typeof ROOT_PROMPT_VALUE_OPTIONS)[number])) {
      hasPromptOption = true
      skipNextValue = true
      continue
    }

    if (ROOT_PROMPT_VALUE_OPTIONS.some((flag) => token.startsWith(`${flag}=`))) {
      hasPromptOption = true
      continue
    }

    if (
      ROOT_PROMPT_CONTEXT_VALUE_OPTIONS.includes(
        token as (typeof ROOT_PROMPT_CONTEXT_VALUE_OPTIONS)[number],
      )
    ) {
      skipNextValue = true
      continue
    }

    if (ROOT_PROMPT_CONTEXT_VALUE_OPTIONS.some((flag) => token.startsWith(`${flag}=`))) {
      continue
    }

    if (ROOT_TUI_BOOLEAN_OPTIONS.has(token)) {
      continue
    }

    return false
  }

  return hasPromptOption
}
